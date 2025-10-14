from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from timm.models.swin_transformer import SwinTransformerBlock


@dataclass
class ModelConfig:
    mel_bins: int = 128
    num_frames: int = 512  # 512, 1024
    patch_size: int = 16
    frequency_stride: int = 16
    time_stride: int = 16
    input_channels: int = 1
    encoder_mlp_ratio: int = 4
    encoder_embed_dim: int = 768
    encoder_num_heads: int = 16
    encoder_depth: int = 12
    decoder_embed_dim: int = 512
    decoder_num_heads: int = 16
    decoder_depth: int = 8
    num_labels: int = 2  # Downstream classification
    mask_ratio: float = 0.5
    classification_strategy: str = "global_pool"  # global_pool, cls_token
    decoder_attention: str = "global"  # global, local
    normalise_patches: bool = True  # Normalise patches before reconstruction

    @property
    def num_patches(self):
        return self.mel_bins * self.num_frames // self.patch_size**2

    @property
    def num_patches_vertical(self):
        return self.mel_bins // self.patch_size

    @property
    def num_patches_horizontal(self):
        return self.num_frames // self.patch_size


class PatchGenerator(nn.Module):
    """
    This class manually extracts patches from `input_values` and flattens them.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.frequency_stride = config.frequency_stride
        self.time_stride = config.time_stride

    def __call__(self, input_values: torch.Tensor) -> torch.Tensor:
        patches = input_values.unfold(1, self.patch_size, self.frequency_stride).unfold(
            2, self.patch_size, self.time_stride
        )

        # Reshape patches and flatten
        batch_size, h_patches, w_patches, patch_height, patch_width = patches.shape
        patches = patches.reshape(batch_size, h_patches * w_patches, -1)

        return patches


class PatchEmbed(nn.Module):
    """
    Spectrogram to Patch Embeddings. Defaults to non overlapping patches.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.input_channels = config.input_channels
        self.encoder_embed_dim = config.encoder_embed_dim
        self.patch_size = config.patch_size
        self.frequency_stride = config.frequency_stride
        self.time_stride = config.time_stride

        self.projection = nn.Conv2d(
            config.input_channels,
            config.encoder_embed_dim,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.frequency_stride, config.time_stride),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        input_values: (batch_size, frequency, time)
        output: (batch_size, num_patches, encoder_embed_dim)
        """
        input_values = input_values.unsqueeze(1)
        input_values = input_values.transpose(2, 3)

        return self.projection(input_values).flatten(2).transpose(1, 2)


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed_flexible(
    embed_dim: int, grid_size: tuple[int], cls_token=False
):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        # Prepends a fixed zero vector for the CLS token
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PositionAndMasking(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mask_ratio = config.mask_ratio

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.encoder_embed_dim))

        position_embeddings = get_2d_sincos_pos_embed_flexible(
            config.encoder_embed_dim,
            (config.num_patches_horizontal, config.num_patches_vertical),
            cls_token=True,
        )
        self.position_embeddings = nn.Parameter(
            torch.from_numpy(position_embeddings).float().unsqueeze(0)
        )
        self.position_embeddings.requires_grad = False

        self._seed = None

    def set_seed(self, seed: int = 42):
        """Set a permanent seed for masking for visualisation purposes"""
        self._seed = seed

    def random_mask(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        input_values: (batch_size, num_patches, encoder_embed_dim)
        """
        if self._seed is not None:
            torch.manual_seed(self._seed)

        mask_ratio = self.mask_ratio
        batch_size, num_patches, encoder_embed_dim = input_values.shape
        len_keep = int(num_patches * (1 - mask_ratio))

        noise = torch.rand(batch_size, num_patches, device=input_values.device)
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # Random shuffle of patch indices for each sample
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # New indices in original order
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(
            input_values,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, encoder_embed_dim),
        )

        # Binary mask: True is keep
        mask = torch.zeros(
            [batch_size, num_patches], device=input_values.device, dtype=torch.bool
        )
        mask[:, :len_keep] = True
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        x = input_values + self.position_embeddings[:, 1:, :]
        x, mask, ids_restore = self.random_mask(x)
        cls_token = self.cls_token + self.position_embeddings[:, 0, :]
        cls_tokens = cls_token.expand(input_values.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x, mask, ids_restore


class ViTEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.block = nn.ModuleList(
            [
                Block(
                    config.encoder_embed_dim,
                    config.encoder_num_heads,
                    mlp_ratio=config.encoder_mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(config.encoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(config.encoder_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.block:
            x = blk(x)
        x = self.norm(x)
        return x


class AudioMAEModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.patch_embed = PatchEmbed(config)
        self.position_and_masking = PositionAndMasking(config)
        self.encoder = ViTEncoder(config)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(input_values)
        x, mask, ids_restore = self.position_and_masking(x)
        x = self.encoder(x)
        return x, mask, ids_restore


class ReconstructionHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.num_patches_horizontal = config.num_patches_horizontal
        self.num_patches_vertical = config.num_patches_vertical
        self.decoder_attention = config.decoder_attention

        self.decoder_embed = nn.Linear(
            config.encoder_embed_dim, config.decoder_embed_dim, bias=True
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))

        position_embeddings = get_2d_sincos_pos_embed_flexible(
            config.decoder_embed_dim,
            (config.num_patches_horizontal, config.num_patches_vertical),
            cls_token=True,
        )
        self.position_embeddings = nn.Parameter(
            torch.from_numpy(position_embeddings).float().unsqueeze(0)
        )
        self.position_embeddings.requires_grad = False

        self.decoder_blocks = self.create_transformer_blocks(config)
        self.decoder_norm = nn.LayerNorm(config.decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            config.decoder_embed_dim, config.patch_size**2, bias=True
        )

    @staticmethod
    def create_transformer_blocks(config: ModelConfig) -> nn.ModuleList:
        # https://github.com/facebookresearch/AudioMAE/blob/main/models_mae.py
        if config.decoder_attention == "global":
            return nn.ModuleList(
                [
                    Block(
                        config.decoder_embed_dim,
                        config.decoder_num_heads,
                        config.encoder_mlp_ratio,
                        qkv_bias=True,
                        norm_layer=nn.LayerNorm,
                    )
                    for _ in range(config.decoder_depth)
                ]
            )
        elif config.decoder_attention == "local":
            decoder_modules = []
            for i in range(16):
                if i % 2 == 0:
                    shift_size = (0, 0)
                else:
                    shift_size = (2, 0)
                decoder_modules.append(
                    SwinTransformerBlock(
                        dim=config.decoder_embed_dim,
                        input_resolution=(
                            config.num_patches_horizontal,
                            config.num_patches_vertical,
                        ),
                        num_heads=16,
                        window_size=(4, 4),
                        shift_size=shift_size,
                        mlp_ratio=config.encoder_mlp_ratio,
                        norm_layer=nn.LayerNorm,
                    )
                )
            return nn.ModuleList(decoder_modules)
        else:
            raise ValueError(f"Invalid attention type: {config.decoder_attention}")

    def add_mask_tokens(
        self, hidden_state: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        mask_tokens = self.mask_token.repeat(
            hidden_state.shape[0], ids_restore.shape[1] + 1 - hidden_state.shape[1], 1
        )

        cls_token = hidden_state[:, 0:1, :]
        hidden_state = hidden_state[
            :, 1:, :
        ]  # Temporarily remove cls token for reordering

        x = torch.cat(
            [hidden_state, mask_tokens], dim=1
        )  # (batch_size, num_patches, decoder_embed_dim)
        x = torch.gather(
            x,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, hidden_state.shape[2]),
        )  # Restore ordering of patch representations (including mask tokens)

        x = torch.cat(
            [cls_token, x], dim=1
        )  # (batch_size, num_patches+1, decoder_embed_dim)

        x = x + self.position_embeddings
        return x

    def forward(
        self, encoder_output: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """
        hidden_state: (batch_size, num_unmasked_patches {+1 if using cls token}, encoder_embed_dim)
        ids_restore: (batch_size, num_patches)
        output: (batch_size, num_patches, patch_size**2)
        """
        hidden_state = self.decoder_embed(encoder_output)

        x = self.add_mask_tokens(hidden_state, ids_restore)

        if self.decoder_attention == "local":
            # Reshape for swin transformer blocks
            x = x[:, 1:, :]  # Don't use cls token for swin
            B, L, D = x.shape
            x = x.reshape(B, self.num_patches_horizontal, self.num_patches_vertical, D)

            for blk in self.decoder_blocks:
                x = blk(x)

            x = x.flatten(1, 2)
            x = self.decoder_norm(x)
            x = self.decoder_pred(x)
        elif self.decoder_attention == "global":
            # Keep cls token so that decoder representations attend to the entire spectrogram
            for blk in self.decoder_blocks:
                x = blk(x)
            x = x[:, 1:, :]  # Remove cls token for pixel reconstruction
            x = self.decoder_norm(x)
            x = self.decoder_pred(x)

        return x


class AudioMAEForPreTraining(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.model = AudioMAEModel(config)
        self.reconstruction_head = ReconstructionHead(config)
        self.patch_generator = PatchGenerator(config)
        self.normalise_patches = config.normalise_patches

    def calculate_loss(
        self,
        input_values: torch.Tensor,
        reconstructed_patches: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the patch normalised loss for the masked patches

        input_values: (batch_size, frames, mel_bins)
        reconstructed_patches: (batch_size, num_patches, patch_size**2)
        mask: (batch_size, num_patches)
        """
        original_patches = self.patch_generator(input_values)

        if self.normalise_patches:
            mean = original_patches.mean(dim=2, keepdim=True)  # mean per patch
            var = original_patches.var(dim=2, keepdim=True)  # variance per patch
            normalised_original_patches = (original_patches - mean) / (
                var + 1.0e-6
            ) ** 0.5  # audiomae
            original_patches = normalised_original_patches

        loss = (reconstructed_patches - original_patches) ** 2  # loss per pixel
        loss = loss.mean(dim=2)  # loss per patch
        loss = (
            loss * ~mask
        )  # False is masked, so we need to invert the mask to get the loss for the masked patches
        num_masked_patches = mask.sum(dim=1)
        loss = loss.sum(dim=1) / num_masked_patches  # loss per sample
        loss = loss.mean()  # mean loss over batch
        return loss, original_patches

    def forward(
        self, input_values: torch.Tensor, return_loss: bool = True
    ) -> torch.Tensor:
        x, mask, ids_restore = self.model(input_values)
        reconstructed_patches = self.reconstruction_head(x, ids_restore)

        if return_loss:
            loss, original_patches = self.calculate_loss(
                input_values, reconstructed_patches, mask
            )
            return loss, original_patches, reconstructed_patches, mask

        return reconstructed_patches, mask


class AudioMAEForClassification(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        config.mask_ratio = 0.0
        self.config = config
        self.model = AudioMAEModel(config)
        self.norm = nn.LayerNorm(config.encoder_embed_dim)
        self.classification_head = nn.Linear(
            config.encoder_embed_dim, config.num_labels
        )
        self.classification_strategy = config.classification_strategy

    def forward(self, input_values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x, mask, ids_restore = self.model(input_values)

        if self.classification_strategy == "cls_token":
            x = x[:, 0, :]
        elif self.classification_strategy == "global_pool":
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        else:
            raise ValueError(
                f"Invalid classification strategy: {self.classification_strategy}"
            )
        x = self.norm(x)
        logits = self.classification_head(x)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss, logits


class CoughSegmentor:
    def __init__(
        self,
        model,
        transform: Callable,
        window_length_frames: int = 40,
        window_hop_time: float = 0.01,
        sampling_rate: int = 16000,
        cough_minimum_duration: float = 0.15,
        frame_length_time: float = 0.025,
        frame_hop_time: float = 0.01,
        model_name: str = "coughvit",
    ):
        self.model = model
        self.transform = transform
        self.window_length_frames = window_length_frames
        self.window_hop_time = window_hop_time
        self.sampling_rate = sampling_rate
        self.cough_minimum_duration = cough_minimum_duration
        self.frame_length_time = frame_length_time
        self.frame_hop_time = frame_hop_time
        self.model_name = model_name

    def predict_on_window(self, audio_window: torch.Tensor) -> int:
        spectrogram = self.transform(audio_window).unsqueeze(0)
        placeholder_label = torch.tensor([0])
        if self.model_name == "ast":
            output = self.model(spectrogram)
            logits = output.logits
        else:
            _, logits = self.model(spectrogram, placeholder_label)
        return int(torch.argmax(logits, dim=1))

    @property
    def window_length_samples(self) -> int:
        frame_length_samples = int(self.frame_length_time * self.sampling_rate)
        frame_hop_samples = int(self.frame_hop_time * self.sampling_rate)
        return (
            self.window_length_frames - 1
        ) * frame_hop_samples + frame_length_samples

    def get_windows(self, audio_length_samples: int) -> list[tuple[int, int]]:
        window_hop_samples = int(self.window_hop_time * self.sampling_rate)
        window_ends_samples = range(
            self.window_length_samples, audio_length_samples, window_hop_samples
        )
        return [(end - self.window_length_samples, end) for end in window_ends_samples]

    def sliding_window_prediction(self, audio: torch.Tensor) -> dict:
        predictions = {}
        for start, end in self.get_windows(len(audio)):
            audio_window = audio[start:end]
            prediction = self.predict_on_window(audio_window)
            predictions[start] = prediction
        return predictions

    def filter_close_coughs(self, predictions: dict) -> dict:
        cough_minimum_samples = int(self.cough_minimum_duration * self.sampling_rate)
        filtered_predictions = {}
        for start in sorted(predictions.keys()):
            if len(filtered_predictions) == 0:
                filtered_predictions[start] = predictions[start]
            elif start - max(filtered_predictions.keys()) > cough_minimum_samples:
                # cough is far enough from previous cough
                filtered_predictions[start] = predictions[start]
        return filtered_predictions

    def calculate_event_times(
        self, cough_start_samples: list[int]
    ) -> tuple[list[float], list[float]]:
        start_times = [x / self.sampling_rate for x in cough_start_samples]
        end_times = []

        for i in range(len(start_times)):
            potential_end_time = (
                start_times[i] + self.window_length_samples / self.sampling_rate
            )

            if i == len(start_times) - 1 or potential_end_time < start_times[i + 1]:
                end_times.append(potential_end_time)
            else:
                end_times.append(start_times[i + 1])
        return start_times, end_times

    def __call__(self, audio):
        predictions = self.sliding_window_prediction(audio.squeeze())
        windows_with_coughs = {
            start: pred for start, pred in predictions.items() if pred == 1
        }
        filtered_predictions = self.filter_close_coughs(windows_with_coughs)
        return self.calculate_event_times(sorted(filtered_predictions.keys()))
