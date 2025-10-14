from typing import Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import ASTFeatureExtractor


def vis_spec(spectrogram: torch.Tensor, normalised: bool = True) -> None:
    spectrogram = spectrogram.clone()
    spectrogram_np = spectrogram.detach().cpu().numpy()

    if normalised:
        plt.imshow(spectrogram_np.T, vmin=-5, vmax=5)
    else:
        plt.imshow(spectrogram_np.T, vmin=-15, vmax=0)
    plt.show()


def reverse_flatten(patches: torch.Tensor, original_shape=(512, 128)):
    if patches.dim == 3:
        """Squeeze batch dimension if it exists"""
        patches = patches.squeeze()

    # Reshape flattened patches to 16x16
    reshaped_patches = patches.view(-1, 16, 16)

    # Allocate tensor for reconstructed spectrogram
    reconstructed = torch.zeros(original_shape)

    h, w = original_shape
    num_patches_vertical = h // 16
    num_patches_horizontal = w // 16

    # Place each reshaped patch into the correct position
    for i in range(num_patches_vertical):
        for j in range(num_patches_horizontal):
            patch_idx = i * num_patches_horizontal + j
            reconstructed[
                i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16
            ] = reshaped_patches[patch_idx]

    return reconstructed


def vis_patches(
    patches: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
    normalised: bool = True,
) -> None:
    if patches.dim() == 3:
        patches = patches.squeeze()
    if mask is not None and mask.dim() == 2:
        mask = mask.squeeze()

    if mask is not None:
        mask = mask.unsqueeze(1)
        patches = patches.masked_fill(
            ~mask, torch.nan
        )  # We want to mask all False values

    spectrogram = reverse_flatten(patches)

    vis_spec(spectrogram, normalised=normalised)


class SpectrogramDatasetTransform:
    def __init__(
        self,
        max_length: int = 512,
        sampling_rate: int = 16000,
        normalise_to: Optional[str] = None,  # c19s, c19s-5s or audioset
    ):
        if normalise_to == "c19s-5s":
            mean = -13.24971
            std = 2.219659
            self.feature_extractor = ASTFeatureExtractor(
                max_length=max_length, do_normalize=True, mean=mean, std=std
            )
        if normalise_to == "c19s":
            mean = -11.2328
            std = 5.3759
            self.feature_extractor = ASTFeatureExtractor(
                max_length=max_length, do_normalize=True, mean=mean, std=std
            )
        elif normalise_to == "audioset":
            self.feature_extractor = ASTFeatureExtractor(
                max_length=max_length, do_normalize=True
            )
        else:
            self.feature_extractor = ASTFeatureExtractor(
                max_length=max_length, do_normalize=False
            )

        self.sampling_rate = sampling_rate

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Assumes the waveform is in the shape [T]. Outputs a tensor of shape [128, 1024]"""
        output = self.feature_extractor(
            waveform, sampling_rate=self.sampling_rate, return_tensors="pt"
        )["input_values"].squeeze()
        return output
