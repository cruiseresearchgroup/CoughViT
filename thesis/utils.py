import json
import random
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, random_split


class OversamplingDataset(Dataset):
    def __init__(
        self, dataset: Dataset, sample_labels: list[int], target_ratio: float = 1
    ):
        self.dataset = dataset
        self.sample_labels = sample_labels
        self.target_ratio = target_ratio

        self.class_counts = torch.bincount(torch.tensor(self.sample_labels))
        original_ratio = self.class_counts[1] / self.class_counts[0]
        if original_ratio > target_ratio:
            raise ValueError(
                f"Original ratio is greater than target ratio: {original_ratio} > {target_ratio}"
            )

        self.neg_indices = [
            idx for idx, label in enumerate(self.sample_labels) if label == 0
        ]
        self.pos_indices = [
            idx for idx, label in enumerate(self.sample_labels) if label == 1
        ]

        pos_samples_to_add = (
            int(self.class_counts[0] * self.target_ratio) - self.class_counts[1]
        )
        self.oversampled_pos_indices = self.pos_indices + random.choices(
            self.pos_indices, k=pos_samples_to_add
        )

        self.indices = self.neg_indices + self.oversampled_pos_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def extract_labels_from_indices(
    indices: list[int], get_label: Callable[[int], int]
) -> list[int]:
    return [get_label(idx) for idx in indices]


def split_dataset(
    dataset: Dataset, train_ratio: float = 0.8, manual_seed: int = 42
) -> tuple[Subset]:
    torch.manual_seed(manual_seed)
    train_length = int(train_ratio * len(dataset))
    test_length = len(dataset) - train_length
    train, test = random_split(dataset, [train_length, test_length])
    return train, test


def save_log_history(log_history: dict, save_path: str):
    with open(save_path, "w") as f:
        json.dump(log_history, f, indent=4)


def save_results(results: list, file_name: str, output_dir: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    average = sum([result[2] for result in results]) / len(results)
    with open(output_dir / file_name, "w") as f:
        for result in results:
            f.write(f"{result[0]},{result[1]},{result[2]}\n")
        f.write(f"Average,,{average}\n")


def load_weights(
    model: nn.Module,
    pretrained_path: Path,
    ignore_size_mismatch: bool = True,
    strict: bool = False,
) -> None:
    """
    Loads model weights in place fom a pretrained model.

    ignore_size_mismatch: Ignores the layer if the size of the pretrained model and the model are different
    strict: Ignores if there are different layers in the model and the pretrained model
    """
    state_dict: dict = torch.load(pretrained_path)

    # Filter out mismatched parameters
    if ignore_size_mismatch:
        temp = {}
        for layer, tensor in state_dict.items():
            if (
                layer in model.state_dict()
                and tensor.shape == model.state_dict()[layer].shape
            ):
                temp[layer] = tensor
        state_dict = temp

    # Load the filtered state dict into the model
    model.load_state_dict(state_dict, strict=strict)
