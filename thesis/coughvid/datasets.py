from functools import cached_property
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, Subset, random_split

from thesis import constants as c

IONA_COUGHVID_DIR = c.IONA_DATASETS_DIRECTORY / "coughvid_20211012"


def merge_annotations(row):
    """
    Function to union annotations from multiple experts.
    """
    # Filter out NaN values
    non_nan_values = row.dropna()

    # If the row contains only NaN values, return NaN
    if len(non_nan_values) == 0:
        return pd.NA

    # Count occurrences of each value
    value_counts = non_nan_values.value_counts()

    # If the top two values have the same count, return NaN
    if len(value_counts) > 1 and value_counts.iloc[0] == value_counts.iloc[1]:
        return pd.NA

    # Otherwise, return the most frequent value
    return value_counts.idxmax()


class CoughvidDataset(Dataset):
    def __init__(
        self,
        coughvid_dir: Path = IONA_COUGHVID_DIR,
        transform=None,
    ):
        self.metadata_path = coughvid_dir / "metadata_compiled.csv"
        self.transform = transform
        self.files = coughvid_dir / "converted"

    @cached_property
    def metadata(self):
        df = pd.read_csv(self.metadata_path)
        df["cough_type_merged"] = df["cough_type_4"]
        df = df[
            df["cough_type_merged"].notna() & (df["cough_type_merged"] != "unknown")
        ]
        return df[["uuid", "cough_type_merged"]].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def _load_transform(self, audio_path: Path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.squeeze(0)
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform

    def interpret_label(self, label: str) -> int:
        if label == "dry":
            return 0
        elif label == "wet":
            return 1
        else:
            raise ValueError(f"Invalid label: {label}")

    def __getitem__(self, idx: int):
        uuid = self.metadata.iloc[idx]["uuid"]
        waveform = self._load_transform(self.files / f"{uuid}.wav")
        raw_label = self.metadata.iloc[idx]["cough_type_merged"]
        return {
            "input_values": waveform,
            "labels": self.interpret_label(raw_label),
        }


class SplitManager:
    def __init__(self, coughvid_dataset: CoughvidDataset):
        self.coughvid_dataset = coughvid_dataset
        self._dev, self._test = self._train_test_split(coughvid_dataset)

    def _train_test_split(
        self, dataset, train_ratio: float = 0.8, manual_seed: int = 42
    ) -> tuple[Subset, Subset]:
        torch.manual_seed(manual_seed)
        train_length = int(train_ratio * len(dataset))
        test_length = len(dataset) - train_length
        train, test = random_split(dataset, [train_length, test_length])
        return train, test

    def get_fold(self, fold: int) -> tuple[Subset, Subset]:
        """Returns a fold for 5 fold cross validation."""
        assert 0 <= fold < 5

        # Create deterministic folds based on the given fold number
        torch.manual_seed(fold)

        chunks = list(torch.chunk(torch.tensor(self._dev.indices), 5))
        validation_indices = chunks[fold].tolist()
        training_indices = [
            idx for i, chunk in enumerate(chunks) if i != fold for idx in chunk.tolist()
        ]

        return Subset(self.coughvid_dataset, training_indices), Subset(
            self.coughvid_dataset, validation_indices
        )

    @property
    def dev(self) -> Subset:
        return self._dev

    @property
    def test(self) -> Subset:
        return self._test


TEST_SET_DIR = (
    c.IONA_DATASETS_DIRECTORY / "coughvit-test-sets" / "coughvid" / "converted"
)


class CoughvidTestSet(Dataset):
    def __init__(
        self, coughvid_dir: Path = TEST_SET_DIR, transform: Optional[Callable] = None
    ):
        self.transform = transform
        self.coughvid_dir = coughvid_dir

    @cached_property
    def files(self) -> list[Path]:
        return list(self.coughvid_dir.glob("*.wav"))

    def __len__(self) -> int:
        return len(self.files)

    def _load_transform(self, audio_path: Path) -> torch.Tensor:
        waveform, _ = torchaudio.load(audio_path)
        waveform = waveform.squeeze(0)
        if waveform.dim() == 2:
            waveform = waveform[0]
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        input_values = self._load_transform(path)
        return {"input_values": input_values, "path": path}
