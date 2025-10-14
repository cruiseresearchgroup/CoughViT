import logging
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Callable, Optional, Union, Protocol

import pandas as pd
import torch
import torchaudio
import torch.utils.data as data
from torchaudio.functional import resample

import thesis.constants as c
from thesis.dicova2 import utils as d2utils


class Dicova2Splits(Enum):
    DEV = "dev"
    TEST = "test"


class Dicova2Dataset(data.Dataset):
    def __init__(
        self,
        audio_type: Union[d2utils.Dicova2AudioTypes, str],
        root: Union[str, Path] = c.IONA_DATASETS_DIRECTORY / "second_dicova",
        split: Dicova2Splits = Dicova2Splits.DEV,
        target_sr: int = 16000,
        transform: Optional[Callable] = None,
    ):
        if str(audio_type) not in (
            str(audio_type).lower() for audio_type in d2utils.Dicova2AudioTypes
        ):
            raise ValueError(f"Invalid audio type: {audio_type}")

        self.root = Path(root)
        self.split = split
        self.audio_type = audio_type
        self.target_sr = target_sr
        self.transform = transform
        self.label_mapping = d2utils.LABEL_MAPPING

        self.logger = logging.getLogger(self.__class__.__name__)

    @cached_property
    def metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self.split_folder / "metadata.csv", sep=" ")
        return df

    @property
    def split_folder(self) -> Path:
        if self.split == Dicova2Splits.DEV:
            return self.root / "Second_DiCOVA_Challenge_Dev_Data_Release"
        elif self.split == Dicova2Splits.TEST:
            return self.root / "Second_DiCOVA_Challenge_Test_Data_Release"
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def __len__(self):
        return len(self.metadata)

    def _load_transform(self, audio_file: Path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_file)

        if sample_rate != self.target_sr:
            waveform = resample(waveform, sample_rate, self.target_sr)
            waveform = waveform.squeeze()

        if self.transform:
            waveform = self.transform(waveform)

        return waveform

    def _get_audio_file(self, idx) -> Path:
        row = self.metadata.iloc[idx]
        return self.split_folder / "AUDIO" / self.audio_type / f"{row['SUB_ID']}.flac"

    def get_label(self, idx) -> int:
        row = self.metadata.iloc[idx]
        return self.label_mapping[row["COVID_STATUS"]]

    @property
    def indices(self) -> list[int]:
        return list(range(len(self)))

    def __getitem__(self, idx) -> dict:
        """Returns a tensor of size [N] and a label."""
        audio_file = self._get_audio_file(idx)
        waveform = self._load_transform(audio_file).squeeze()
        label = torch.tensor(self.get_label(idx))
        return {"input_values": waveform, "labels": label}


class DatasetWithIndices(Protocol):
    indices: list[int]

    def get_label(self, idx) -> int:
        ...


def extract_labels(dataset: DatasetWithIndices) -> list[int]:
    return [dataset.get_label(idx) for idx in dataset.indices]


class Dicova2DevFoldHandler:
    """Responsible for managing the folds for the Second Dicova dataset."""

    def __init__(self, dicova2_dev_dataset: Dicova2Dataset):
        if dicova2_dev_dataset.split != Dicova2Splits.DEV:
            raise ValueError("Dataset must be the dev split.")

        self.dicova2_dev_dataset = dicova2_dev_dataset
        self.lists_dir = dicova2_dev_dataset.split_folder / "LISTS"
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_fold_members(self, fold: int) -> tuple[pd.Series, pd.Series]:
        train_members = pd.read_csv(
            self.lists_dir / f"train_{fold}.csv", header=None
        ).squeeze()
        val_members = pd.read_csv(
            self.lists_dir / f"val_{fold}.csv", header=None
        ).squeeze()
        return train_members, val_members

    def _create_subset(self, indices: list[int]) -> data.Subset:
        subset = data.Subset(self.dicova2_dev_dataset, indices)
        subset.get_label = self.dicova2_dev_dataset.get_label
        return subset

    def get_fold(self, fold: int) -> tuple[Dicova2Dataset, Dicova2Dataset]:
        """Returns the train and test datasets for the given fold."""
        if fold not in d2utils.VALID_FOLDS:
            raise ValueError(f"Invalid fold: {fold}")

        train_members, val_members = self._get_fold_members(fold)

        metadata = self.dicova2_dev_dataset.metadata
        train_indices = metadata[metadata["SUB_ID"].isin(train_members)].index
        val_indices = metadata[metadata["SUB_ID"].isin(val_members)].index

        self.logger.info(
            f"{len(train_indices)} train members and {len(val_indices)} val members."
        )

        return (
            self._create_subset(train_indices),
            self._create_subset(val_indices),
        )
