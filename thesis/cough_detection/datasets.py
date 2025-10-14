import random
from functools import cached_property
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
import torch.utils.data as data
import torchaudio

import thesis.constants as c

EXTRACTED_AUDIO_DIRECTORY = (
    c.IONA_DATASETS_DIRECTORY / "edge_cough_detection" / "extracted_audio"
)


def get_subject_ids(extracted_audio_dir: Path = EXTRACTED_AUDIO_DIRECTORY) -> list[str]:
    return sorted([str(p.name) for p in extracted_audio_dir.glob("*") if p.is_dir()])


class EDCDataset(data.Dataset):
    def __init__(
        self,
        root: Path = EXTRACTED_AUDIO_DIRECTORY,
        transform: Optional[Callable] = None,
    ):
        self.root = root
        self.metadata_file = self.root / "metadata.csv"
        self.transform = transform

    @property
    def metadata(self):
        return pd.read_csv(self.metadata_file)

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_transform(self, audio_path: Path) -> torch.Tensor:
        # 6400 samples, 16000 Hz
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform[0]
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        segment_dir = self.root / str(row["subject_id"]) / str(row["segment_id"])
        waveform = self._load_transform(segment_dir / "audio.wav")
        label = int(row["label"])
        return {"input_values": waveform, "labels": label}


def get_fold(
    fold: int, transform: Optional[Callable] = None
) -> tuple[data.Subset, data.Subset]:
    """Returns a dict containing the train and validation datasets for a given fold"""
    metadata = pd.read_csv(EXTRACTED_AUDIO_DIRECTORY / "metadata.csv")
    dataset = EDCDataset(transform=transform)
    train_dataset = data.Subset(
        dataset, metadata[metadata.val_fold != fold].index.to_list()
    )
    val_dataset = data.Subset(
        dataset, metadata[metadata.val_fold == fold].index.to_list()
    )
    return train_dataset, val_dataset


def create_folds(subject_ids: list[int], seed=42) -> dict[int, int]:
    """Function uses to assign validation folds to each subject (0, 4)"""
    random.seed(seed)
    random.shuffle(subject_ids)

    fold_assignments = {}
    num_folds = 5

    for idx, subject_id in enumerate(subject_ids):
        fold_number = idx % num_folds
        fold_assignments[subject_id] = fold_number

    return fold_assignments


def create_metadata_df(
    extracted_audio_dir: Path, fold_assignments: dict[int, int]
) -> pd.DataFrame:
    """Creates a dataframe with metadata for each sample"""
    sample_dirs = [match.parent for match in extracted_audio_dir.rglob("audio.wav")]
    metadata = []
    for sample_dir in sample_dirs:
        subject_id = sample_dir.parts[-2]
        segment_id = sample_dir.parts[-1]
        label_file = sample_dir / "label.txt"
        with open(label_file) as f:
            label = int(f.read())
        val_fold = fold_assignments[subject_id]

        metadata.append(
            {
                "subject_id": subject_id,
                "segment_id": segment_id,
                "label": label,
                "val_fold": val_fold,
            }
        )

    return pd.DataFrame(metadata)


COUGH_DETECTION_TEST_SET = (
    c.IONA_DATASETS_DIRECTORY / "coughvit-test-sets" / "cough-detection"
)


class CoughDetectionTestSet(data.Dataset):
    def __init__(
        self,
        root: Path = COUGH_DETECTION_TEST_SET,
    ):
        self.root = root

    @cached_property
    def files(self) -> list[Path]:
        return sorted(self.root.rglob("*outward_facing_mic.wav"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        audio, _ = torchaudio.load(path)
        return path, audio
