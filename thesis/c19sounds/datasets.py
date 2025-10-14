import bisect
import itertools
from functools import cached_property
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.functional import resample

import thesis.constants as c

AUDIO_TYPES = set(["breath", "cough", "voice"])
C19SOUNDS_ROOT_DIR = c.IONA_DATASETS_DIRECTORY / "covid19-sounds"


def interpret_label(label: str) -> int:
    positive_labels = set(["positiveLast14", "last14", "yes"])
    if label in positive_labels:
        return 1
    else:
        return 0


def get_metadata_dfs(metadata_dir: Path) -> dict[pd.DataFrame]:
    files = {
        "android": "results_raw_20210426_lan_yamnet_android_noloc.csv",
        "ios": "results_raw_20210426_lan_yamnet_ios_noloc.csv",
        "web": "results_raw_20210426_lan_yamnet_web_noloc.csv",
    }
    dataframes = {}
    for platform, filename in files.items():
        dataframes[platform] = pd.read_csv(
            metadata_dir / filename, sep=";", index_col=0
        )
    return dataframes


class C19SoundsDataset(Dataset):
    def __init__(
        self,
        audio_type: str,
        root_dir: Path = C19SOUNDS_ROOT_DIR,
        target_sr: int = 16000,
        transform: Optional[Callable] = None,
        audio_length: int = 10,
    ):
        if audio_type not in AUDIO_TYPES:
            raise ValueError(
                f"audio_type must be one of {AUDIO_TYPES}, but got {audio_type}"
            )

        self.audio_type = audio_type
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.transform = transform
        self.audio_length = audio_length
        self.audio_samples = self.target_sr * self.audio_length
        self.metadata_dfs = get_metadata_dfs(root_dir / "all_metadata")

    @property
    def indices(self) -> list[int]:
        return list(range(len(self)))

    @cached_property
    def _cumulative_sizes(self) -> list[int]:
        sizes = [len(df) for df in self.metadata_dfs.values()]
        return list(itertools.accumulate(sizes))

    def _find_group_idx(self, idx: int) -> int:
        """Uses binary search to find the group that idx belongs to.

        Why bisect_right?
        Cumulative sizes is a list of the cumulative sizes of each group.
        [G1, G1+G2, G1+G2+G3, ...] where G1 is the size of group 1, G2 is the size of group 2, etc.

        Group 1 contains the idxs in range [0, G1). The idx equal to G1 belongs to group 2.
        In this case bisect_right will correctly return the index of group 2.
        """
        return bisect.bisect_right(self._cumulative_sizes, idx)

    def _get_item_metadata(self, idx: int) -> pd.Series:
        group_idx = self._find_group_idx(idx)
        group_df = self.metadata_dfs[list(self.metadata_dfs.keys())[group_idx]]
        if group_idx == 0:
            return group_df.iloc[idx]
        else:
            within_group_idx = idx - self._cumulative_sizes[group_idx - 1]
            return group_df.iloc[within_group_idx]

    def _get_audio_path(self, item_metadata: pd.Series, audio_type: str) -> Path:
        audio_files_dir = self.root_dir / "covid19_data_0426"

        audio_type_column_mapping = {
            "breath": "Breath filename",
            "cough": "Cough filename",
            "voice": "Voice filename",
        }

        audio_type_column = audio_type_column_mapping[audio_type]

        if pd.isna(item_metadata[audio_type_column]):
            raise ValueError(
                f"Audio file not found for {audio_type}, {item_metadata['Uid']}"
            )

        return (
            audio_files_dir
            / item_metadata["Uid"]
            / item_metadata["Folder Name"]
            / item_metadata[audio_type_column]
        )

    def _load_transform(self, audio_path: Path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform[0]
        if sample_rate != self.target_sr:
            waveform = resample(waveform, sample_rate, self.target_sr)

        if len(waveform) < self.audio_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.audio_samples - len(waveform))
            )
        elif len(waveform) > self.audio_samples:
            waveform = waveform[: self.audio_samples]

        if self.transform:
            waveform = self.transform(waveform)
        return waveform

    def _get_audio(self, item_metadata: pd.Series) -> Optional[torch.Tensor]:
        try:
            audio_path = self._get_audio_path(item_metadata, self.audio_type)
            return self._load_transform(audio_path)
        except Exception as e:
            waveform = torch.zeros(self.audio_samples)
            if self.transform:
                waveform = self.transform(waveform)
            return waveform

    def __getitem__(self, idx) -> dict:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")

        item_metadata = self._get_item_metadata(idx)
        audio = self._get_audio(item_metadata)
        label = interpret_label(str(item_metadata["Covid-Tested"]))
        return {
            "input_values": audio,
            "labels": label,
        }

    def __len__(self) -> int:
        return self._cumulative_sizes[-1]

    def get_label(self, idx: int) -> int:
        item_metadata = self._get_item_metadata(idx)
        return interpret_label(str(item_metadata["Covid-Tested"]))
