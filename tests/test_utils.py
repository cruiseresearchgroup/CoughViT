import pytest

import random
from torch.utils.data import Dataset

from thesis.utils import OversamplingDataset


class DummyDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx]


@pytest.fixture
def dummy_dataset() -> DummyDataset:
    labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    return DummyDataset(labels)


def test_oversampling(dummy_dataset: DummyDataset):
    # Balanced
    oversampled = OversamplingDataset(dummy_dataset, dummy_dataset.labels, 1)
    neg_count = sum(1 for label in oversampled if label == 0)
    pos_count = sum(1 for label in oversampled if label == 1)
    assert neg_count == pos_count

    # Ratio < 1
    ratio = 0.5
    oversampled = OversamplingDataset(dummy_dataset, dummy_dataset.labels, ratio)
    neg_count = sum(1 for label in oversampled if label == 0)
    pos_count = sum(1 for label in oversampled if label == 1)
    assert neg_count * ratio == pos_count

    # Ratio > 1
    ratio = 2
    oversampled = OversamplingDataset(dummy_dataset, dummy_dataset.labels, ratio)
    neg_count = sum(1 for label in oversampled if label == 0)
    pos_count = sum(1 for label in oversampled if label == 1)
    assert neg_count * ratio == pos_count


def test_higher_original_ratio(dummy_dataset: DummyDataset):
    with pytest.raises(ValueError):
        OversamplingDataset(dummy_dataset, dummy_dataset.labels, target_ratio=0.1)
