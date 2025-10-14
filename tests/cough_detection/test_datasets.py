import pytest

from thesis.cough_detection.datasets import EDCFoldHandler


class TestEDCFoldHandler:
    def test_shuffled_subject_ids(self):
        standard_fold_handler = EDCFoldHandler()

        # Same instance should always return the same shuffled subject ids
        assert (
            standard_fold_handler.shuffled_subject_ids()
            == standard_fold_handler.shuffled_subject_ids()
        )
        # Same seed should always return the same shuffled subject ids
        assert (
            standard_fold_handler.shuffled_subject_ids()
            == EDCFoldHandler(seed=standard_fold_handler.seed).shuffled_subject_ids()
        )
        # Different seed should usually return different shuffled subject ids
        assert (
            standard_fold_handler.shuffled_subject_ids()
            != EDCFoldHandler(seed=1000).shuffled_subject_ids()
        )
