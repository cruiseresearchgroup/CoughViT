from enum import Enum
from pathlib import Path


class StringEnum(str, Enum):
    def __str__(self):
        return str(self.value)


IONA_DATASETS_DIRECTORY = Path("/mnt/data728/datasets")

PROJECT_DIR = Path(__file__).parent.parent

EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
