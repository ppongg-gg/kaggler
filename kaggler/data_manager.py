import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_TRAIN_HINTS = ("train",)
_TEST_HINTS = ("test",)
_SAMPLE_HINTS = ("sample_submission", "sample submission", "samplesubmission")


@dataclass
class DataInventory:
    train_file: Path
    test_file: Path
    sample_submission_file: Optional[Path]
    extra_files: list = field(default_factory=list)
    train_shape: tuple = (0, 0)
    test_shape: tuple = (0, 0)
    column_names: list = field(default_factory=list)


class DataManager:
    def __init__(self, competition_name: str, workspace: Path):
        self.competition_name = competition_name
        self.raw_dir = workspace / "data" / "raw"
        self.processed_dir = workspace / "data" / "processed"

    def download(self) -> DataInventory:
        if self._is_downloaded():
            logger.info("Raw data already present, skipping download.")
            return self._inventory()

        import kaggle
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading data for '{self.competition_name}'...")
        kaggle.api.competition_download_files(
            self.competition_name,
            path=str(self.raw_dir),
            quiet=False,
        )
        self._extract_zips()
        return self._inventory()

    def _is_downloaded(self) -> bool:
        if not self.raw_dir.exists():
            return False
        files = list(self.raw_dir.glob("*"))
        # Consider downloaded if there's at least one non-zip file
        return any(f.suffix != ".zip" for f in files)

    def _extract_zips(self) -> None:
        for zip_path in self.raw_dir.glob("*.zip"):
            logger.info(f"Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.raw_dir)
            zip_path.unlink()

    def _inventory(self) -> DataInventory:
        csv_files = list(self.raw_dir.glob("**/*.csv"))
        if not csv_files:
            # Try other tabular formats
            csv_files = (
                list(self.raw_dir.glob("**/*.parquet"))
                + list(self.raw_dir.glob("**/*.tsv"))
            )

        if not csv_files:
            raise FileNotFoundError(
                f"No tabular data files found in {self.raw_dir}. "
                "Download may have failed or competition uses non-tabular data."
            )

        train_file = self._find_by_hints(csv_files, _TRAIN_HINTS)
        test_file = self._find_by_hints(csv_files, _TEST_HINTS)
        sample_file = self._find_by_hints(csv_files, _SAMPLE_HINTS)

        if train_file is None:
            raise FileNotFoundError(
                f"Could not identify train file among: {[f.name for f in csv_files]}"
            )
        if test_file is None:
            raise FileNotFoundError(
                f"Could not identify test file among: {[f.name for f in csv_files]}"
            )

        extra = [f for f in csv_files if f not in {train_file, test_file, sample_file}]

        train_df = self._read_file(train_file, nrows=5)
        test_df = self._read_file(test_file, nrows=5)
        train_shape = self._get_shape(train_file)
        test_shape = self._get_shape(test_file)

        logger.info(
            f"Data inventory: train={train_file.name} {train_shape}, "
            f"test={test_file.name} {test_shape}"
        )

        return DataInventory(
            train_file=train_file,
            test_file=test_file,
            sample_submission_file=sample_file,
            extra_files=extra,
            train_shape=train_shape,
            test_shape=test_shape,
            column_names=list(train_df.columns),
        )

    @staticmethod
    def _find_by_hints(files: list, hints: tuple) -> Optional[Path]:
        name_lower = {f: f.stem.lower() for f in files}
        # Exact match first
        for hint in hints:
            for f, name in name_lower.items():
                if name == hint:
                    return f
        # Substring match
        for hint in hints:
            for f, name in name_lower.items():
                if hint in name:
                    return f
        return None

    @staticmethod
    def _read_file(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            return df.head(nrows) if nrows else df
        sep = "\t" if path.suffix == ".tsv" else ","
        return pd.read_csv(path, nrows=nrows, sep=sep)

    def _get_shape(self, path: Path) -> tuple:
        df = self._read_file(path)
        return df.shape

    def load_train(self, inventory: DataInventory) -> pd.DataFrame:
        return self._read_file(inventory.train_file)

    def load_test(self, inventory: DataInventory) -> pd.DataFrame:
        return self._read_file(inventory.test_file)

    def load_sample_submission(self, inventory: DataInventory) -> Optional[pd.DataFrame]:
        if inventory.sample_submission_file:
            return self._read_file(inventory.sample_submission_file)
        return None
