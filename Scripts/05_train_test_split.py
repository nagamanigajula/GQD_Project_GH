#!/usr/bin/env python3
"""
Split molecules into training, test, and screening sets

Randomly splits the cleaned molecule list into three groups:
  - Training: 4,000 molecules (used to train the ML model)
  - Test: 1,000 molecules (used to check the trained model)
  - Screening: everything else (predicted on later, not run through xTB yet)

The split is random but reproducible (fixed seed), so re-running this
script gives the same three groups every time.

Input:
  - data/cleaned/molecules_cleaned.csv
  - xyz_files/ (all converted xyz files)

Output:
  - xyz_files_train/, xyz_files_test/, xyz_files_screening/ (symlinks)
  - data/train_test_split/train_molecules.csv
  - data/train_test_split/test_molecules.csv
  - data/train_test_split/screening_molecules.csv

Usage:
  python 05_train_test_split.py
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path.home() / "GQD_Project"
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
SPLIT_DIR = DATA_DIR / "train_test_split"
XYZ_DIR = PROJECT_ROOT / "xyz_files"
XYZ_TRAIN_DIR = PROJECT_ROOT / "xyz_files_train"
XYZ_TEST_DIR = PROJECT_ROOT / "xyz_files_test"
XYZ_SCREENING_DIR = PROJECT_ROOT / "xyz_files_screening"
LOGS_DIR = PROJECT_ROOT / "logs"

INPUT_CSV = CLEANED_DIR / "molecules_cleaned.csv"
TRAIN_CSV = SPLIT_DIR / "train_molecules.csv"
TEST_CSV = SPLIT_DIR / "test_molecules.csv"
SCREENING_CSV = SPLIT_DIR / "screening_molecules.csv"

TRAIN_SIZE = 4000
TEST_SIZE = 1000
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "train_test_split.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def validate_inputs():
    """Check the required input files/folders exist before starting."""
    logger.info("Checking input files...")

    if not INPUT_CSV.exists():
        logger.error(f"Input CSV not found: {INPUT_CSV}")
        sys.exit(1)

    if not XYZ_DIR.exists():
        logger.error(f"XYZ directory not found: {XYZ_DIR}")
        sys.exit(1)

    logger.info(f"Input CSV: {INPUT_CSV}")
    logger.info(f"XYZ directory: {XYZ_DIR}")


def load_molecules():
    """Load the cleaned molecule list."""
    logger.info(f"Loading molecules from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df)} molecules")
    return df


def split_molecules(df):
    """Randomly split molecules into training, test, and screening sets."""
    logger.info("Splitting molecules...")

    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(df))

    train_indices = indices[:TRAIN_SIZE]
    test_indices = indices[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
    screening_indices = indices[TRAIN_SIZE + TEST_SIZE:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    screening_df = df.iloc[screening_indices].reset_index(drop=True)

    logger.info(f"Training set: {len(train_df)} molecules")
    logger.info(f"Test set: {len(test_df)} molecules")
    logger.info(f"Screening set: {len(screening_df)} molecules")
    logger.info(f"Total: {len(train_df) + len(test_df) + len(screening_df)} molecules")

    return train_df, test_df, screening_df


def save_split_csvs(train_df, test_df, screening_df):
    """Save each split as its own CSV."""
    logger.info("Saving split CSV files...")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_CSV, index=False)
    logger.info(f"Saved: {TRAIN_CSV}")

    test_df.to_csv(TEST_CSV, index=False)
    logger.info(f"Saved: {TEST_CSV}")

    screening_df.to_csv(SCREENING_CSV, index=False)
    logger.info(f"Saved: {SCREENING_CSV}")


def create_xyz_symlinks(train_df, test_df, screening_df):
    """Create train/test/screening folders containing symlinks to the xyz files."""
    logger.info("Creating xyz folders with symlinks...")

    for d in [XYZ_TRAIN_DIR, XYZ_TEST_DIR, XYZ_SCREENING_DIR]:
        if d.exists():
            logger.warning(f"Removing existing folder: {d}")
            import shutil
            shutil.rmtree(d)

    XYZ_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    XYZ_TEST_DIR.mkdir(parents=True, exist_ok=True)
    XYZ_SCREENING_DIR.mkdir(parents=True, exist_ok=True)

    def link_files(df, target_dir, label):
        logger.info(f"Linking {label} files...")
        for idx, row in df.iterrows():
            mol_filename = row['filename']  # e.g. "gqd_090000.mol"
            xyz_filename = mol_filename.replace('.mol', '.xyz')

            source = XYZ_DIR / xyz_filename
            target = target_dir / xyz_filename

            if source.exists():
                if not target.exists():
                    target.symlink_to(source.resolve())
            else:
                logger.warning(f"Source file not found: {source}")

        logger.info(f"Linked {len(df)} {label} files")

    link_files(train_df, XYZ_TRAIN_DIR, "training")
    link_files(test_df, XYZ_TEST_DIR, "test")
    link_files(screening_df, XYZ_SCREENING_DIR, "screening")


def verify_outputs(total_molecules):
    """Make sure the split folders contain the expected number of files."""
    logger.info("Verifying outputs...")

    train_count = len(list(XYZ_TRAIN_DIR.glob('*.xyz')))
    test_count = len(list(XYZ_TEST_DIR.glob('*.xyz')))
    screening_count = len(list(XYZ_SCREENING_DIR.glob('*.xyz')))

    logger.info(f"Training folder: {train_count} files")
    logger.info(f"Test folder: {test_count} files")
    logger.info(f"Screening folder: {screening_count} files")

    total = train_count + test_count + screening_count

    if total == total_molecules:
        logger.info(f"Total: {total} files (matches input)")
    else:
        logger.error(f"Total: {total} files (expected {total_molecules})")
        return False

    return True


def main():
    logger.info("="*80)
    logger.info("TRAIN / TEST / SCREENING SPLIT")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    validate_inputs()
    print()

    df = load_molecules()
    print()

    train_df, test_df, screening_df = split_molecules(df)
    print()

    save_split_csvs(train_df, test_df, screening_df)
    print()

    create_xyz_symlinks(train_df, test_df, screening_df)
    print()

    if not verify_outputs(len(df)):
        logger.error("Output verification failed.")
        sys.exit(1)
    print()

    logger.info("="*80)
    logger.info("SPLIT COMPLETE")
    logger.info("="*80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
