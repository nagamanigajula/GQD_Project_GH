#!/usr/bin/env python3
"""
Clean and validate the converted molecules

Checks the results from the .mol -> .xyz conversion step and keeps only
the molecules that converted successfully, have a matching mol_id and
filename, and have a corresponding .xyz file on dataset.

Input:  data/raw/molecules.csv (from the xyz conversion step)
        xyz_files/ (3D coordinate files)

Output: data/cleaned/molecules_cleaned.csv
        Only the molecules that passed all checks, ready for feature
        generation.

Usage:
  python 03_data_cleaning.py              # full run, saves output
  python 03_data_cleaning.py --validate   # just run the checks, don't save
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR if os.path.exists(os.path.join(CURRENT_DIR, 'mol_files')) else os.path.dirname(CURRENT_DIR)

INPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'raw', 'molecules.csv')
OUTPUT_XYZ_DIR = os.path.join(PROJECT_ROOT, 'xyz_files')
OUTPUT_CLEANED_CSV = os.path.join(PROJECT_ROOT, 'data', 'cleaned', 'molecules_cleaned.csv')
LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'data_cleaning.log')

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def validate_mol_id_filename_match(mol_id, filename):
    """
    Check that a mol_id and filename refer to the same structure.
    Example: mol_id 'MOL_000001' should match filename 'gqd_000001.mol'.
    """
    try:
        mol_num = mol_id.split('_')[-1]
        filename_no_ext = os.path.splitext(filename)[0]
        file_num = filename_no_ext.split('_')[-1]
        return mol_num == file_num
    except Exception:
        return False


def check_xyz_exists(filename, xyz_dir):
    """Check that the .xyz file for this molecule actually exists."""
    xyz_name = os.path.splitext(filename)[0] + '.xyz'
    xyz_path = os.path.join(xyz_dir, xyz_name)
    return os.path.exists(xyz_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(validate_only=False):
    logger.info("="*80)
    logger.info("DATA CLEANING AND VALIDATION")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80 + "\n")

    if not os.path.exists(INPUT_CSV):
        logger.error(f"Input file not found: {INPUT_CSV}")
        logger.error("Run the mol-to-xyz conversion step first.")
        return

    logger.info(f"Loading: {INPUT_CSV}")
    df_raw = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df_raw)} molecules\n")

    # Keep only molecules that converted successfully
    logger.info("Filtering to successful conversions...")

    if 'status' not in df_raw.columns:
        logger.error("'status' column not found in input CSV")
        return

    df_success = df_raw[df_raw['status'] == 'SUCCESS'].copy()
    failed = len(df_raw) - len(df_success)

    logger.info(f"  Success: {len(df_success)}")
    logger.info(f"  Failed/skipped: {failed}\n")

    if len(df_success) == 0:
        logger.error("No successful conversions found.")
        return

    # Check mol_id and filename actually refer to the same molecule
    logger.info("Checking mol_id / filename alignment...")

    mismatches = []
    for idx, row in df_success.iterrows():
        mol_id = row['mol_id']
        filename = row['filename']

        if not validate_mol_id_filename_match(mol_id, filename):
            mismatches.append({
                'mol_id': mol_id,
                'filename': filename,
                'index': idx
            })

    if mismatches:
        logger.warning(f"Found {len(mismatches)} mol_id/filename mismatches")
        for m in mismatches[:10]:
            logger.warning(f"   {m['mol_id']} vs {m['filename']}")

        mismatch_indices = [m['index'] for m in mismatches]
        df_success = df_success.drop(mismatch_indices)
        logger.info(f"Removed {len(mismatches)} mismatched rows")
    else:
        logger.info("All mol_id/filename pairs match")

    logger.info(f"After this check: {len(df_success)} molecules\n")

    # Check the xyz file actually exists for each remaining molecule
    logger.info("Checking that xyz files exist...")

    missing_xyz = []
    for idx, row in df_success.iterrows():
        filename = row['filename']
        if not check_xyz_exists(filename, OUTPUT_XYZ_DIR):
            missing_xyz.append({
                'mol_id': row['mol_id'],
                'filename': filename,
                'index': idx
            })

    if missing_xyz:
        logger.warning(f"Found {len(missing_xyz)} molecules with a missing xyz file")
        for m in missing_xyz[:10]:
            logger.warning(f"   {m['mol_id']}: {m['filename']}")

        missing_indices = [m['index'] for m in missing_xyz]
        df_success = df_success.drop(missing_indices)
        logger.info(f"Removed {len(missing_xyz)} molecules with missing xyz files")
    else:
        logger.info("All xyz files exist")

    logger.info(f"After this check: {len(df_success)} molecules\n")

    # Quality summary
    logger.info("="*80)
    logger.info("DATA QUALITY SUMMARY")
    logger.info("="*80)
    logger.info(f"Starting molecules:      {len(df_raw)}")
    logger.info(f"After success filter:    {len(df_raw[df_raw['status'] == 'SUCCESS'])}")
    logger.info(f"After mol_id check:      {len(df_success) + len(mismatches) if mismatches else len(df_success)}")
    logger.info(f"After xyz check:         {len(df_success)}")
    logger.info("="*80 + "\n")

    if validate_only:
        logger.info("Validation complete (--validate flag set, nothing saved)")
        return

    # Build the final cleaned table
    logger.info("Preparing cleaned dataset...")

    df_cleaned = df_success[['mol_id', 'filename']].copy()
    df_cleaned['SMILES'] = None  # filled in during feature generation if needed
    df_cleaned['conversion_status'] = 'OK'

    logger.info(f"Columns: {list(df_cleaned.columns)}")
    logger.info(f"Shape: {df_cleaned.shape}\n")

    os.makedirs(os.path.dirname(OUTPUT_CLEANED_CSV), exist_ok=True)
    df_cleaned.to_csv(OUTPUT_CLEANED_CSV, index=False)

    logger.info(f"Saved: {OUTPUT_CLEANED_CSV}")
    logger.info(f"  Molecules: {len(df_cleaned)}")
    logger.info(f"  Columns: {len(df_cleaned.columns)}")

    logger.info("\nFirst 10 rows:")
    logger.info(df_cleaned.head(10).to_string())

    # A few final sanity checks
    logger.info("\nFinal checks:")

    dups = df_cleaned['mol_id'].duplicated().sum()
    logger.info(f"  Duplicate mol_ids: {dups}")

    nan_mol_id = df_cleaned['mol_id'].isna().sum()
    nan_filename = df_cleaned['filename'].isna().sum()
    logger.info(f"  Missing mol_id: {nan_mol_id}")
    logger.info(f"  Missing filename: {nan_filename}")

    sample_rows = df_cleaned.sample(min(5, len(df_cleaned)), random_state=42)
    logger.info("\nSpot-checking a few rows:")
    for idx, row in sample_rows.iterrows():
        mol_id = row['mol_id']
        filename = row['filename']
        ok = "OK" if validate_mol_id_filename_match(mol_id, filename) else "MISMATCH"
        logger.info(f"  [{ok}] {mol_id} -> {filename}")

    logger.info("\n" + "="*80)
    logger.info("DATA CLEANING COMPLETE")
    logger.info("="*80)
    logger.info(f"Input molecules:  {len(df_raw)}")
    logger.info(f"Output molecules: {len(df_cleaned)}")
    logger.info(f"Filtered out:     {len(df_raw) - len(df_cleaned)}")
    logger.info(f"Success rate:     {100*len(df_cleaned)/len(df_raw):.1f}%")
    logger.info("="*80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Data cleaning and validation"
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run checks only, do not save output'
    )

    args = parser.parse_args()

    main(validate_only=args.validate)

    print("\n" + "="*80)
    print("Done. Output saved to:", OUTPUT_CLEANED_CSV)
    print("="*80 + "\n")
