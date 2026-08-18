#!/usr/bin/env python3
"""
Run GFN2-xTB on the training and test sets

Runs xTB on the 4,000 training and 1,000 test molecules to get their
actual HOMO, LUMO, and HOMO-LUMO gap values. These labeled results are
what the ML model is trained and tested on.

Input:
  - xyz_files_train/ (4,000 xyz files)
  - xyz_files_test/ (1,000 xyz files)

Output:
  - xTB_results/gap_values_train.csv
  - xTB_results/gap_values_test.csv
  - xTB_results/logs/ (per-molecule xTB output)

xTB settings:
  - Method: --opt loose --gfn 2 --norestart
  - Timeout: 600 seconds per molecule

Usage:
  python 06_xtb_calculation_5000.py --workers 16
  python 06_xtb_calculation_5000.py --dataset train --workers 16
  python 06_xtb_calculation_5000.py --dataset test --workers 16
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from functools import partial

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path.home() / "GQD_Project"
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = DATA_DIR / "train_test_split"
XYZ_TRAIN_DIR = PROJECT_ROOT / "xyz_files_train"
XYZ_TEST_DIR = PROJECT_ROOT / "xyz_files_test"
XTB_RESULTS_DIR = PROJECT_ROOT / "xTB_results"
XTB_LOGS_DIR = XTB_RESULTS_DIR / "logs"
LOGS_DIR = PROJECT_ROOT / "logs"

TRAIN_CSV = SPLIT_DIR / "train_molecules.csv"
TEST_CSV = SPLIT_DIR / "test_molecules.csv"

OUTPUT_TRAIN = XTB_RESULTS_DIR / "gap_values_train.csv"
OUTPUT_TEST = XTB_RESULTS_DIR / "gap_values_test.csv"

XTB_METHOD = "--opt loose --gfn 2 --norestart"
XTB_TIMEOUT = 600  # seconds per molecule
OMP_NUM_THREADS = "1"  # one thread per worker, to avoid CPU oversubscription
OMP_STACKSIZE = "4G"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(parents=True, exist_ok=True)
XTB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
XTB_LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "xtb_calculation_5000.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def validate_inputs():
    """Check the train/test xyz folders exist before starting."""
    logger.info("Checking inputs...")

    if not XYZ_TRAIN_DIR.exists():
        logger.error(f"Training xyz folder not found: {XYZ_TRAIN_DIR}")
        logger.error("Run the train/test split step first.")
        sys.exit(1)

    if not XYZ_TEST_DIR.exists():
        logger.error(f"Test xyz folder not found: {XYZ_TEST_DIR}")
        logger.error("Run the train/test split step first.")
        sys.exit(1)

    train_count = len(list(XYZ_TRAIN_DIR.glob('*.xyz')))
    test_count = len(list(XYZ_TEST_DIR.glob('*.xyz')))

    logger.info(f"Training set: {train_count} xyz files")
    logger.info(f"Test set: {test_count} xyz files")

    if train_count == 0 or test_count == 0:
        logger.error("No xyz files found.")
        sys.exit(1)


def get_file_list(directory, csv_file):
    """Match up mol_ids from the split CSV with their xyz files on disk."""
    logger.info(f"Loading molecule list from {csv_file}...")

    df = pd.read_csv(csv_file)
    files = []

    for idx, row in df.iterrows():
        mol_filename = row['filename']
        xyz_filename = mol_filename.replace('.mol', '.xyz')
        filepath = directory / xyz_filename

        if filepath.exists():
            files.append({
                'mol_id': row['mol_id'],
                'filepath': str(filepath),
                'filename': xyz_filename
            })

    logger.info(f"Found {len(files)} molecules")
    return files


def run_xtb_on_molecule(mol_info):
    """Run xTB on a single molecule and parse the HOMO/LUMO/gap from its output."""
    mol_id = mol_info['mol_id']
    filepath = mol_info['filepath']
    filename = mol_info['filename']

    mol_dir = XTB_LOGS_DIR / mol_id
    mol_dir.mkdir(parents=True, exist_ok=True)

    try:
        os.chdir(mol_dir)

        import shutil
        shutil.copy(filepath, mol_dir / filename)

        cmd = f"xtb {filename} {XTB_METHOD} --json"

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=XTB_TIMEOUT,
            env={
                **os.environ,
                'OMP_NUM_THREADS': OMP_NUM_THREADS,
                'OMP_STACKSIZE': OMP_STACKSIZE
            }
        )

        if result.returncode == 0:
            try:
                json_file = mol_dir / 'xtbout.json'

                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    gap_ev = data.get('HOMO-LUMO gap / eV')
                    orbitals = data.get('orbital energies / eV', [])

                    if gap_ev is not None and len(orbitals) > 0:
                        # HOMO and LUMO sit either side of the midpoint of
                        # the orbital energy list
                        homo_idx = len(orbitals) // 2 - 1
                        lumo_idx = len(orbitals) // 2

                        homo_ev = orbitals[homo_idx]
                        lumo_ev = orbitals[lumo_idx]

                        return {
                            'mol_id': mol_id,
                            'status': 'success',
                            'HOMO_eV': homo_ev,
                            'LUMO_eV': lumo_ev,
                            'HOMO_LUMO_gap_eV': gap_ev,
                            'error': None
                        }
                    else:
                        return {
                            'mol_id': mol_id,
                            'status': 'parse_failed',
                            'HOMO_eV': None,
                            'LUMO_eV': None,
                            'HOMO_LUMO_gap_eV': None,
                            'error': 'Could not find gap or orbital energies in output'
                        }
                else:
                    return {
                        'mol_id': mol_id,
                        'status': 'parse_failed',
                        'HOMO_eV': None,
                        'LUMO_eV': None,
                        'HOMO_LUMO_gap_eV': None,
                        'error': 'xtbout.json not found'
                    }

            except json.JSONDecodeError as e:
                return {
                    'mol_id': mol_id,
                    'status': 'parse_error',
                    'HOMO_eV': None,
                    'LUMO_eV': None,
                    'HOMO_LUMO_gap_eV': None,
                    'error': f'JSON parse error: {str(e)}'
                }

            except Exception as e:
                return {
                    'mol_id': mol_id,
                    'status': 'parse_error',
                    'HOMO_eV': None,
                    'LUMO_eV': None,
                    'HOMO_LUMO_gap_eV': None,
                    'error': str(e)
                }
        else:
            return {
                'mol_id': mol_id,
                'status': 'failed',
                'HOMO_eV': None,
                'LUMO_eV': None,
                'HOMO_LUMO_gap_eV': None,
                'error': result.stderr
            }

    except subprocess.TimeoutExpired:
        return {
            'mol_id': mol_id,
            'status': 'timeout',
            'HOMO_eV': None,
            'LUMO_eV': None,
            'HOMO_LUMO_gap_eV': None,
            'error': f'xTB calculation exceeded {XTB_TIMEOUT}s timeout'
        }

    except Exception as e:
        return {
            'mol_id': mol_id,
            'status': 'exception',
            'HOMO_eV': None,
            'LUMO_eV': None,
            'HOMO_LUMO_gap_eV': None,
            'error': str(e)
        }


def run_parallel_xtb(files, num_workers):
    """Run xTB on a list of molecules in parallel."""
    logger.info(f"Running xTB with {num_workers} workers...")

    results = []
    success_count = 0
    failed_count = 0

    with Pool(processes=num_workers) as pool:
        for idx, result in enumerate(pool.imap_unordered(run_xtb_on_molecule, files)):
            results.append(result)

            if result['status'] == 'success':
                success_count += 1
            else:
                failed_count += 1

            if (idx + 1) % 100 == 0:
                logger.info(f"Progress: {idx + 1}/{len(files)}  "
                            f"success={success_count} failed={failed_count}")

    logger.info(f"Completed {success_count}/{len(files)} successfully "
                f"({failed_count} failed)")

    return results


def save_results(results, output_file):
    """Save all xTB results (including failures) to CSV."""
    logger.info(f"Saving results to {output_file}...")

    df = pd.DataFrame(results)
    df_success = df[df['status'] == 'success'].copy()

    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df)} rows ({len(df_success)} successful)")

    return df_success


def verify_outputs(output_file, expected_count):
    """Quick sanity check on the saved output file."""
    logger.info("Verifying output...")

    if not output_file.exists():
        logger.error(f"Output file not found: {output_file}")
        return False

    df = pd.read_csv(output_file)
    success_df = df[df['status'] == 'success']

    logger.info(f"Output file: {output_file}")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Successful: {len(success_df)}/{expected_count}")
    logger.info(f"Gap range: {success_df['HOMO_LUMO_gap_eV'].min():.2f} - "
                f"{success_df['HOMO_LUMO_gap_eV'].max():.2f} eV")

    return True


def main():
    parser = argparse.ArgumentParser(description='Run GFN2-xTB on the training and test sets')

    parser.add_argument('--workers', type=int, default=16,
                       help='Number of parallel workers, one worker per CPU core (default: 16)')

    parser.add_argument('--dataset', type=str, default='both',
                       choices=['train', 'test', 'both'],
                       help='Which set to process: train, test, or both (default: both)')

    parser.add_argument('--test', type=int, default=None,
                       help='Debug mode: run on N molecules only')
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("xTB CALCULATIONS - INITIAL TRAIN/TEST SET")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"xTB method: {XTB_METHOD}")
    logger.info(f"Timeout per molecule: {XTB_TIMEOUT}s\n")

    validate_inputs()
    print()

    logger.info("Loading file lists...")
    train_files = get_file_list(XYZ_TRAIN_DIR, TRAIN_CSV)
    print()

    test_files = get_file_list(XYZ_TEST_DIR, TEST_CSV)
    print()

    datasets_to_process = []

    if args.dataset in ['train', 'both']:
        datasets_to_process.append(('train', train_files, OUTPUT_TRAIN))

    if args.dataset in ['test', 'both']:
        datasets_to_process.append(('test', test_files, OUTPUT_TEST))

    for dataset_name, files, output_file in datasets_to_process:
        logger.info(f"Processing {dataset_name} set...\n")

        if args.test:
            logger.warning(f"Debug mode: using only {args.test} molecules")
            files = files[:args.test]

        logger.info(f"Running xTB on {dataset_name} set...")
        results = run_parallel_xtb(files, args.workers)
        print()

        save_results(results, output_file)
        print()

        verify_outputs(output_file, len(files))
        print()

    logger.info("="*80)
    logger.info("xTB CALCULATIONS COMPLETE")
    logger.info("="*80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
