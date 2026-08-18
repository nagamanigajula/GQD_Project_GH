#!/usr/bin/env python3
"""
Run GFN2-xTB on the 11,000 selected molecules

Runs xTB on the 11,000 molecules selected by the LightGBM prediction step,
using the same xTB settings as the initial 4,000/1,000 run. Results are
checkpointed periodically so the run can be resumed if interrupted.

Input:
  - 11000_selected_molecules_with_filenames.csv
  - Molecule files (xyz or mol) from the screening set

Output:
  - xtb_results.csv (all 11,000, including any failures)
  - detailed_results.csv (adds a prediction_error_ev column)
  - checkpoint_NNNNN.csv (saved periodically)
  - summary.json (run statistics)

xTB settings:
  - Method: --opt loose --gfn 2 --norestart --json
  - Timeout: 600 seconds per molecule

Usage:
  python 10_xtb_calculation_11000.py --workers 8
  python 10_xtb_calculation_11000.py --workers 16 --checkpoint 500
  python 10_xtb_calculation_11000.py --resume
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import logging
import json
import argparse
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path.home() / "GQD_Project"
SELECTION_DIR = PROJECT_DIR / "selection"
INPUT_CSV = SELECTION_DIR / "11000_selected_molecules_with_filenames.csv"
MOLECULE_XYZ_DIR = PROJECT_DIR / "xyz_files_screening"
MOLECULE_MOL_DIR = PROJECT_DIR / "mol_files"

XTB_RESULTS_DIR = PROJECT_DIR / "xtb_results_11000"
XTB_LOGS_DIR = XTB_RESULTS_DIR / "molecule_logs"
CHECKPOINT_DIR = XTB_RESULTS_DIR / "checkpoints"

OUTPUT_CSV = XTB_RESULTS_DIR / "xtb_results.csv"
DETAILED_CSV = XTB_RESULTS_DIR / "detailed_results.csv"
SUMMARY_JSON = XTB_RESULTS_DIR / "summary.json"
LOG_FILE = PROJECT_DIR / "logs" / "xtb_calculation_11000.log"

XTB_METHOD = "--opt loose --gfn 2 --norestart --json"
XTB_TIMEOUT = 600  # seconds per molecule
OMP_NUM_THREADS = "1"
OMP_STACKSIZE = "4G"

DEFAULT_CHECKPOINT_FREQ = 500

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
for d in [XTB_RESULTS_DIR, XTB_LOGS_DIR, CHECKPOINT_DIR, LOG_FILE.parent]:
    d.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)


def log_section(title):
    logger.info("\n" + "="*80)
    logger.info(title)
    logger.info("="*80)


def find_molecule_file(mol_filename):
    """Look for a molecule's coordinate file across the possible folders it could be in."""
    filename = Path(mol_filename)

    search_dirs = [
        MOLECULE_XYZ_DIR,
        PROJECT_DIR / "xyz_files",
        MOLECULE_MOL_DIR,
    ]

    for search_dir in search_dirs:
        if search_dir.exists():
            filepath = search_dir / filename
            if filepath.exists():
                return filepath

            xyz_path = search_dir / filename.with_suffix('.xyz')
            if xyz_path.exists():
                return xyz_path

            mol_path = search_dir / filename.with_suffix('.mol')
            if mol_path.exists():
                return mol_path

    return None


def run_xtb_single(mol_info):
    """Run xTB on one molecule and parse its HOMO/LUMO/gap from the output."""
    mol_id = mol_info.get('mol_id')
    mol_filename = mol_info.get('molecule_filename')
    predicted_gap = mol_info['predicted_gap']
    range_label = mol_info['range']

    try:
        os.environ['OMP_NUM_THREADS'] = OMP_NUM_THREADS
        os.environ['OMP_STACKSIZE'] = OMP_STACKSIZE

        mol_file = find_molecule_file(mol_filename)
        if not mol_file:
            return {
                'mol_id': mol_id, 'molecule_filename': mol_filename, 'range': range_label,
                'predicted_gap': predicted_gap, 'homo_ev': None, 'lumo_ev': None,
                'actual_gap_ev': None, 'status': 'FILE_NOT_FOUND',
                'error': f'Molecule file not found: {mol_filename}'
            }

        mol_work_dir = XTB_LOGS_DIR / mol_id
        mol_work_dir.mkdir(parents=True, exist_ok=True)

        xyz_basename = mol_file.name
        xyz_work_path = mol_work_dir / xyz_basename
        shutil.copy(mol_file, xyz_work_path)

        original_dir = os.getcwd()
        os.chdir(mol_work_dir)

        try:
            cmd = f"xtb {xyz_basename} {XTB_METHOD}"

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=XTB_TIMEOUT,
                env=os.environ.copy()
            )

            os.chdir(original_dir)

            if result.returncode != 0:
                return {
                    'mol_id': mol_id, 'molecule_filename': mol_filename, 'range': range_label,
                    'predicted_gap': predicted_gap, 'homo_ev': None, 'lumo_ev': None,
                    'actual_gap_ev': None, 'status': 'XTB_FAILED',
                    'error': result.stderr[:200]
                }

            json_file = mol_work_dir / 'xtbout.json'

            if not json_file.exists():
                return {
                    'mol_id': mol_id, 'molecule_filename': mol_filename, 'range': range_label,
                    'predicted_gap': predicted_gap, 'homo_ev': None, 'lumo_ev': None,
                    'actual_gap_ev': None, 'status': 'JSON_NOT_FOUND',
                    'error': 'xtbout.json not found'
                }

            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                gap_ev = data.get('HOMO-LUMO gap / eV')
                orbitals = data.get('orbital energies / eV', [])

                if gap_ev is not None and len(orbitals) > 0:
                    homo_idx = len(orbitals) // 2 - 1
                    lumo_idx = len(orbitals) // 2

                    homo_ev = orbitals[homo_idx]
                    lumo_ev = orbitals[lumo_idx]

                    return {
                        'mol_id': mol_id, 'molecule_filename': mol_filename, 'range': range_label,
                        'predicted_gap': predicted_gap, 'homo_ev': homo_ev, 'lumo_ev': lumo_ev,
                        'actual_gap_ev': gap_ev, 'status': 'SUCCESS', 'error': None
                    }
                else:
                    return {
                        'mol_id': mol_id, 'molecule_filename': mol_filename, 'range': range_label,
                        'predicted_gap': predicted_gap, 'homo_ev': None, 'lumo_ev': None,
                        'actual_gap_ev': None, 'status': 'PARSE_FAILED',
                        'error': 'Could not extract gap/orbitals from JSON'
                    }

            except json.JSONDecodeError as e:
                return {
                    'mol_id': mol_id, 'molecule_filename': mol_filename, 'range': range_label,
                    'predicted_gap': predicted_gap, 'homo_ev': None, 'lumo_ev': None,
                    'actual_gap_ev': None, 'status': 'JSON_ERROR',
                    'error': f'JSON parse error: {str(e)}'
                }

        finally:
            os.chdir(original_dir)

    except subprocess.TimeoutExpired:
        try:
            os.chdir(original_dir)
        except Exception:
            pass
        return {
            'mol_id': mol_id, 'molecule_filename': mol_filename, 'range': range_label,
            'predicted_gap': predicted_gap, 'homo_ev': None, 'lumo_ev': None,
            'actual_gap_ev': None, 'status': 'TIMEOUT',
            'error': f'xTB exceeded {XTB_TIMEOUT}s timeout'
        }

    except Exception as e:
        try:
            os.chdir(original_dir)
        except Exception:
            pass
        return {
            'mol_id': mol_id, 'molecule_filename': mol_filename, 'range': range_label,
            'predicted_gap': predicted_gap, 'homo_ev': None, 'lumo_ev': None,
            'actual_gap_ev': None, 'status': 'EXCEPTION', 'error': str(e)[:200]
        }


def run_xtb_parallel(molecules, num_workers, checkpoint_freq=500):
    """Run xTB across all molecules in parallel, saving a checkpoint periodically."""
    log_section("STARTING xTB CALCULATIONS (11,000 MOLECULES)")

    logger.info(f"Total molecules: {len(molecules)}")
    logger.info(f"Workers: {num_workers}")
    logger.info(f"Checkpoint frequency: {checkpoint_freq}")
    logger.info(f"xTB method: {XTB_METHOD}")
    logger.info(f"Timeout: {XTB_TIMEOUT}s per molecule")

    results = []
    start_time = time.time()
    start_idx = 0

    checkpoint_files = sorted(CHECKPOINT_DIR.glob("checkpoint_*.csv"))
    if checkpoint_files:
        last_checkpoint = checkpoint_files[-1]
        df_checkpoint = pd.read_csv(last_checkpoint)
        completed = len(df_checkpoint)
        logger.warning(f"Found checkpoint with {completed} molecules already done")
        response = input("Resume from checkpoint? (y/n): ").strip().lower()
        if response == 'y':
            results = df_checkpoint.to_dict('records')
            start_idx = completed
            logger.info(f"Resuming from index {start_idx}")
        else:
            logger.info("Starting fresh...")

    with Pool(processes=num_workers) as pool:
        for idx, result in enumerate(pool.imap_unordered(
            run_xtb_single,
            molecules[start_idx:],
            chunksize=10
        ), start=start_idx):

            results.append(result)
            current_count = len(results)

            if current_count % checkpoint_freq == 0:
                checkpoint_file = CHECKPOINT_DIR / f"checkpoint_{current_count:05d}.csv"
                pd.DataFrame(results).to_csv(checkpoint_file, index=False)

                elapsed = time.time() - start_time
                rate = current_count / elapsed
                remaining = len(molecules) - current_count
                eta = remaining / rate if rate > 0 else 0

                df_temp = pd.DataFrame(results)
                success_count = (df_temp['status'] == 'SUCCESS').sum()

                logger.info(f"Checkpoint: {current_count}/{len(molecules)} "
                            f"({100*current_count/len(molecules):.1f}%) - "
                            f"elapsed {elapsed/60:.1f} min - ETA {eta/60:.1f} min - "
                            f"{success_count} success / {len(results)-success_count} other")

            elif current_count % 100 == 0:
                elapsed = time.time() - start_time
                rate = current_count / elapsed
                remaining = len(molecules) - current_count
                eta = remaining / rate if rate > 0 else 0

                logger.info(f"Progress: {current_count}/{len(molecules)} "
                          f"({100*current_count/len(molecules):.1f}%) - ETA: {eta/60:.0f} min")

    elapsed = time.time() - start_time
    logger.info(f"All calculations complete in {elapsed/60:.1f} minutes")

    return results


def save_results(results):
    """Save the raw and detailed result CSVs, plus a summary of statistics."""
    log_section("SAVING RESULTS")

    df = pd.DataFrame(results)

    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved: {OUTPUT_CSV} ({len(df)} molecules)")

    df_detailed = df.copy()
    df_detailed['prediction_error_ev'] = df_detailed['actual_gap_ev'] - df_detailed['predicted_gap']
    df_detailed.to_csv(DETAILED_CSV, index=False)
    logger.info(f"Saved: {DETAILED_CSV}")

    log_section("STATISTICS")

    status_counts = df['status'].value_counts()
    logger.info("Results by status:")
    for status, count in status_counts.items():
        percentage = 100 * count / len(df)
        logger.info(f"  {status:20s}: {count:6d} ({percentage:5.1f}%)")

    success_df = df[df['status'] == 'SUCCESS']
    logger.info(f"\nSuccessful calculations: {len(success_df)}/{len(df)}")

    error = None
    if len(success_df) > 0:
        logger.info(f"\nActual HOMO-LUMO gaps (xTB):")
        logger.info(f"  Mean: {success_df['actual_gap_ev'].mean():.4f} eV")
        logger.info(f"  Range: {success_df['actual_gap_ev'].min():.4f} - {success_df['actual_gap_ev'].max():.4f} eV")

        error = success_df['actual_gap_ev'] - success_df['predicted_gap']
        logger.info(f"\nPrediction error (actual - predicted):")
        logger.info(f"  Mean error: {error.mean():.4f} eV")
        logger.info(f"  MAE:  {abs(error).mean():.4f} eV")
        logger.info(f"  RMSE: {np.sqrt((error**2).mean()):.4f} eV")

        within_01 = (abs(error) < 0.1).sum()
        within_02 = (abs(error) < 0.2).sum()
        logger.info(f"\n  Within +/-0.1 eV: {within_01} ({100*within_01/len(success_df):.1f}%)")
        logger.info(f"  Within +/-0.2 eV: {within_02} ({100*within_02/len(success_df):.1f}%)")

    summary = {
        'total_molecules': len(df),
        'successful': len(success_df),
        'success_rate': f"{100*len(success_df)/len(df):.1f}%",
        'status_breakdown': status_counts.to_dict(),
        'timestamp': datetime.now().isoformat()
    }

    if len(success_df) > 0:
        summary.update({
            'actual_gap_mean_ev': float(success_df['actual_gap_ev'].mean()),
            'actual_gap_std_ev': float(success_df['actual_gap_ev'].std()),
            'predicted_gap_mean_ev': float(success_df['predicted_gap'].mean()),
            'mean_error_ev': float(error.mean()),
            'mae_ev': float(abs(error).mean()),
            'rmse_ev': float(np.sqrt((error**2).mean()))
        })

    with open(SUMMARY_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved: {SUMMARY_JSON}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run GFN2-xTB on the 11,000 selected molecules"
    )

    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of workers (default: half of available CPU cores)')

    parser.add_argument('--checkpoint', type=int, default=DEFAULT_CHECKPOINT_FREQ,
                       help=f'Checkpoint frequency (default: {DEFAULT_CHECKPOINT_FREQ})')

    parser.add_argument('--test', type=int, default=None,
                       help='Debug mode: run on N molecules only')

    parser.add_argument('--resume', action='store_true',
                       help='Resume from the last checkpoint')

    args = parser.parse_args()

    if args.workers is None:
        args.workers = max(1, cpu_count() // 2)
        logger.info(f"Using {args.workers} workers ({cpu_count()} cores available)")

    log_section("xTB CALCULATIONS FOR 11,000 SELECTED MOLECULES")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info(f"Loading molecules from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    molecules = df.to_dict('records')

    if args.test:
        logger.warning(f"Debug mode: processing only {args.test} molecules")
        molecules = molecules[:args.test]

    logger.info(f"Loaded {len(molecules)} molecules")

    results = run_xtb_parallel(
        molecules,
        num_workers=args.workers,
        checkpoint_freq=args.checkpoint
    )

    save_results(results)

    log_section("COMPLETE")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results directory: {XTB_RESULTS_DIR}")


if __name__ == "__main__":
    main()
