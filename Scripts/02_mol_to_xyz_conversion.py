#!/usr/bin/env python3
"""
Convert .mol files to .xyz (3D coordinates)

Takes the 2D .mol structures and converts each one into a 3D geometry
(embedding + MMFF optimization), saving the result as an .xyz file.
Runs in parallel across multiple worker processes since this step is
the slowest part of the pipeline when done one molecule at a time.

Each molecule is converted inside its own child process. This matters
because RDKit's 3D embedding step can occasionally crash on a bad
structure at the C level, which a normal Python try/except can't catch.
Running each conversion in its own process means one bad molecule only
kills that one worker's task, not the whole batch.

Usage:
  python 02_mol_to_xyz_conversion.py --test 100    # try it on 100 files first

Re-running the same command will skip molecules that were already
converted and only process what's left.
"""

import os
import sys
import time
import glob
import argparse
import logging
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR if os.path.exists(os.path.join(CURRENT_DIR, 'mol_files')) else os.path.dirname(CURRENT_DIR)

INPUT_MOL_DIR = os.path.join(PROJECT_ROOT, 'mol_files')
OUTPUT_XYZ_DIR = os.path.join(PROJECT_ROOT, 'xyz_files')
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
OUTPUT_CSV = os.path.join(DATA_RAW_DIR, 'molecules.csv')
LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'mol_to_xyz.log')

# ---------------------------------------------------------------------------
# Settings — adjust worker count to match your CPU core count
# ---------------------------------------------------------------------------
DEFAULT_WORKERS = 40
TIMEOUT_PER_MOL = 60  # seconds

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
# Single-molecule conversion (runs inside a child process)
# ---------------------------------------------------------------------------
def _xyz_worker(mol_path, xyz_path, result_queue):
    """Load a .mol file, embed it in 3D, optimize with MMFF, and write .xyz."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog('rdApp.*')

        mol = Chem.MolFromMolFile(mol_path, sanitize=False, removeHs=False)
        if mol is None:
            result_queue.put(("FAIL", "RDKit could not load mol file"))
            return

        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(mol)
        except Exception:
            pass

        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except Exception:
            pass

        # 3D embedding — this is the step most likely to crash on a bad structure
        status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if status != 0:
            status = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)

        if status != 0:
            result_queue.put(("FAIL", "Could not embed 3D coordinates"))
            return

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=100)
        except Exception:
            pass  # if optimization fails, still write out the unoptimized geometry

        Chem.MolToXYZFile(mol, xyz_path)
        result_queue.put(("OK", xyz_path))

    except Exception as e:
        result_queue.put(("FAIL", str(e)[:100]))


def convert_mol_to_xyz(mol_path, xyz_path, timeout=TIMEOUT_PER_MOL):
    """
    Run the conversion for one molecule in a separate process, with a
    timeout. Returns ("OK", xyz_path), ("FAIL", reason), ("TIMEOUT", ...),
    or ("CRASH", ...) if the child process died unexpectedly.
    """
    q = mp.Queue()
    p = mp.Process(target=_xyz_worker, args=(mol_path, xyz_path, q))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return ("TIMEOUT", "timeout exceeded")

    if p.exitcode != 0:
        label = "SIGSEGV" if p.exitcode == -11 else f"exit_code_{p.exitcode}"
        return ("CRASH", label)

    try:
        return q.get_nowait()
    except Exception:
        return ("FAIL", "no result from worker process")


# ---------------------------------------------------------------------------
# Batch processing — each worker handles a slice of the molecule list
# ---------------------------------------------------------------------------
def process_batch(batch_id, mol_batch, output_xyz_dir, timeout):
    """Convert a batch of molecules and return a list of result dicts."""
    results = []
    total_in_batch = len(mol_batch)

    print(f"[Worker {batch_id}] Starting batch: {total_in_batch} molecules")

    for mol_count, (mol_id, filename, mol_path) in enumerate(mol_batch, 1):
        xyz_name = filename.replace('.mol', '.xyz')
        xyz_path = os.path.join(output_xyz_dir, xyz_name)

        if os.path.exists(xyz_path):
            results.append({
                'mol_id': mol_id,
                'filename': filename,
                'SMILES': None,
                'status': 'SKIP'
            })
            if mol_count % 50 == 0:
                print(f"[Worker {batch_id}] Progress: {mol_count}/{total_in_batch}")
            continue

        status, detail = convert_mol_to_xyz(mol_path, xyz_path, timeout=timeout)

        if status == "OK":
            results.append({
                'mol_id': mol_id,
                'filename': filename,
                'SMILES': None,
                'status': 'SUCCESS'
            })
        else:
            results.append({
                'mol_id': mol_id,
                'filename': filename,
                'SMILES': None,
                'status': f'FAIL_{status}',
                'error': detail
            })

        if mol_count % 50 == 0:
            success = sum(1 for r in results if r['status'] == 'SUCCESS')
            failed = sum(1 for r in results if r['status'].startswith('FAIL_'))
            skipped = sum(1 for r in results if r['status'] == 'SKIP')
            print(f"[Worker {batch_id}] Progress: {mol_count}/{total_in_batch}")
            print(f"[Worker {batch_id}]   success={success}, failed={failed}, skipped={skipped}")

    success = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed = sum(1 for r in results if r['status'].startswith('FAIL_'))
    skipped = sum(1 for r in results if r['status'] == 'SKIP')
    print(f"[Worker {batch_id}] Batch complete: {mol_count} molecules "
          f"(success={success}, failed={failed}, skipped={skipped})")

    return results


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def load_existing_results():
    """Load molecules.csv if it already exists, so we can skip finished work."""
    if os.path.exists(OUTPUT_CSV):
        try:
            df = pd.read_csv(OUTPUT_CSV)
            done_files = set(df['filename'].tolist()) if 'filename' in df.columns else set()
            logger.info(f"Found existing results: {len(df)} molecules already processed")
            return df, done_files
        except Exception as e:
            logger.warning(f"Could not load {OUTPUT_CSV}: {e}")
            return pd.DataFrame(), set()
    return pd.DataFrame(), set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(num_workers, test_limit):
    logger.info("="*80)
    logger.info("CONVERTING .mol FILES TO .xyz (3D COORDINATES)")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Workers: {num_workers}")
    logger.info(f"Timeout per molecule: {TIMEOUT_PER_MOL}s")
    logger.info("="*80 + "\n")

    os.makedirs(OUTPUT_XYZ_DIR, exist_ok=True)
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    if not os.path.exists(INPUT_MOL_DIR):
        logger.error(f"Input directory not found: {INPUT_MOL_DIR}")
        return

    mol_files = sorted(glob.glob(os.path.join(INPUT_MOL_DIR, '*.mol')))

    if not mol_files:
        logger.error(f"No .mol files found in {INPUT_MOL_DIR}")
        return

    logger.info(f"Found {len(mol_files)} .mol files in {INPUT_MOL_DIR}\n")

    if test_limit:
        mol_files = mol_files[:test_limit]
        logger.info(f"Test mode: processing first {len(mol_files)} files\n")

    existing_df, done_files = load_existing_results()

    pending_mol_files = [
        f for f in mol_files
        if os.path.basename(f) not in done_files
    ]

    logger.info(f"Pending: {len(pending_mol_files)} molecules\n")

    if not pending_mol_files:
        logger.info("All molecules already processed.")
        return

    # Split the pending molecules evenly across workers
    batch_size = (len(pending_mol_files) + num_workers - 1) // num_workers
    batches = []

    for worker_id in range(num_workers):
        start_idx = worker_id * batch_size
        end_idx = min((worker_id + 1) * batch_size, len(pending_mol_files))

        if start_idx < len(pending_mol_files):
            batch = []
            for mol_path in pending_mol_files[start_idx:end_idx]:
                mol_basename = os.path.basename(mol_path)
                mol_name_no_ext = os.path.splitext(mol_basename)[0]  # gqd_000001
                file_number = mol_name_no_ext.split('_')[-1]         # 000001
                mol_id = f'MOL_{file_number}'                        # MOL_000001

                batch.append((mol_id, mol_basename, mol_path))

            batches.append((worker_id, batch))

    logger.info(f"Divided into {len(batches)} batches (~{batch_size} molecules each)")
    for worker_id, batch in batches:
        logger.info(f"  Worker {worker_id}: {len(batch)} molecules")
    logger.info("")

    logger.info(f"Starting {len(batches)} parallel workers...\n")

    all_results = []
    start_time = time.time()

    process_batch_fn = partial(
        process_batch,
        output_xyz_dir=OUTPUT_XYZ_DIR,
        timeout=TIMEOUT_PER_MOL
    )

    with ProcessPoolExecutor(max_workers=len(batches)) as executor:
        futures = {
            executor.submit(process_batch_fn, worker_id, batch): worker_id
            for worker_id, batch in batches
        }

        completed = 0
        for future in as_completed(futures):
            worker_id = futures[future]
            batch_results = future.result()
            all_results.extend(batch_results)
            completed += 1
            logger.info(f"Worker {worker_id} finished ({completed}/{len(batches)} batches done)")

    elapsed = round((time.time() - start_time) / 60, 2)

    logger.info(f"\nProcessing complete. Elapsed: {elapsed} minutes")
    logger.info("Saving results...\n")

    if not existing_df.empty:
        df_new = pd.DataFrame(all_results)
        df_final = pd.concat([existing_df, df_new], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=['filename'], keep='first')
    else:
        df_final = pd.DataFrame(all_results)

    df_final.to_csv(OUTPUT_CSV, index=False)

    success_count = (df_final['status'] == 'SUCCESS').sum()
    skip_count = (df_final['status'] == 'SKIP').sum()
    fail_count = len(df_final) - success_count - skip_count

    logger.info("="*80)
    logger.info("CONVERSION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total molecules : {len(mol_files)}")
    logger.info(f"  Success : {success_count} (new conversions)")
    logger.info(f"  Skipped : {skip_count} (already existed)")
    logger.info(f"  Failed  : {fail_count}")
    logger.info(f"  Time    : {elapsed} minutes")
    logger.info(f"  XYZ files : {OUTPUT_XYZ_DIR}/")
    logger.info(f"  Metadata  : {OUTPUT_CSV}")
    logger.info("="*80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .mol files to .xyz (3D coordinates)"
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS}). "
             f"A good starting point is your CPU core count."
    )
    parser.add_argument(
        '--test',
        type=int,
        default=None,
        help='Process only the first N files (useful for a quick test run)'
    )

    args = parser.parse_args()

    main(num_workers=args.workers, test_limit=args.test)

    print("\n" + "="*80)
    print("Done. Output saved to:", OUTPUT_CSV)
    print("="*80 + "\n")
