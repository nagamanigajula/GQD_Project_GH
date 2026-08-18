#!/usr/bin/env python3
"""
Generate molecular features

Builds a feature set for each molecule: Morgan fingerprints, MACCS keys,
and RDKit's 2D descriptors. The list of RDKit descriptors is discovered
automatically at runtime rather than hardcoded, so it stays up to date
with whatever version of RDKit is installed.

Features generated per molecule:
  - Morgan fingerprints (2048-bit, radii 2/3/4/5): 8,192 features
  - MACCS keys (167-bit): 167 features
  - RDKit 2D descriptors (auto-discovered): 80+ features
  Total: roughly 8,400+ features per molecule
"""

import os
# limit each worker to a single thread so parallel workers don't compete
# with each other for CPU resources
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_STACKSIZE'] = '4G'
os.environ['NUMBA_NUM_THREADS'] = '1'

import sys
import time
import pandas as pd
import numpy as np
import logging
import argparse
import inspect
from datetime import datetime
from multiprocessing import Pool
from functools import partial

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR if os.path.exists(os.path.join(CURRENT_DIR, 'mol_files')) else os.path.dirname(CURRENT_DIR)

INPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'cleaned', 'molecules_cleaned.csv')
INPUT_MOL_DIR = os.path.join(PROJECT_ROOT, 'mol_files')
OUTPUT_FEATURES_DIR = os.path.join(PROJECT_ROOT, 'features')
OUTPUT_PARQUET = os.path.join(OUTPUT_FEATURES_DIR, 'molecular_features.parquet')
OUTPUT_NPZ = os.path.join(OUTPUT_FEATURES_DIR, 'molecular_features.npz')
OUTPUT_FEATURE_NAMES = os.path.join(OUTPUT_FEATURES_DIR, 'feature_names.txt')
OUTPUT_DESCRIPTOR_LIST = os.path.join(OUTPUT_FEATURES_DIR, 'available_descriptors.txt')
LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'feature_generation.log')

DEFAULT_WORKERS = 40
TIMEOUT_PER_MOL = 300

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


def check_dependencies():
    """Make sure the required packages are installed before starting."""
    required = ['pandas', 'numpy', 'rdkit', 'pyarrow']
    missing = []

    for pkg_name in required:
        try:
            __import__(pkg_name)
        except ImportError:
            missing.append(pkg_name)

    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        sys.exit(1)

    logger.info("All dependencies available")


def discover_all_descriptors():
    """Find every callable descriptor function in RDKit's Descriptors module."""
    from rdkit.Chem import Descriptors

    logger.info("Discovering available RDKit descriptors...")

    descriptors = {}
    all_attributes = dir(Descriptors)

    for attr_name in all_attributes:
        if attr_name.startswith('_'):
            continue

        try:
            attr = getattr(Descriptors, attr_name)
            if callable(attr):
                try:
                    sig = inspect.signature(attr)
                    if len(sig.parameters) > 0:
                        descriptors[attr_name] = attr
                except Exception:
                    descriptors[attr_name] = attr
        except Exception:
            continue

    logger.info(f"Found {len(descriptors)} candidate descriptor functions\n")
    return descriptors


def test_descriptors_on_sample(descriptors):
    """Try each descriptor on a simple test molecule (benzene) and keep the ones that work."""
    from rdkit import Chem

    logger.info("Testing descriptors on a sample molecule (benzene)...")

    mol = Chem.MolFromSmiles('c1ccccc1')
    if mol is None:
        logger.error("Could not build the test molecule")
        return {}

    working_descriptors = {}
    failed_descriptors = []

    for desc_name, desc_func in descriptors.items():
        try:
            result = desc_func(mol)
            if isinstance(result, (int, float)):
                if not np.isnan(float(result)):
                    working_descriptors[desc_name] = desc_func
                else:
                    failed_descriptors.append((desc_name, "result is NaN"))
            else:
                failed_descriptors.append((desc_name, f"non-numeric: {type(result).__name__}"))
        except Exception as e:
            failed_descriptors.append((desc_name, str(e)[:30]))

    logger.info(f"{len(working_descriptors)} descriptors work")
    logger.info(f"{len(failed_descriptors)} descriptors failed\n")

    return working_descriptors


def generate_features_for_molecule(mol_path, mol_id, morgan_radii, maccs, descriptor_names, timeout=TIMEOUT_PER_MOL):
    """
    Build the full feature vector for one molecule: Morgan fingerprints,
    MACCS keys, and the working RDKit descriptors.

    descriptor_names is passed as plain strings (not functions) so this
    can be sent to a worker process without pickling issues - each worker
    looks the function up by name from the Descriptors module itself.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, MACCSkeys, Descriptors
        import numpy as np

        mol = Chem.MolFromMolFile(mol_path, sanitize=False, removeHs=False)
        if mol is None:
            return {'mol_id': mol_id, 'status': 'FAIL_LOAD'}

        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(mol)
            mol = Chem.RemoveHs(mol)
        except Exception:
            pass

        result = {'mol_id': mol_id, 'status': 'OK'}

        # Morgan fingerprints
        try:
            for radius in morgan_radii:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=2048)
                for i, bit in enumerate(fp):
                    result[f'Morgan_R{radius}_{i}'] = int(bit)
        except Exception:
            pass

        # MACCS keys
        if maccs:
            try:
                maccs_keys = MACCSkeys.GenMACCSKeys(mol)
                for i in range(len(maccs_keys)):
                    result[f'MACCS_{i}'] = int(maccs_keys[i])
            except Exception:
                pass

        # RDKit 2D descriptors
        for desc_name in descriptor_names:
            try:
                if hasattr(Descriptors, desc_name):
                    desc_func = getattr(Descriptors, desc_name)
                    value = desc_func(mol)
                    if isinstance(value, (int, float)) and not np.isnan(float(value)):
                        result[f'Desc_{desc_name}'] = float(value)
            except Exception:
                pass  # skip any descriptor that fails on this molecule

        return result

    except Exception:
        return {'mol_id': mol_id, 'status': 'FAIL_EXCEPTION'}


def main(num_workers, test_limit, morgan_radii, use_maccs):
    check_dependencies()

    logger.info("="*80)
    logger.info("FEATURE GENERATION")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Workers: {num_workers}")
    logger.info(f"Morgan radii: {morgan_radii}")
    logger.info(f"MACCS: {use_maccs}")
    logger.info("="*80 + "\n")

    all_descriptors = discover_all_descriptors()
    working_descriptors = test_descriptors_on_sample(all_descriptors)

    logger.info(f"Using {len(working_descriptors)} descriptors")
    expected_total = len(morgan_radii) * 2048 + (167 if use_maccs else 0) + len(working_descriptors)
    logger.info(f"Expected total: ~{expected_total} features per molecule\n")

    os.makedirs(OUTPUT_FEATURES_DIR, exist_ok=True)
    with open(OUTPUT_DESCRIPTOR_LIST, 'w') as f:
        f.write("Descriptors used in this run:\n")
        f.write("="*80 + "\n\n")
        for desc_name in sorted(working_descriptors.keys()):
            f.write(f"{desc_name}\n")
    logger.info(f"Saved descriptor list: {OUTPUT_DESCRIPTOR_LIST}\n")

    if not os.path.exists(INPUT_CSV):
        logger.error(f"Input file not found: {INPUT_CSV}")
        logger.error("Run the data cleaning step first.")
        return

    logger.info(f"Loading: {INPUT_CSV}")
    df_input = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df_input)} molecules\n")

    if test_limit:
        df_input = df_input[:test_limit]
        logger.info(f"Test mode: processing first {len(df_input)} molecules\n")

    # Resume from a previous run if output already exists
    existing_features = None
    done_mol_ids = set()

    if os.path.exists(OUTPUT_PARQUET):
        try:
            existing_features = pd.read_parquet(OUTPUT_PARQUET)
            done_mol_ids = set(existing_features['mol_id'].tolist())
            logger.info(f"Found existing features for {len(done_mol_ids)} molecules, resuming\n")
        except Exception as e:
            logger.warning(f"Could not load existing features: {e}")

    pending = df_input[~df_input['mol_id'].isin(done_mol_ids)].copy()
    logger.info(f"Pending: {len(pending)} molecules\n")

    if len(pending) == 0:
        logger.info("All molecules already processed.")
        return

    mol_data = []
    for idx, row in pending.iterrows():
        mol_id = row['mol_id']
        filename = row['filename']
        mol_path = os.path.join(INPUT_MOL_DIR, filename)

        if os.path.exists(mol_path):
            mol_data.append((mol_path, mol_id))
        else:
            logger.warning(f"Molecule file not found: {mol_id}")

    logger.info(f"Found {len(mol_data)} molecule files\n")

    if not mol_data:
        logger.error("No molecule files found.")
        return

    logger.info(f"Generating features for {len(mol_data)} molecules with {num_workers} workers...\n")

    descriptor_names_list = list(working_descriptors.keys())

    start_time = time.time()
    generate_fn = partial(
        generate_features_for_molecule,
        morgan_radii=morgan_radii,
        maccs=use_maccs,
        descriptor_names=descriptor_names_list,
        timeout=TIMEOUT_PER_MOL
    )

    all_features = []
    with Pool(num_workers) as pool:
        for idx, features_dict in enumerate(pool.starmap(generate_fn, mol_data), 1):
            all_features.append(features_dict)

            if idx % 500 == 0:
                logger.info(f"Progress: {idx}/{len(mol_data)} molecules")

                try:
                    df_features = pd.DataFrame(all_features)
                    if existing_features is not None and len(existing_features) > 0:
                        df_combined = pd.concat([existing_features, df_features], ignore_index=True)
                        df_combined = df_combined.drop_duplicates(subset=['mol_id'], keep='first')
                    else:
                        df_combined = df_features
                    df_combined.to_parquet(OUTPUT_PARQUET)
                except Exception as e:
                    logger.warning(f"Checkpoint save failed: {str(e)[:30]}")

    elapsed = round((time.time() - start_time) / 60, 2)

    logger.info("\nFinalizing features...")

    df_new = pd.DataFrame(all_features)

    if existing_features is not None and len(existing_features) > 0:
        df_final = pd.concat([existing_features, df_new], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=['mol_id'], keep='first')
    else:
        df_final = df_new

    feature_cols = [c for c in df_final.columns if c not in ['mol_id', 'status']]

    logger.info(f"Total molecules: {len(df_final)}")
    logger.info(f"Total features: {len(feature_cols)}")

    morgan_features = len([c for c in feature_cols if c.startswith('Morgan_')])
    maccs_features = len([c for c in feature_cols if c.startswith('MACCS_')])
    desc_features = len([c for c in feature_cols if c.startswith('Desc_')])

    logger.info(f"  Morgan fingerprints: {morgan_features}")
    logger.info(f"  MACCS keys: {maccs_features}")
    logger.info(f"  RDKit descriptors: {desc_features}")

    os.makedirs(OUTPUT_FEATURES_DIR, exist_ok=True)

    df_final.to_parquet(OUTPUT_PARQUET)
    logger.info(f"Saved: {OUTPUT_PARQUET}")

    X = df_final[feature_cols].fillna(0).values
    np.savez(
        OUTPUT_NPZ,
        features=X,
        mol_ids=df_final['mol_id'].values,
        feature_names=np.array(feature_cols)
    )
    logger.info(f"Saved: {OUTPUT_NPZ}")

    with open(OUTPUT_FEATURE_NAMES, 'w') as f:
        for fname in feature_cols:
            f.write(fname + '\n')
    logger.info(f"Saved: {OUTPUT_FEATURE_NAMES}")

    logger.info("\n" + "="*80)
    logger.info("FEATURE GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Input:     {len(df_input)}")
    logger.info(f"Processed: {len(df_new)}")
    logger.info(f"Total:     {len(df_final)}")
    logger.info(f"Features:  {len(feature_cols)}")
    logger.info(f"Time:      {elapsed} minutes")
    logger.info("="*80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate molecular features (Morgan, MACCS, RDKit descriptors)"
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=DEFAULT_WORKERS,
        help=f'Number of parallel workers (default: {DEFAULT_WORKERS})'
    )
    parser.add_argument(
        '--test',
        type=int,
        default=None,
        help='Test mode: process only the first N molecules'
    )
    parser.add_argument(
        '--morgan-radii',
        type=int,
        nargs='+',
        default=[2, 3, 4, 5],
        help='Morgan fingerprint radii (default: 2 3 4 5)'
    )
    parser.add_argument(
        '--no-maccs',
        action='store_true',
        help='Skip MACCS key generation'
    )

    args = parser.parse_args()

    main(
        num_workers=args.workers,
        test_limit=args.test,
        morgan_radii=args.morgan_radii,
        use_maccs=not args.no_maccs
    )

    print("\n" + "="*80)
    print("Done.")
    print("="*80 + "\n")
