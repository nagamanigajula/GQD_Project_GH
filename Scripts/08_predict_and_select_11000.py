#!/usr/bin/env python3
"""
Predict band gaps and select 11,000 molecules for xTB validation

Uses the trained LightGBM model to predict band gaps for the screening
set (the molecules not used in training/testing), then selects 11,000
of them for actual xTB calculation.

Selection is stratified across ten band-gap bins spanning 0.70-1.75 eV,
with roughly 1,100 molecules drawn at random from each bin. This spreads
the validation set across the model's full prediction range rather than
concentrating only on the most promising region. If a bin doesn't have
enough molecules, the shortfall is filled from the well-populated middle
range (0.90-1.40 eV), and each filled molecule keeps its own actual
predicted gap.

Input:
  - LightGBM_model.pkl (trained model)
  - molecular_features.parquet
  - screening_molecules.csv (the ~94,648 molecules not used for training/testing)

Output:
  - 11000_selected_molecules.csv
  - selection_statistics.csv
  - gap distribution plots (PNG)

Usage:
  python 08_predict_and_select_11000.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path.home() / "GQD_Project"
MODELS_DIR = PROJECT_DIR / "models"
FEATURES_DIR = PROJECT_DIR / "features"
SELECTION_DIR = PROJECT_DIR / "selection"
LOGS_DIR = SELECTION_DIR / "logs"

DATA_DIR = PROJECT_DIR / "data/train_test_split"

SELECTION_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_FILE = FEATURES_DIR / "molecular_features.parquet"
LIGHTGBM_MODEL_FILE = MODELS_DIR / "LightGBM_model.pkl"
SCREENING_MOLECULES_FILE = DATA_DIR / "screening_molecules.csv"

# ---------------------------------------------------------------------------
# Band-gap bins for stratified selection
# ---------------------------------------------------------------------------
RANGES = [
    {"range": "0.70-0.80", "min": 0.70, "max": 0.80, "count": 1100, "role": "Lower boundary learning"},
    {"range": "0.80-0.90", "min": 0.80, "max": 0.90, "count": 1100, "role": "Safe zone (high confidence)"},
    {"range": "0.90-1.00", "min": 0.90, "max": 1.00, "count": 1100, "role": "Safe zone"},
    {"range": "1.00-1.10", "min": 1.00, "max": 1.10, "count": 1100, "role": "Safe zone (peak)"},
    {"range": "1.10-1.20", "min": 1.10, "max": 1.20, "count": 1100, "role": "Safe zone (peak)"},
    {"range": "1.20-1.30", "min": 1.20, "max": 1.30, "count": 1100, "role": "Safe zone"},
    {"range": "1.30-1.40", "min": 1.30, "max": 1.40, "count": 1100, "role": "Safe zone"},
    {"range": "1.40-1.50", "min": 1.40, "max": 1.50, "count": 1100, "role": "Safe zone"},
    {"range": "1.50-1.65", "min": 1.50, "max": 1.65, "count": 1100, "role": "Good confidence"},
    {"range": "1.65-1.75", "min": 1.65, "max": 1.75, "count": 1100, "role": "Upper boundary learning"},
]

TOTAL_TARGET = sum(r["count"] for r in RANGES)


def log_section(title):
    logger.info("")
    logger.info("="*80)
    logger.info(title)
    logger.info("="*80)


def load_lightgbm_model():
    """Load the LightGBM model trained on the 4,000/1,000 xTB dataset."""
    log_section("LOAD LIGHTGBM MODEL")

    if not LIGHTGBM_MODEL_FILE.exists():
        raise FileNotFoundError(f"LightGBM model not found at {LIGHTGBM_MODEL_FILE}")

    logger.info(f"Loading model from: {LIGHTGBM_MODEL_FILE}")
    with open(LIGHTGBM_MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

    logger.info("Model loaded")
    return model


def load_features_and_ids():
    """Load the screening molecule list and its features."""
    log_section("LOAD FEATURES AND MOLECULE IDs")

    if not SCREENING_MOLECULES_FILE.exists():
        raise FileNotFoundError(f"Screening molecules file not found at {SCREENING_MOLECULES_FILE}")

    logger.info(f"Loading screening molecules from: {SCREENING_MOLECULES_FILE}")
    screening_df = pd.read_csv(SCREENING_MOLECULES_FILE)
    logger.info(f"Screening molecules: {screening_df.shape[0]}")

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Features file not found at {FEATURES_FILE}")

    logger.info(f"Loading features from: {FEATURES_FILE}")
    features_df = pd.read_parquet(FEATURES_FILE)
    logger.info(f"Features loaded: {features_df.shape[0]} molecules x {features_df.shape[1]} features")

    # Keep only the feature rows that correspond to the screening molecules
    if len(features_df) > len(screening_df):
        start_idx = len(features_df) - len(screening_df)
        features_df = features_df.iloc[start_idx:].reset_index(drop=True)
        logger.info(f"Filtered features to {len(features_df)} screening molecules")

    screening_df = screening_df.reset_index(drop=True)

    return features_df, screening_df


def predict_all_molecules(model, features_df):
    """Predict the band gap for every screening molecule."""
    log_section("PREDICT GAPS FOR SCREENING MOLECULES")

    logger.info(f"Predicting for {len(features_df)} molecules...")

    numeric_features = features_df.select_dtypes(include=[np.number])
    logger.info(f"Using {numeric_features.shape[1]} numeric features")

    predictions = model.predict(numeric_features.values)

    logger.info(f"Predicted gap range: {predictions.min():.4f} - {predictions.max():.4f} eV")
    logger.info(f"Mean: {predictions.mean():.4f} eV, median: {np.median(predictions):.4f} eV")

    return predictions


def select_stratified(features_df, screening_df, predictions):
    """
    Select molecules across the ten band-gap bins. First tries to fill
    each bin to its target count with a random draw. Any shortfall is
    then filled from the middle range (0.90-1.40 eV), where predictions
    are most reliable.
    """
    log_section("STRATIFIED SELECTION ACROSS RANGES")

    selected_molecules = []
    selection_stats = []
    total_selected = 0

    logger.info("Step 1: primary selection, target 1,100 per range")

    for idx, range_config in enumerate(RANGES, 1):
        range_min = range_config["min"]
        range_max = range_config["max"]
        target_count = range_config["count"]
        range_label = range_config["range"]

        mask = (predictions >= range_min) & (predictions < range_max)
        available_count = mask.sum()

        logger.info(f"Range {idx}: {range_label} eV ({range_config['role']}) - "
                    f"available: {available_count}, target: {target_count}")

        range_indices = np.where(mask)[0]

        if available_count == 0:
            logger.warning(f"  No molecules available in this range - will fill from safe zone")
            selection_stats.append({
                "range": range_label, "role": range_config["role"], "target": target_count,
                "available": 0, "selected": 0, "percentage": 0.0, "status": "NO_AVAILABLE"
            })
            continue

        if available_count >= target_count:
            selected_indices = np.random.choice(range_indices, size=target_count, replace=False)
            selected_count = target_count
            status = "MET"
        else:
            selected_indices = range_indices
            selected_count = available_count
            logger.warning(f"  Only {selected_count} available (target {target_count}) - will fill from pool")
            status = "PARTIAL"

        for mol_idx in selected_indices:
            mol_id = f"screening_mol_{mol_idx}"
            selected_molecules.append({
                "molecule_id": mol_id,
                "range": range_label,
                "predicted_gap": predictions[mol_idx]
            })

        total_selected += selected_count

        percentage = (selected_count / available_count) * 100 if available_count > 0 else 0
        selection_stats.append({
            "range": range_label, "role": range_config["role"], "target": target_count,
            "available": available_count, "selected": selected_count,
            "percentage": percentage, "status": status
        })

    logger.info(f"Primary selection complete: {total_selected} molecules")

    for stat in selection_stats:
        stat.setdefault('filled', 0)
        stat.setdefault('final_selected', stat['selected'])

    # Fill any remaining shortfall from the reliable middle range
    remaining_needed = TOTAL_TARGET - total_selected

    if remaining_needed > 0:
        log_section("STEP 2: FILL REMAINING SHORTFALL")
        logger.info(f"Need {remaining_needed} more molecules to reach {TOTAL_TARGET}")
        logger.info("Filling from the 0.90-1.40 eV range (molecules keep their real predicted gap)")

        safe_fill_mask = (predictions >= 0.90) & (predictions < 1.40)
        available_for_fill = np.where(safe_fill_mask)[0]

        already_selected_indices = set(int(m['molecule_id'].split('_')[-1]) for m in selected_molecules)
        available_for_fill = np.array([idx for idx in available_for_fill if idx not in already_selected_indices])

        logger.info(f"Available in safe zone for filling: {len(available_for_fill)}")

        fill_count = min(remaining_needed, len(available_for_fill))
        if fill_count < remaining_needed:
            logger.warning(f"Only {len(available_for_fill)} available (needed {remaining_needed}) - "
                           f"selecting all available")

        if fill_count > 0:
            fill_indices = np.random.choice(available_for_fill, size=fill_count, replace=False)
            filled_by_range = {r["range"]: 0 for r in RANGES}

            for mol_idx in fill_indices:
                mol_id = f"screening_mol_{mol_idx}"
                mol_gap = predictions[mol_idx]

                assigned_range = None
                for range_config in RANGES:
                    if range_config["min"] <= mol_gap < range_config["max"]:
                        assigned_range = range_config["range"]
                        filled_by_range[assigned_range] += 1
                        break

                if assigned_range:
                    selected_molecules.append({
                        "molecule_id": mol_id, "range": assigned_range, "predicted_gap": mol_gap
                    })

                    stat_idx = [i for i, s in enumerate(selection_stats) if s['range'] == assigned_range][0]
                    selection_stats[stat_idx]["filled"] += 1
                    selection_stats[stat_idx]["final_selected"] += 1

            logger.info("Filled molecules by range:")
            for range_label, count in filled_by_range.items():
                if count > 0:
                    logger.info(f"  {range_label}: +{count}")

    selected_df = pd.DataFrame(selected_molecules)
    logger.info(f"Total selected: {len(selected_df)} molecules")

    log_section("SELECTION SUMMARY")
    for stat in selection_stats:
        logger.info(f"{stat['range']}: target={stat['target']} selected={stat['selected']} "
                    f"filled={stat.get('filled', 0)} total={stat.get('final_selected', stat['selected'])}")

    return selected_df, selection_stats


def prepare_output(selected_df, screening_df):
    """Sort the selection into a clean, ordered output table."""
    log_section("PREPARE OUTPUT")

    output_df = selected_df[['molecule_id', 'range', 'predicted_gap']].copy()

    range_order_map = {r["range"]: i + 1 for i, r in enumerate(RANGES)}
    output_df['range_order'] = output_df['range'].map(range_order_map)
    output_df = output_df.sort_values(['range_order', 'predicted_gap']).drop('range_order', axis=1)

    logger.info(f"Output ready: {len(output_df)} molecules")

    return output_df


def save_results(output_df, selection_stats):
    """Save the selected molecule list and selection statistics."""
    log_section("SAVE RESULTS")

    output_file = SELECTION_DIR / "11000_selected_molecules.csv"
    output_df.to_csv(output_file, index=False)
    logger.info(f"Saved: {output_file}")

    stats_df = pd.DataFrame(selection_stats)
    cols_to_save = ['range', 'role', 'target', 'selected', 'filled', 'final_selected']
    cols_to_save = [c for c in cols_to_save if c in stats_df.columns]

    stats_file = SELECTION_DIR / "selection_statistics.csv"
    stats_df[cols_to_save].to_csv(stats_file, index=False)
    logger.info(f"Saved: {stats_file}")

    total_selected = stats_df['selected'].sum()
    total_filled = stats_df['filled'].sum() if 'filled' in stats_df.columns else 0

    logger.info(f"\nGrand total: {total_selected} originally selected, "
                f"{total_filled} filled, {len(output_df)} final")

    return output_file, stats_file


def create_visualizations(output_df, predictions, selection_stats):
    """Plot the prediction distribution and the selection breakdown by range."""
    log_section("CREATE VISUALIZATIONS")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(predictions, bins=100, alpha=0.7, color='skyblue', edgecolor='black', label='All molecules')

    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    for idx, range_config in enumerate(RANGES):
        ax.axvline(range_config["min"], color=colors[idx], linestyle='--', alpha=0.5)

    ax.axvspan(0.73, 1.70, alpha=0.1, color='green', label='Target range')
    ax.set_xlabel('Predicted Gap (eV)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Predicted Band Gaps (screening set)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    viz1_file = SELECTION_DIR / "gap_distribution_all_molecules.png"
    plt.savefig(viz1_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {viz1_file}")
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(output_df['predicted_gap'], bins=50, alpha=0.7, color='lightgreen',
            edgecolor='black', label=f'Selected ({len(output_df)})')
    ax.axvspan(0.73, 1.70, alpha=0.1, color='green', label='Target range')

    for range_config in RANGES:
        ax.axvline(range_config["min"], color='red', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Predicted Gap (eV)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Selected Molecules', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    viz2_file = SELECTION_DIR / "gap_distribution_selected.png"
    plt.savefig(viz2_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {viz2_file}")
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 6))
    range_labels = [r["range"] for r in RANGES]
    target_counts = [r["count"] for r in RANGES]

    stats_df = pd.DataFrame(selection_stats)
    final_counts = []
    for label in range_labels:
        row = stats_df[stats_df['range'] == label]
        final_counts.append(row.iloc[0].get('final_selected', row.iloc[0]['selected']) if len(row) > 0 else 0)

    x = np.arange(len(range_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, target_counts, width, label='Target', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width/2, final_counts, width, label='Selected', alpha=0.8, color='lightgreen')

    ax.set_xlabel('Range (eV)', fontsize=12)
    ax.set_ylabel('Number of Molecules', fontsize=12)
    ax.set_title('Target vs Selected Molecules by Range', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels, rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    viz3_file = SELECTION_DIR / "target_vs_selected.png"
    plt.savefig(viz3_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {viz3_file}")
    plt.close()


def main():
    start_time = datetime.now()
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_section("SELECT 11,000 MOLECULES FOR xTB VALIDATION")

    try:
        features_df, screening_df = load_features_and_ids()
        model = load_lightgbm_model()

        predictions = predict_all_molecules(model, features_df)

        selected_df, selection_stats = select_stratified(features_df, screening_df, predictions)

        output_df = prepare_output(selected_df, screening_df)

        output_file, stats_file = save_results(output_df, selection_stats)

        create_visualizations(output_df, predictions, selection_stats)

        log_section("SELECTION COMPLETE")
        logger.info(f"Selected {len(output_df)} molecules")
        logger.info(f"Output saved to: {SELECTION_DIR}")

        end_time = datetime.now()
        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Duration: {end_time - start_time}")

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
