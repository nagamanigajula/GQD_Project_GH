#!/usr/bin/env python3
"""
Map selected molecules back to their original filenames

The 11,000 selection step outputs molecule IDs in the form
"screening_mol_<index>", based on their position in the screening set.
This script maps each index back to the original mol_id and .mol
filename, so the xTB step can find the actual structure files.

Input:
  - screening_molecules.csv (mol_id and filename for each screening molecule)
  - 11000_selected_molecules.csv (selected molecule_id, range, predicted_gap)

Output:
  - 11000_selected_molecules_with_filenames.csv
    (mol_id, molecule_filename, range, predicted_gap)

Usage:
  python 09_map_molecule_filenames.py
"""

import pandas as pd
from pathlib import Path
import sys

PROJECT_DIR = Path.home() / "GQD_Project"
SCREENING_CSV = PROJECT_DIR / "data/train_test_split/screening_molecules.csv"
SELECTED_CSV = PROJECT_DIR / "selection/11000_selected_molecules.csv"
OUTPUT_CSV = PROJECT_DIR / "selection/11000_selected_molecules_with_filenames.csv"

print("="*80)
print("MAPPING SELECTED MOLECULE INDICES TO ORIGINAL FILENAMES")
print("="*80)

print(f"\n1. Loading screening molecules from {SCREENING_CSV.name}...")
screening_df = pd.read_csv(SCREENING_CSV)
print(f"   Total screening molecules: {len(screening_df)}")

print(f"\n2. Loading selected molecules from {SELECTED_CSV.name}...")
selected_df = pd.read_csv(SELECTED_CSV)
print(f"   Total selected: {len(selected_df)}")

print(f"\n3. Extracting index from molecule_id...")
selected_df['screening_index'] = selected_df['molecule_id'].str.extract(r'(\d+)').astype(int)
print(f"   Index range: {selected_df['screening_index'].min()} - {selected_df['screening_index'].max()}")

print(f"\n4. Mapping to original screening molecules...")
screening_with_index = screening_df.reset_index().rename(columns={'index': 'screening_index'})
selected_df = selected_df.merge(
    screening_with_index[['screening_index', 'mol_id', 'filename']],
    on='screening_index',
    how='left'
)

print(f"   Found original filenames: {selected_df['filename'].notna().sum()}")
if selected_df['filename'].isna().sum() > 0:
    print(f"   Missing filenames: {selected_df['filename'].isna().sum()}")

print(f"\n5. Example of mapped molecules:")
example_cols = ['molecule_id', 'screening_index', 'mol_id', 'filename', 'range', 'predicted_gap']
for i, row in selected_df[example_cols].head(10).iterrows():
    print(f"   {row['molecule_id']:20s} -> {row['filename']:20s} (gap: {row['predicted_gap']:.4f} eV)")

selected_df_output = selected_df[['mol_id', 'filename', 'range', 'predicted_gap']].copy()
selected_df_output.rename(columns={'filename': 'molecule_filename'}, inplace=True)
selected_df_output.to_csv(OUTPUT_CSV, index=False)

print(f"\n6. Saved mapped file:")
print(f"   {OUTPUT_CSV}")
print(f"   Columns: mol_id, molecule_filename, range, predicted_gap")

print(f"\n" + "="*80)
print(f"DONE")
print(f"="*80)
