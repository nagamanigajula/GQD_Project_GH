# GQD Project

Scripts and results for generating fusion-based graphene quantum dot (GQD)
structures and screening them for near-infrared (NIR) emission using GFN2-xTB,
machine learning, and TD-DFT.

## Folder structure

```
scripts/            All pipeline scripts, in order (01 to 11)
xtb_results/
  5000/              Initial xTB results (4,000 train + 1,000 test)
  11000/              ML-predicted + xTB-validated results for 11,000 molecules
top_50_candidates/   Final selected candidates (Excel file)
```

## What each script does

1. **01_structure_generation.py** — Generates the GQD structures. Starts from
   a single hexagon and grows it outward randomly, up to a random size
   between 20-40 rings, checking each structure for validity along the way.

2. **02_mol_to_xyz_conversion.py** — Converts the generated `.mol` files into
   `.xyz` files (3D coordinates), which is what the xTB calculations need.

3. **03_data_cleaning.py** — Checks the converted structures and keeps only
   the ones that converted successfully.

4. **04_feature_generation.py** — Calculates molecular features for every
   structure (fingerprints and RDKit descriptors), used later for ML.

5. **05_train_test_split.py** — Randomly splits the structures into a
   training set (4,000), a test set (1,000), and a screening set (the rest).

6. **06_xtb_calculation_5000.py** — Runs the actual xTB calculation on the
   4,000 + 1,000 split to get real HOMO, LUMO, and band gap values.

7. **07_ml_model_training.py** — Trains five ML models (Random Forest,
   XGBoost, LightGBM, CatBoost, Neural Network) on the 4,000/1,000 xTB
   results and picks the best one.

8. **08_predict_and_select_11000.py** — Uses the trained model to predict
   band gaps for the rest of the structures, then selects 11,000 of them
   spread across the full band gap range for further xTB validation.

9. **09_map_molecule_filenames.py** — Small helper script that matches the
   selected molecules back to their original filenames.

10. **10_xtb_calculation_11000.py** — Runs the actual xTB calculation on
    those 11,000 selected structures.

11. **11_select_top100_top50.py** — From the 11,000 xTB-validated results,
    filters down to the top 100 candidates by band gap, then narrows that
    to the top 50 based on synthetic accessibility (how easy the structure
    would be to make in a lab).

## About the input molecules

The full set of generated `.mol` files (100,000 structures) is not included
in this repo — the file count and size are too large for GitHub.

To reproduce them, just run script 01. It uses a fixed random seed, so it
will generate the same structures every time.

If you want to test the pipeline quickly before running the full 100,000,
try generating a small batch first:

```
python 01_structure_generation.py --target 100
```

This creates 100 structures in a few seconds so you can check everything
is working before running the full run. Once that looks good, run the
remaining scripts in order (02 through 11) on the small batch to make sure
the whole pipeline runs end to end, then scale back up to the full
100,000 for the real run.

## Results included in this repo

- `xtb_results/5000/` — the initial xTB-labeled dataset used to train and
  test the ML models
- `xtb_results/11000/` — the ML-predicted and xTB-validated results for the
  11,000 selected molecules
- `top_50_candidates/` — the final 50 candidates selected for TD-DFT
  validation, as reported in the paper
