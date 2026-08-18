#!/usr/bin/env python3
"""
Train machine learning models to predict the HOMO-LUMO gap

Trains five regression models on the 4,000-molecule training set (Random
Forest, XGBoost, LightGBM, CatBoost, and a neural network), evaluates each
on the 1,000-molecule test set, and picks the best one based on test R².

Input:
  - molecular_features.parquet (features for all molecules)
  - gap_values_train.csv (4,000 training molecules with xTB gaps)
  - gap_values_test.csv (1,000 test molecules with xTB gaps)

Output:
  - RandomForest_model.pkl
  - XGBoost_model.pkl
  - LightGBM_model.pkl
  - CatBoost_model.pkl
  - NeuralNetwork_model.pkl
  - feature_scaler.pkl
  - training_metrics.json

Usage:
  python 07_ml_model_training.py
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import json
import pickle
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from sklearn.neural_network import MLPRegressor
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path.home() / "GQD_Project"
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = DATA_DIR / "train_test_split"
FEATURES_DIR = PROJECT_ROOT / "features"
RESULTS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

FEATURES_PARQUET = FEATURES_DIR / "molecular_features.parquet"
TRAIN_GAP_CSV = PROJECT_ROOT / "xTB_results" / "gap_values_train.csv"
TEST_GAP_CSV = PROJECT_ROOT / "xTB_results" / "gap_values_test.csv"
TRAIN_CSV = SPLIT_DIR / "train_molecules.csv"
TEST_CSV = SPLIT_DIR / "test_molecules.csv"

TRAINING_LOG = RESULTS_DIR / "training_metrics.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class StageLogger:
    """Logs to both a file and the console, flushing after each line so
    output shows up immediately when running under nohup."""

    def __init__(self, log_file):
        self.log_file = log_file
        self.handlers = []

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.handlers.append(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.handlers.append(console_handler)

        self.logger = logging.getLogger('ml_training')
        self.logger.setLevel(logging.INFO)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        for handler in self.handlers:
            self.logger.addHandler(handler)

    def info(self, msg):
        self.logger.info(msg)
        sys.stdout.flush()

    def error(self, msg):
        self.logger.error(msg)
        sys.stderr.flush()

logger = StageLogger(LOGS_DIR / "ml_model_training.log")


def log_section(title):
    logger.info("")
    logger.info("="*80)
    logger.info(title)
    logger.info("="*80)


def load_training_data():
    """Load molecular features and xTB gap values for the training set."""
    log_section("LOAD TRAINING DATA")

    logger.info("Loading features...")
    df_features = pd.read_parquet(FEATURES_PARQUET)
    logger.info(f"Features shape: {df_features.shape}")

    feature_cols = [c for c in df_features.columns
                   if c != 'mol_id' and pd.api.types.is_numeric_dtype(df_features[c])]
    logger.info(f"Numeric feature columns: {len(feature_cols)}")

    logger.info("Loading training molecule IDs...")
    df_train = pd.read_csv(TRAIN_CSV)
    train_ids = set(df_train['mol_id'].values)
    logger.info(f"Training molecules: {len(train_ids)}")

    logger.info("Loading gap values...")
    df_gaps = pd.read_csv(TRAIN_GAP_CSV)
    df_gaps_success = df_gaps[df_gaps['status'] == 'success'].copy()
    logger.info(f"Successful gaps: {len(df_gaps_success)}")

    logger.info("Filtering to training molecules...")
    df_train_features = df_features[['mol_id'] + feature_cols]
    df_train_features = df_train_features[df_train_features['mol_id'].isin(train_ids)].copy()

    logger.info("Merging features with gap values...")
    df_merged = df_train_features.merge(
        df_gaps_success[['mol_id', 'HOMO_LUMO_gap_eV']],
        on='mol_id',
        how='inner'
    )

    X_train = df_merged[feature_cols].values
    y_train = df_merged['HOMO_LUMO_gap_eV'].values

    logger.info(f"Training set: {X_train.shape[0]} molecules x {X_train.shape[1]} features")
    logger.info(f"Gap range: {y_train.min():.4f} - {y_train.max():.4f} eV")

    return X_train, y_train, feature_cols


def load_test_data(feature_cols):
    """Load molecular features and xTB gap values for the test set."""
    log_section("LOAD TEST DATA")

    logger.info("Loading features...")
    df_features = pd.read_parquet(FEATURES_PARQUET)

    logger.info("Loading test molecule IDs...")
    df_test = pd.read_csv(TEST_CSV)
    test_ids = set(df_test['mol_id'].values)
    logger.info(f"Test molecules: {len(test_ids)}")

    logger.info("Loading gap values...")
    df_gaps = pd.read_csv(TEST_GAP_CSV)
    df_gaps_success = df_gaps[df_gaps['status'] == 'success'].copy()
    logger.info(f"Successful gaps: {len(df_gaps_success)}")

    logger.info("Filtering to test molecules...")
    df_test_features = df_features[['mol_id'] + feature_cols]
    df_test_features = df_test_features[df_test_features['mol_id'].isin(test_ids)].copy()

    logger.info("Merging features with gap values...")
    df_merged = df_test_features.merge(
        df_gaps_success[['mol_id', 'HOMO_LUMO_gap_eV']],
        on='mol_id',
        how='inner'
    )

    X_test = df_merged[feature_cols].values
    y_test = df_merged['HOMO_LUMO_gap_eV'].values

    logger.info(f"Test set: {X_test.shape[0]} molecules x {X_test.shape[1]} features")
    logger.info(f"Gap range: {y_test.min():.4f} - {y_test.max():.4f} eV")

    return X_test, y_test


def normalize_features(X_train, X_test):
    """Clean up and standardize the feature matrices before training."""
    log_section("NORMALIZE FEATURES")

    logger.info("Converting to float64...")
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    logger.info("Handling infinity values...")
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)

    train_nan = np.isnan(X_train).sum()
    test_nan = np.isnan(X_test).sum()
    logger.info(f"Found {train_nan} NaN in training, {test_nan} NaN in test")

    if train_nan > 0 or test_nan > 0:
        logger.info("Replacing NaN values with column median...")
        for col in range(X_train.shape[1]):
            col_data = X_train[:, col]
            col_median = np.nanmedian(col_data)
            if np.isnan(col_median):
                col_median = 0.0
            X_train[np.isnan(X_train[:, col]), col] = col_median
            X_test[np.isnan(X_test[:, col]), col] = col_median

    train_max = np.max(np.abs(X_train))
    test_max = np.max(np.abs(X_test))
    logger.info(f"Max absolute value - train: {train_max:.2e}, test: {test_max:.2e}")

    if train_max > 1e6 or test_max > 1e6:
        logger.info("Clipping very large values to [-1e6, 1e6]...")
        X_train = np.clip(X_train, -1e6, 1e6)
        X_test = np.clip(X_test, -1e6, 1e6)

    logger.info("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"Train shape: {X_train_scaled.shape}")
    logger.info(f"Test shape: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, scaler


def train_models(X_train, y_train, X_test, y_test):
    """Train each available model and record its performance on the test set."""
    log_section("TRAINING MODELS")

    results = {}

    logger.info("Training RandomForest...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train, y_train)
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)

    logger.info(f"  R2 train: {r2_train:.4f} | R2 test: {r2_test:.4f}")
    logger.info(f"  RMSE: {rmse_test:.4f} eV | MAE: {mae_test:.4f} eV")

    results['RandomForest'] = {
        'model': rf, 'r2_train': r2_train, 'r2_test': r2_test,
        'rmse': rmse_test, 'mae': mae_test
    }

    if XGBOOST_AVAILABLE:
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        y_pred_train = xgb_model.predict(X_train)
        y_pred_test = xgb_model.predict(X_test)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)

        logger.info(f"  R2 train: {r2_train:.4f} | R2 test: {r2_test:.4f}")
        logger.info(f"  RMSE: {rmse_test:.4f} eV | MAE: {mae_test:.4f} eV")

        results['XGBoost'] = {
            'model': xgb_model, 'r2_train': r2_train, 'r2_test': r2_test,
            'rmse': rmse_test, 'mae': mae_test
        }

    if LIGHTGBM_AVAILABLE:
        logger.info("Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        y_pred_train = lgb_model.predict(X_train)
        y_pred_test = lgb_model.predict(X_test)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)

        logger.info(f"  R2 train: {r2_train:.4f} | R2 test: {r2_test:.4f}")
        logger.info(f"  RMSE: {rmse_test:.4f} eV | MAE: {mae_test:.4f} eV")

        results['LightGBM'] = {
            'model': lgb_model, 'r2_train': r2_train, 'r2_test': r2_test,
            'rmse': rmse_test, 'mae': mae_test
        }

    if CATBOOST_AVAILABLE:
        logger.info("Training CatBoost...")
        cb_model = cb.CatBoostRegressor(
            iterations=200,
            depth=7,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=False
        )
        cb_model.fit(X_train, y_train)
        y_pred_train = cb_model.predict(X_train)
        y_pred_test = cb_model.predict(X_test)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)

        logger.info(f"  R2 train: {r2_train:.4f} | R2 test: {r2_test:.4f}")
        logger.info(f"  RMSE: {rmse_test:.4f} eV | MAE: {mae_test:.4f} eV")

        results['CatBoost'] = {
            'model': cb_model, 'r2_train': r2_train, 'r2_test': r2_test,
            'rmse': rmse_test, 'mae': mae_test
        }

    if NEURAL_AVAILABLE:
        logger.info("Training Neural Network (hidden layers: 256, 128, 64)...")
        nn_model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            max_iter=1000,
            learning_rate_init=0.001,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=0
        )
        nn_model.fit(X_train, y_train)
        y_pred_train = nn_model.predict(X_train)
        y_pred_test = nn_model.predict(X_test)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)

        logger.info(f"  R2 train: {r2_train:.4f} | R2 test: {r2_test:.4f}")
        logger.info(f"  RMSE: {rmse_test:.4f} eV | MAE: {mae_test:.4f} eV")

        results['NeuralNetwork'] = {
            'model': nn_model, 'r2_train': r2_train, 'r2_test': r2_test,
            'rmse': rmse_test, 'mae': mae_test
        }
    else:
        logger.info("Neural network not available (sklearn MLPRegressor missing)")

    return results


def select_best_model(results):
    """Compare all trained models and pick the one with the highest test R²."""
    log_section("MODEL COMPARISON")

    logger.info(f"{'Model':<20} {'R2 Train':<12} {'R2 Test':<12} {'RMSE':<12} {'MAE':<12}")
    logger.info("-"*70)

    best_model_name = None
    best_r2 = -np.inf

    for name in sorted(results.keys()):
        data = results[name]
        r2 = data['r2_test']

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

        logger.info(f"{name:<20} {data['r2_train']:<12.4f} {r2:<12.4f} {data['rmse']:<12.4f} {data['mae']:<12.4f}")

    logger.info("-"*70)
    logger.info(f"Best model: {best_model_name} (R2 = {best_r2:.4f})")

    return best_model_name, results[best_model_name]


def save_all_models(results, scaler):
    """Save every trained model and the feature scaler to disk."""
    log_section("SAVE MODELS")

    for model_name, data in results.items():
        model = data['model']
        model_file = RESULTS_DIR / f"{model_name}_model.pkl"

        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved: {model_file}")

    scaler_file = RESULTS_DIR / "feature_scaler.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved: {scaler_file}")


def save_training_metrics(results, feature_names):
    """Save a summary of every model's metrics as JSON."""
    log_section("SAVE TRAINING METRICS")

    metrics_data = {
        'timestamp': datetime.now().isoformat(),
        'training_set_size': 4000,
        'test_set_size': 1000,
        'feature_count': len(feature_names),
        'models': {}
    }

    best_model_name = max(results.keys(), key=lambda k: results[k]['r2_test'])
    metrics_data['best_model'] = best_model_name

    for name, data in sorted(results.items()):
        metrics_data['models'][name] = {
            'r2_train': float(data['r2_train']),
            'r2_test': float(data['r2_test']),
            'rmse_test': float(data['rmse']),
            'mae_test': float(data['mae'])
        }

    with open(TRAINING_LOG, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    logger.info(f"Saved: {TRAINING_LOG}")


def main():
    log_section("ML MODEL TRAINING")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    X_train, y_train, feature_cols = load_training_data()
    X_test, y_test = load_test_data(feature_cols)

    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    results = train_models(X_train_scaled, y_train, X_test_scaled, y_test)

    best_model_name, best_result = select_best_model(results)

    save_all_models(results, scaler)
    save_training_metrics(results, feature_cols)

    log_section("TRAINING COMPLETE")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Models trained:")
    for name in sorted(results.keys()):
        logger.info(f"  - {name}")
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"  R2:   {best_result['r2_test']:.4f}")
    logger.info(f"  RMSE: {best_result['rmse']:.4f} eV")
    logger.info(f"  MAE:  {best_result['mae']:.4f} eV")


if __name__ == "__main__":
    main()
