"""C-MAPSS (NASA Turbofan Engine Degradation) Benchmark

Downloads the original NASA C-MAPSS FD001 dataset and uses it for
Remaining Useful Life (RUL) prediction via SSM + MAML meta-learning.

Data source:
    Original NASA C-MAPSS data files (FD001 subset)
    Downloaded from GitHub mirror of NASA's Prognostics Data Repository.
    NASA's official download portal (data.nasa.gov) is currently under
    management review (as of 2024+). The GitHub mirror contains the
    identical original files.

Dataset structure (FD001):
    - 100 training engines (run-to-failure)
    - 100 test engines (stopped before failure)
    - 21 sensors + 3 operational settings per timestep
    - Ground truth RUL for test engines

Reference:
    A. Saxena, K. Goebel, D. Simon, N. Eklund (2008).
    "Damage propagation modeling for aircraft engine run-to-failure simulation."
    International Conference on Prognostics and Health Management (PHM).

Usage:
    python experiments/benchmark_cmapss.py
"""

import sys
import os
import time
import json
import platform
import logging
import urllib.request
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.ssm import StateSpaceModel
from core.ssm_mamba import MambaSSM
from meta_rl.meta_maml import MetaMAML

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================================
# C-MAPSS Data Download & Preprocessing
# ============================================================================

CMAPSS_BASE_URL = (
    "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/"
    "master/notebooks/jupyter/Dataset/CMAPSSData/"
)

CMAPSS_FILES = {
    'train': 'train_FD001.txt',
    'test': 'test_FD001.txt',
    'rul': 'RUL_FD001.txt',
}

# Column names for the C-MAPSS dataset
COLUMN_NAMES = (
    ['unit_id', 'cycle'] +
    [f'op_setting_{i}' for i in range(1, 4)] +
    [f'sensor_{i}' for i in range(1, 22)]
)

# Sensors known to be informative (constant sensors are dropped)
INFORMATIVE_SENSORS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
    'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
    'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21',
]

# Maximum RUL cap (standard practice in C-MAPSS literature)
MAX_RUL = 125

# Reduced mode for pipeline validation
REDUCED_MODE = False


def download_cmapss(data_dir: Path) -> Dict[str, Path]:
    """Download C-MAPSS FD001 files from NASA data mirror.

    Args:
        data_dir: Directory to save downloaded files

    Returns:
        Dictionary mapping file types to local paths
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    file_paths = {}

    for key, filename in CMAPSS_FILES.items():
        filepath = data_dir / filename

        if filepath.exists():
            logger.info(f"  Found cached: {filename}")
            file_paths[key] = filepath
            continue

        url = CMAPSS_BASE_URL + filename
        logger.info(f"  Downloading: {url}")

        try:
            urllib.request.urlretrieve(url, filepath)
            file_size = filepath.stat().st_size
            logger.info(f"  Saved: {filename} ({file_size:,} bytes)")
            file_paths[key] = filepath
        except Exception as e:
            logger.error(f"  Failed to download {filename}: {e}")
            raise RuntimeError(
                f"Cannot download C-MAPSS data from {url}. "
                f"Please check your internet connection."
            )

    return file_paths


def parse_cmapss_file(filepath: Path) -> np.ndarray:
    """Parse a C-MAPSS data file (space-separated, no header).

    Args:
        filepath: Path to the data file

    Returns:
        Numpy array with all columns
    """
    data = np.loadtxt(filepath)
    return data


def preprocess_cmapss(
    file_paths: Dict[str, Path],
    window_size: int = 50,
) -> Dict[str, any]:
    """Full preprocessing pipeline for C-MAPSS FD001.

    Steps:
        1. Parse raw text files
        2. Compute RUL for training data
        3. Drop constant/non-informative sensors
        4. Normalize features (min-max per column)
        5. Create sliding window sequences
        6. Cap RUL at MAX_RUL

    Args:
        file_paths: Dictionary from download_cmapss()
        window_size: Sliding window length for sequences

    Returns:
        Dictionary with processed training and test data
    """
    logger.info("Preprocessing C-MAPSS FD001...")

    # Parse files
    train_raw = parse_cmapss_file(file_paths['train'])
    test_raw = parse_cmapss_file(file_paths['test'])
    rul_raw = np.loadtxt(file_paths['rul'])

    logger.info(f"  Train: {train_raw.shape} ({int(train_raw[:, 0].max())} engines)")
    logger.info(f"  Test:  {test_raw.shape} ({int(test_raw[:, 0].max())} engines)")
    logger.info(f"  RUL:   {rul_raw.shape}")

    # Extract unit IDs and cycle counts
    train_units = train_raw[:, 0].astype(int)
    train_cycles = train_raw[:, 1].astype(int)
    test_units = test_raw[:, 0].astype(int)

    # Feature columns: operational settings (3) + sensors (21) = columns 2-25
    # We use only informative sensors
    sensor_start_idx = 5  # first sensor is column index 5
    informative_indices = [int(s.split('_')[1]) - 1 + sensor_start_idx
                           for s in INFORMATIVE_SENSORS]

    # Also include operational settings (columns 2, 3, 4)
    feature_indices = [2, 3, 4] + informative_indices
    num_features = len(feature_indices)

    train_features = train_raw[:, feature_indices]
    test_features = test_raw[:, feature_indices]

    logger.info(f"  Selected {num_features} features (3 ops + {len(INFORMATIVE_SENSORS)} sensors)")

    # Compute RUL for training data
    # RUL = max_cycle_for_unit - current_cycle
    train_rul = np.zeros(len(train_raw))
    for unit_id in np.unique(train_units):
        mask = train_units == unit_id
        max_cycle = train_cycles[mask].max()
        train_rul[mask] = max_cycle - train_cycles[mask]

    # Cap RUL at MAX_RUL
    train_rul = np.clip(train_rul, 0, MAX_RUL)

    # Normalize features using training statistics
    feature_min = train_features.min(axis=0)
    feature_max = train_features.max(axis=0)
    feature_range = feature_max - feature_min
    feature_range[feature_range == 0] = 1.0  # avoid division by zero

    train_features_norm = (train_features - feature_min) / feature_range
    test_features_norm = (test_features - feature_min) / feature_range

    # Normalize RUL target to [0, 1]
    rul_max = MAX_RUL
    train_rul_norm = train_rul / rul_max

    # Create sliding window sequences per engine
    def create_windows(features, rul, unit_ids, window_size):
        """Create sliding window sequences for each engine unit."""
        windows_x = []
        windows_y = []
        unit_labels = []

        for uid in np.unique(unit_ids):
            mask = unit_ids == uid
            feat = features[mask]
            r = rul[mask]

            if len(feat) < window_size:
                # Pad short sequences
                pad_len = window_size - len(feat)
                feat = np.vstack([np.zeros((pad_len, feat.shape[1])), feat])
                r = np.concatenate([np.full(pad_len, r[0]), r])

            for i in range(len(feat) - window_size + 1):
                windows_x.append(feat[i:i + window_size])
                windows_y.append(r[i + window_size - 1])
                unit_labels.append(uid)

        return (
            np.array(windows_x),
            np.array(windows_y),
            np.array(unit_labels),
        )

    train_x, train_y, train_uid = create_windows(
        train_features_norm, train_rul_norm, train_units, window_size
    )

    # For test: create last window per engine, use provided RUL
    test_last_windows = []
    test_rul_targets = []
    for i, uid in enumerate(np.unique(test_units)):
        mask = test_units == uid
        feat = test_features_norm[mask]

        if len(feat) >= window_size:
            window = feat[-window_size:]
        else:
            pad_len = window_size - len(feat)
            window = np.vstack([np.zeros((pad_len, feat.shape[1])), feat])

        test_last_windows.append(window)
        test_rul_targets.append(min(rul_raw[i], MAX_RUL) / rul_max)

    test_x = np.array(test_last_windows)
    test_y = np.array(test_rul_targets)

    logger.info(f"  Train windows: {train_x.shape}, targets: {train_y.shape}")
    logger.info(f"  Test windows:  {test_x.shape}, targets: {test_y.shape}")

    return {
        'train_x': train_x,
        'train_y': train_y,
        'train_unit_ids': train_uid,
        'test_x': test_x,
        'test_y': test_y,
        'test_rul_raw': rul_raw,
        'num_features': num_features,
        'feature_names': ['op_1', 'op_2', 'op_3'] + INFORMATIVE_SENSORS,
        'rul_max': rul_max,
        'window_size': window_size,
        'num_train_engines': int(train_units.max()),
        'num_test_engines': int(test_units.max()),
    }


# ============================================================================
# C-MAPSS Scoring
# ============================================================================

def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the official C-MAPSS scoring function.

    This is the asymmetric scoring function from the original paper:
        s = sum(exp(-d/13) - 1)  if d < 0 (early prediction)
        s = sum(exp(d/10) - 1)   if d >= 0 (late prediction)

    Late predictions are penalized more heavily than early ones.

    Args:
        y_true: True RUL values (unnormalized)
        y_pred: Predicted RUL values (unnormalized)

    Returns:
        C-MAPSS score (lower is better)
    """
    d = y_pred - y_true
    score = 0.0
    for di in d:
        if di < 0:
            score += np.exp(-di / 13.0) - 1
        else:
            score += np.exp(di / 10.0) - 1
    return score


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_cmapss_benchmark(
    model_type: str,
    data: Dict,
    num_epochs: int = 30,
    tasks_per_epoch: int = 8,
) -> Dict:
    """Run C-MAPSS RUL prediction benchmark for one model type.

    Uses MAML meta-learning where each "task" is a different engine unit.

    Args:
        model_type: 'legacy' or 'mamba'
        data: Preprocessed data from preprocess_cmapss()
        num_epochs: MAML meta-training epochs
        tasks_per_epoch: Number of engine tasks per meta-batch

    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"{'='*60}")
    logger.info(f"C-MAPSS Benchmark: {model_type}")
    logger.info(f"{'='*60}")

    num_features = data['num_features']

    # Create model — input is sensor sequence, output is RUL scalar
    if model_type == 'mamba':
        model = MambaSSM(
            state_dim=16,
            input_dim=num_features,
            output_dim=1,
            d_model=64,
        )
    else:
        model = StateSpaceModel(
            state_dim=16,
            input_dim=num_features,
            output_dim=1,
            hidden_dim=64,
        )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_type}, Parameters: {total_params:,}")
    logger.info(f"Input dim: {num_features}, Output dim: 1 (RUL)")

    # MAML meta-learner
    meta_learner = MetaMAML(
        model=model,
        inner_lr=0.005,
        outer_lr=0.001,
    )

    # Group training data by engine unit for task-based meta-learning
    unique_units = np.unique(data['train_unit_ids'])
    unit_data = {}
    for uid in unique_units:
        mask = data['train_unit_ids'] == uid
        unit_data[uid] = {
            'x': data['train_x'][mask],
            'y': data['train_y'][mask],
        }

    # Training loop
    epoch_losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        # Sample task batch (each task = one engine)
        sampled_units = np.random.choice(
            unique_units,
            size=min(tasks_per_epoch, len(unique_units)),
            replace=False,
        )

        maml_tasks = []
        for uid in sampled_units:
            ud = unit_data[uid]
            n_samples = len(ud['x'])
            if n_samples < 4:
                continue

            # Split into support/query
            split = n_samples // 2

            # Input: sensor windows (batch, time, features)
            sx = torch.tensor(ud['x'][:split], dtype=torch.float32)
            qx = torch.tensor(ud['x'][split:], dtype=torch.float32)

            # Target: RUL values (batch, time, 1) — expand to match sequence output
            # Model outputs (batch, time, 1), so we need target at each timestep
            # Use the last timestep's RUL as target for the full window
            sy = torch.tensor(
                ud['y'][:split], dtype=torch.float32
            ).unsqueeze(-1).unsqueeze(-1).expand(-1, ud['x'].shape[1], 1)
            qy = torch.tensor(
                ud['y'][split:], dtype=torch.float32
            ).unsqueeze(-1).unsqueeze(-1).expand(-1, ud['x'].shape[1], 1)

            maml_tasks.append((sx, sy, qx, qy))

        if not maml_tasks:
            continue

        initial_hidden = model.init_hidden(batch_size=1)
        loss = meta_learner.meta_update(
            maml_tasks,
            initial_hidden_state=initial_hidden,
            loss_fn=nn.MSELoss(),
        )
        epoch_losses.append(loss)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            logger.info(f"  Epoch {epoch:3d}/{num_epochs}: loss={loss:.6f}")

    training_time = time.time() - start_time

    # Evaluation on test set
    model.eval()
    test_x = torch.tensor(data['test_x'], dtype=torch.float32)
    test_y_norm = data['test_y']
    test_y_raw = data['test_rul_raw']

    with torch.no_grad():
        hidden = model.init_hidden(batch_size=test_x.shape[0])
        pred, _ = model(test_x, hidden)
        # Take the last timestep's prediction
        pred_rul_norm = pred[:, -1, 0].numpy()

    # Denormalize predictions
    pred_rul = pred_rul_norm * data['rul_max']
    pred_rul = np.clip(pred_rul, 0, MAX_RUL)

    true_rul = np.clip(test_y_raw, 0, MAX_RUL)

    # Metrics
    rmse = float(np.sqrt(np.mean((pred_rul - true_rul) ** 2)))
    mae = float(np.mean(np.abs(pred_rul - true_rul)))
    score = cmapss_score(true_rul, pred_rul)

    result = {
        'model_type': model_type,
        'total_params': total_params,
        'num_epochs': num_epochs,
        'initial_loss': epoch_losses[0] if epoch_losses else float('inf'),
        'final_loss': epoch_losses[-1] if epoch_losses else float('inf'),
        'best_loss': min(epoch_losses) if epoch_losses else float('inf'),
        'training_time_sec': training_time,
        'test_rmse': rmse,
        'test_mae': mae,
        'cmapss_score': score,
        'loss_curve': epoch_losses,
        'sample_predictions': {
            'true_rul': true_rul[:10].tolist(),
            'pred_rul': pred_rul[:10].tolist(),
        },
    }

    logger.info(f"\n  Results for {model_type}:")
    logger.info(f"    Final loss:    {result['final_loss']:.6f}")
    logger.info(f"    Training time: {training_time:.2f}s")
    logger.info(f"    Test RMSE:     {rmse:.2f} cycles")
    logger.info(f"    Test MAE:      {mae:.2f} cycles")
    logger.info(f"    C-MAPSS Score: {score:.2f}")
    logger.info(f"    Sample predictions (first 5):")
    for i in range(min(5, len(true_rul))):
        logger.info(f"      Engine {i+1}: true={true_rul[i]:.0f}, pred={pred_rul[i]:.1f}")

    return result


def main():
    print("=" * 70)
    print("  Benchmark 2: C-MAPSS (NASA Turbofan Engine Degradation)")
    print("  RUL Prediction — FD001 Dataset")
    print("  Source: NASA Prognostics Data Repository (GitHub mirror)")
    print("=" * 70)

    # Download data
    data_dir = project_root / 'data' / 'cmapss'
    print(f"\nDownloading C-MAPSS FD001 data to: {data_dir}")
    file_paths = download_cmapss(data_dir)

    # Verify data integrity
    for key, path in file_paths.items():
        size = path.stat().st_size
        print(f"  {key}: {path.name} ({size:,} bytes)")

    # Preprocess
    data = preprocess_cmapss(file_paths, window_size=50)

    print(f"\nDataset summary:")
    print(f"  Training engines: {data['num_train_engines']}")
    print(f"  Test engines:     {data['num_test_engines']}")
    print(f"  Features:         {data['num_features']} ({', '.join(data['feature_names'][:5])}...)")
    print(f"  Window size:      {data['window_size']}")
    print(f"  Max RUL cap:      {data['rul_max']}")
    print()

    # Run benchmarks
    num_epochs = 5 if REDUCED_MODE else 100
    if REDUCED_MODE:
        print(f"!!! REDUCED MODE ENABLED: Running {num_epochs} epochs for validation !!!")
        
    results = {}
    for model_type in ['legacy', 'mamba']:
        result = run_cmapss_benchmark(
            model_type=model_type,
            data=data,
            num_epochs=num_epochs,
            tasks_per_epoch=8,
        )
        results[model_type] = result
        print()

    # Comparison table
    print("\n" + "=" * 70)
    print("  C-MAPSS BENCHMARK RESULTS (FD001)")
    print("=" * 70)
    print(f"{'Metric':<25} {'Legacy SSM':>15} {'Mamba SSM':>15}")
    print("-" * 55)

    legacy = results['legacy']
    mamba = results['mamba']

    rows = [
        ('Parameters', f"{legacy['total_params']:,}", f"{mamba['total_params']:,}"),
        ('Final Loss', f"{legacy['final_loss']:.6f}", f"{mamba['final_loss']:.6f}"),
        ('Training Time (s)', f"{legacy['training_time_sec']:.2f}", f"{mamba['training_time_sec']:.2f}"),
        ('Test RMSE (cycles)', f"{legacy['test_rmse']:.2f}", f"{mamba['test_rmse']:.2f}"),
        ('Test MAE (cycles)', f"{legacy['test_mae']:.2f}", f"{mamba['test_mae']:.2f}"),
        ('C-MAPSS Score', f"{legacy['cmapss_score']:.2f}", f"{mamba['cmapss_score']:.2f}"),
    ]

    for label, v1, v2 in rows:
        print(f"{label:<25} {v1:>15} {v2:>15}")

    print("=" * 70)

    # Save results
    results_path = project_root / 'results' / 'cmapss_benchmark.json'
    results_path.parent.mkdir(exist_ok=True)

    save_data = {
        'benchmark': 'C-MAPSS NASA Turbofan Engine Degradation (FD001)',
        'data_source': CMAPSS_BASE_URL,
        'description': (
            'RUL prediction on NASA C-MAPSS FD001 dataset. '
            '100 training engines (run-to-failure), 100 test engines. '
            '21 sensors + 3 operational settings. '
            'RUL capped at 125 cycles (standard practice).'
        ),
        'platform': {
            'system': platform.system(),
            'processor': platform.processor(),
            'python': platform.python_version(),
            'torch': torch.__version__,
            'device': 'cpu',
        },
        'config': {
            'window_size': data['window_size'],
            'num_features': data['num_features'],
            'max_rul': data['rul_max'],
            'num_epochs': num_epochs,
            'tasks_per_epoch': 8,
            'reduced_mode': REDUCED_MODE,
            'num_train_engines': data['num_train_engines'],
            'num_test_engines': data['num_test_engines'],
        },
        'results': {
            model_type: {k: v for k, v in res.items() if k != 'loss_curve'}
            for model_type, res in results.items()
        },
        'loss_curves': {
            model_type: res['loss_curve']
            for model_type, res in results.items()
        },
    }

    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
