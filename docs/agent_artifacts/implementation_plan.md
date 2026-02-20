# SSM Benchmark: Switching LDS + C-MAPSS

## Overview
Two benchmarks evaluating Legacy SSM vs MambaSSM on real sequence modeling tasks:

1. **Main — Non-stationary Switching LDS (Synthetic)**: System switches between different linear dynamics regimes. Tests SSM's ability to track regime changes in real-time.
2. **Secondary — C-MAPSS (NASA Turbofan)**: Real NASA dataset, RUL prediction from 21 sensor readings. Downloaded from NASA's open data (GitHub mirror of original).

---

## Proposed Changes

### Benchmark 1: Switching LDS

#### [NEW] [benchmark_switching_lds.py](file:///c:/Users/starg/OneDrive/바탕 화면/test3/SSM-MetaRL-TestCompute/experiments/benchmark_switching_lds.py)

**Data generation** — no external data, pure math:
- Generate K regimes, each with distinct (A, B, C) matrices
- System switches regime every N steps (e.g., 50-100 steps)
- State: `x_{t+1} = A_k * x_t + B_k * u_t + noise`
- Observation: `y_t = C_k * x_t + noise`
- Task: predict next observation from current (one-step ahead)

**Training**:
- MAML meta-learning: each "task" = a different switching sequence
- Compare Legacy SSM vs MambaSSM
- Metrics: prediction MSE, regime-switch detection delay

---

### Benchmark 2: C-MAPSS NASA Turbofan

#### [NEW] [benchmark_cmapss.py](file:///c:/Users/starg/OneDrive/바탕 화면/test3/SSM-MetaRL-TestCompute/experiments/benchmark_cmapss.py)

**Data source**: NASA C-MAPSS original data files
- Download from: `https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData/`
- Files: `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
- 100 engines, 21 sensors, 3 operational settings per timestep
- This is the **original NASA dataset**, not a modified copy

> [!IMPORTANT]
> NASA's official download portal (data.nasa.gov) is currently under review. The GitHub mirror contains the identical original NASA C-MAPSS files (FD001). This is the same data used in thousands of published papers.

**Preprocessing**:
- Normalize sensor readings (min-max per column)
- Window sequences (length 50 timesteps)
- RUL target: clipped at 125 cycles (standard practice)

**Training**:
- MAML meta-learning: each "task" = a different engine unit
- Predict RUL from sensor sequence
- Compare Legacy SSM vs MambaSSM
- Metrics: RMSE, score function (standard C-MAPSS metric)

---

## Verification Plan

### Automated Tests
- `python experiments/benchmark_switching_lds.py` — full run, JSON results saved
- `python experiments/benchmark_cmapss.py` — Trial run (5 epochs, FD001) to validate pipeline, followed by full run (if trial passes).
- Both output comparison tables (Legacy vs Mamba)

## User Review Required

> [!IMPORTANT]
> **Refined Mamba TTA Constraints**:
> 1. **Learning Rate**: 0.001 (or lower).
> 2. **Gradient Clipping**: `norm=1.0`.
> 3. **Update Scope**: `output_projection` layer only (Mamba) / `output_network` layer only (Legacy).
> 4. **Freeze Scan Parameters**: All core/selective scan parameters are frozen during adaptation.
> 5. **Update Frequency**: Per-step adaptation.

> [!NOTE]
> **Success Criteria for Stability**:
> - **No Explosion**: Mamba ON `Shock` metric must not exceed 2× `Shock` of Mamba OFF.
> - **Efficiency**: `AUA(ON) <= AUA(OFF)` (Adaptation must improve or maintain overall MSE).

> [!WARNING]
> **C-MAPSS Batch Size Fix**: The meta-learning loop handles variable engine-specific batch sizes by dynamically expanding the initial hidden state [1, D, N] -> [B, D, N] to match inputs.
