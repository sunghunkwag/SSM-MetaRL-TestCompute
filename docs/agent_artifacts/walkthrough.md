# SSM Benchmark: Result Verification & Stability Fixes

This walkthrough confirms the resolution of the Mamba TTA instability and the C-MAPSS pipeline failures. Both benchmarks have been successfully executed with refined rules and generalized batch handling.

## 1. Switching LDS Stability Re-run
Verified the refined TTA strategy: **reduced LR (0.001)**, **gradient clipping (1.0)**, and **output-only parameter updates**.

### Stability Results (Mamba SSM)
| Mode | Pre MSE | Shock MSE | Steady MSE | AUA | RecTime (Δt) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TTA OFF** | 0.0368 | 0.0236 | 0.0244 | 0.4930 | 0.61 |
| **TTA ON** | 0.0372 | 0.0249 | 0.0240 | 0.5096 | 0.78 |

> [!IMPORTANT]
> **No Explosion Verified**: The `Shock(ON)` MSE (0.0249) is only **1.05×** the `Shock(OFF)` MSE (0.0236), well below the **2.0×** limit. Numerical stability is achieved.

## 2. C-MAPSS Reduced Trial (NASA Turbofan)
Successfully validated the C-MAPSS pipeline in reduced mode (5 epochs) across 100 engine units.

### Trial Metrics (Reduced Mode)
| Metric | Legacy SSM | Mamba SSM |
| :--- | :--- | :--- |
| Final Training Loss | 0.1331 | **0.0275** |
| Test RMSE (cycles) | **52.24** | 56.12 |
| C-MAPSS Score | 26,553 | **23,299** |

> [!NOTE]
> Mamba SSM shows significantly better training convergence (0.0275 vs 0.1331) even in just 5 epochs. The `MetaMAML` batch size expansion now correctly handles engine-specific data shapes.

## 3. Critical Fixes Implemented
- **Parameter Restriction**: Corrected `output_projection` vs `out_proj` naming to ensure core scan parameters remain frozen during TTA.
- **Batch Expansion**: Generalized `MetaMAML` to expand hidden states [1, D, N] -> [B, D, N] for task-based meta-learning.
- **Sequence Handling**: Fixed `StateSpaceModel` (Legacy) to support 3D sequence tensors during evaluation.

## Artifacts Generated
- [switching_lds_benchmark.json](file:///c:/Users/starg/OneDrive/바탕%20화면/test3/SSM-MetaRL-TestCompute/results/switching_lds_benchmark.json)
- [cmapss_benchmark.json](file:///c:/Users/starg/OneDrive/바탕%20화면/test3/SSM-MetaRL-TestCompute/results/cmapss_benchmark.json)
- [model_legacy_lds.pt](file:///c:/Users/starg/OneDrive/바탕%20화면/test3/SSM-MetaRL-TestCompute/results/model_legacy_lds.pt)
- [model_mamba_lds.pt](file:///c:/Users/starg/OneDrive/바탕%20화면/test3/SSM-MetaRL-TestCompute/results/model_mamba_lds.pt)

---

## Benchmark 2: C-MAPSS NASA Turbofan
**Status**: IN PROGRESS (Reduced Mode Trial)

- Target: FD001 (100 engines)
- Goal: RUL (Remaining Useful Life) prediction
- Reduced Mode: 5 epochs for pipeline validation
- Current Activity: Downloading and Preprocessing
