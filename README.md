# State Space Models under Switching Dynamics and Meta-Adaptation

This repository contains experimental implementations of State Space Models (SSMs) evaluated under distributional switching conditions and task-adaptive settings.

The primary objective is to investigate numerical stability, adaptation behavior, and evaluation methodology rather than to claim state-of-the-art performance.

---

## Scope

This project explores:

- Switching Linear Dynamical System (LDS) benchmarks with aligned shock evaluation metrics.
- Output-restricted test-time adaptation (TTA) for Mamba-style SSMs.
- Stability-oriented training rules (low learning rate, gradient clipping, frozen core scan parameters).
- C-MAPSS (NASA Turbofan FD001) for Remaining Useful Life (RUL) prediction.
- Meta-learning scaffolding (MetaMAML and related baselines).

The repository functions as a research sandbox for probing stability and adaptation under constrained conditions.

---

## Current Status

### 1. Switching LDS Benchmark
- Switch-aligned metrics implemented (Pre / Shock / Recover / Steady / AUA / RecTime).
- Hidden state persistence verified.
- Mamba TTA numerical instability resolved via:
  - Output-only parameter updates
  - Reduced learning rate (0.001)
  - Gradient clipping (1.0)

Shock-phase stability constraint verified:
```
Shock_ON ≤ 2.0 × Shock_OFF
```

Positive adaptation gain remains under investigation.

---

### 2. C-MAPSS Benchmark (FD001, Full Run)

**Dataset**: NASA C-MAPSS FD001 — 100 training engines (run-to-failure), 100 test engines. 17 sensor features, window size 50, RUL capped at 125 cycles (standard practice).

**Training**: 100 epochs, MAML meta-learning with 8 tasks per epoch.

| Metric | Legacy SSM | Mamba SSM |
| :--- | ---: | ---: |
| Parameters | 3,587 | 65,729 |
| Final Training Loss | 0.0182 | 0.0361 |
| Best Training Loss | 0.0117 | 0.0169 |
| Training Time (CPU) | 141 s | 3,309 s |
| **Test RMSE (cycles)** | **46.24** | **21.80** |
| **Test MAE (cycles)** | **38.36** | **18.22** |
| **C-MAPSS Score** | **10,806** | **906** |

**Observations** (subject to further validation):

- On the test set, the Mamba-based model achieved approximately 53% lower RMSE and a substantially lower C-MAPSS score compared to the MLP-based baseline, suggesting that the structured state-space formulation may offer representational advantages for sequential degradation modeling.
- Mamba's higher training time on CPU is attributable to the pure-PyTorch fallback implementation; the official CUDA kernel (`mamba-ssm`) is expected to reduce this significantly on GPU hardware.
- These results are from a single-seed run. Multi-seed validation with statistical significance testing is required before drawing definitive conclusions.

> **Note**: The C-MAPSS scoring function is asymmetric — late predictions (predicting longer RUL than actual) are penalized more heavily than early predictions. The substantial difference in scores (906 vs. 10,806) reflects Mamba's tendency to produce predictions closer to true RUL values, but further analysis of the error distribution is warranted.

---

## Reproducibility

Switching LDS benchmark:
```bash
python experiments/benchmark_switching_lds.py
```

C-MAPSS benchmark (full 100-epoch run):
```bash
python experiments/benchmark_cmapss.py
```

Artifacts generated:
* `results/switching_lds_benchmark.json`
* `results/cmapss_benchmark.json`
* `model_legacy_lds.pt` / `model_mamba_lds.pt`

Large artifacts are excluded from version control where appropriate.

---

## Design Philosophy

This repository prioritizes:

* Stability verification over performance claims
* Explicit constraint logging
* Incremental validation
* Clear separation between preliminary and confirmed results

No claims of scalability or production readiness are made.

---

## Disclaimer

This is experimental research code.
Architectural and performance conclusions remain provisional pending full multi-seed validation and extended training runs.
The benchmark results reported above are from a single experimental configuration and should be interpreted with appropriate caution.
