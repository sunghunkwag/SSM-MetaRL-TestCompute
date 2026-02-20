# State Space Models under Switching Dynamics and Meta-Adaptation

This repository contains experimental implementations of State Space Models (SSMs) evaluated under distributional switching conditions and task-adaptive settings.

The primary objective is to investigate numerical stability, adaptation behavior, and evaluation methodology rather than to claim state-of-the-art performance.

---

## Scope

This project explores:

- Switching Linear Dynamical System (LDS) benchmarks with aligned shock evaluation metrics.
- Output-restricted test-time adaptation (TTA) for Mamba-style SSMs.
- Stability-oriented training rules (low learning rate, gradient clipping, frozen core scan parameters).
- Reduced-mode validation of C-MAPSS (NASA Turbofan FD001) for Remaining Useful Life (RUL) prediction.
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

### 2. C-MAPSS (Reduced Mode)
- Dataset: FD001 (100 engines)
- Training: 5 epochs (pipeline validation only)
- Purpose: End-to-end shape handling and batching verification.

Results should be considered preliminary and not indicative of final performance.

---

## Reproducibility

Switching LDS benchmark:
```bash
python experiments/benchmark_switching_lds.py
```

Artifacts generated:
* `switching_lds_benchmark.json`
* `model_legacy_lds.pt`
* `model_mamba_lds.pt`

C-MAPSS reduced trial:
```bash
python experiments/benchmark_cmapss.py
```

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
