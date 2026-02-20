# SSM Benchmark: Switching LDS + C-MAPSS

## Benchmark 1: Non-stationary Switching LDS (Synthetic)
- [x] 1.1 Implement `SwitchingLDS` data generator (K regimes, transitions)
- [/] 1.2 MAML training loop with Legacy SSM and MambaSSM
    - [x] State persistence sanity check for MambaSSM
    - [x] Live progress logging with ETA
- [x] 1.3 Aligned switch evaluation: Pre/Shock/Recover/Steady MSE + AUA
- [x] 1.4 Adaptation ON/OFF comparison with TTA update rule
    - [x] Refine Mamba TTA: 0.001 LR, clipping, output-only head updates
- [x] 1.5 JSON results output and Markdown table reporting
- [x] 1.6 Run and verify end-to-end (LDS re-run with stability constraints)

## Benchmark 2: C-MAPSS (NASA Turbofan)
- [x] 2.1 Download FD001 data from GitHub mirror (original NASA files)
- [x] 2.2 Preprocess: normalize sensors, window sequences, RUL targets
- [x] 2.3 MAML training loop with Legacy SSM and MambaSSM (Reduced 5-epoch trial)
    - [x] Fix batch size mismatch in `MetaMAML`
- [x] 2.4 Evaluation: RMSE, C-MAPSS score function
- [x] 2.5 JSON results output with comparison table
- [x] 2.6 Run and verify end-to-end
