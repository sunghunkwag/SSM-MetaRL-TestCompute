# SSM-MetaRL-TestCompute

![CI](https://github.com/sunghunkwag/SSM-MetaRL-TestCompute/actions/workflows/ci.yml/badge.svg)

A research implementation combining State Space Models (SSM) with Meta-Reinforcement Learning for fast task adaptation. This repo provides a unified training/eval/adaptation pipeline with optional improvement hooks.

- Python: 3.9+
- Required: numpy
- Recommended: torch (CPU ok). The pipeline runs without torch but some modules may use it.

## Repository structure

- main.py: CLI entrypoint and orchestration
- core/ssm.py: SSM policy and config
- meta_rl/meta_maml.py: Meta-learner and config
- env_runner/environment.py: Vectorized environment batch runner and config
- adaptation/test_time_adaptation.py: Test-time adapter and config
- experiments/quick_benchmark.py: Quick benchmark for reproducible testing
- .github/workflows/ci.yml: CI/CD automation
- tests: minimal smoke/unit tests (added below)

## Quickstart

- Install: pip install numpy [torch]
- Train: python main.py train --config basic --outer-steps 100 --improve attention bn
- Eval: python main.py eval --config basic --checkpoint checkpoints/latest.npz --episodes 10
- Adapted eval: python main.py adapt --config basic --checkpoint checkpoints/latest.npz --episodes 10 --adapt-steps 5

## CI/CD & Testing

### Automated Testing

This repository includes GitHub Actions CI that automatically:
- Tests on Python 3.9, 3.10, and 3.11
- Installs dependencies (numpy, torch optional)
- Runs all `test_*.py` scripts

The CI workflow runs on every push and pull request to the `main` branch.

### Running Tests Locally

```bash
# Run all tests
python -m pytest -q  # if you have pytest installed

# Or run individual test files
python core/test_ssm.py
python meta_rl/test_meta_rl.py
python adaptation/test_adaptation.py
```

## Quick Benchmark & Reproducible Experiments

### Running the Quick Benchmark

The `experiments/quick_benchmark.py` script provides a fast way to validate the training pipeline and compare performance with different configurations:

```bash
python experiments/quick_benchmark.py
```

This benchmark:
- Runs multiple experiment configurations (baseline, with improvements, larger batch)
- Completes in under 2 minutes
- Provides performance comparison and timing metrics
- Validates the entire pipeline end-to-end

### Sample Output

```
╔═══════════════════════════════════════════════════════════╗
║   SSM-MetaRL Quick Benchmark                            ║
║   Testing pipeline & performance tuning                 ║
╚═══════════════════════════════════════════════════════════╝

============================================================
Running: Baseline (minimal)
============================================================
Command: python main.py --steps 100 --batch_size 32 --lr 0.001
✓ Completed in 28.45s

============================================================
Running: With improvements
============================================================
Command: python main.py --steps 100 --batch_size 32 --lr 0.001 --improve
✓ Completed in 31.20s

============================================================
Running: Larger batch
============================================================
Command: python main.py --steps 100 --batch_size 64 --lr 0.001
✓ Completed in 24.15s

============================================================
BENCHMARK SUMMARY
============================================================

Experiment                          Status     Time (s)
------------------------------------------------------------
Baseline (minimal)                  ✓ success     28.4
With improvements                   ✓ success     31.2
Larger batch                        ✓ success     24.1

============================================================

Performance Comparison:
  With improvements vs Baseline (minimal):
    Time difference: +2.75s (0.91x)
  Larger batch vs Baseline (minimal):
    Time difference: -4.30s (1.18x)

💡 Performance Tuning Tips:
   - Increase --batch_size for better GPU utilization
   - Adjust --lr based on convergence speed
   - Use --improve flag to enable optimizations
   - Monitor GPU memory with nvidia-smi

📊 For full experiments, increase --steps and --episodes

✓ All benchmarks passed!
```

## Performance Tuning Tips

### Batch Size Optimization

- **Small batches (16-32)**: Lower memory usage, faster iteration, potentially noisier gradients
- **Medium batches (64-128)**: Good balance for most GPUs, stable training
- **Large batches (256+)**: Better GPU utilization, may require learning rate adjustment

Recommendation: Start with batch size 64 and scale up until GPU memory is ~80% utilized.

### Learning Rate Tuning

- **Meta-learning**: Typically requires smaller outer learning rates (1e-3 to 1e-4)
- **Inner adaptation**: Can use larger rates (1e-2) for fast adaptation
- **With larger batches**: Consider scaling LR proportionally (e.g., batch size 2x → LR 2x)

### Using the --improve Flag

The `--improve` flag enables various optimizations:

```bash
# Enable attention mechanism
python main.py train --improve attention

# Enable batch normalization
python main.py train --improve bn

# Combine multiple improvements
python main.py train --improve attention bn logger

# Neural architecture search
python main.py train --improve nas
```

Available improvement tags:
- `attention` / `attn`: Inject attention mechanisms
- `nas` / `search`: Neural architecture search
- `bn` / `batch_norm`: Enable batch normalization
- `recursive` / `recursion`: Enable recursive policies
- `logger`: Attach performance logging
- `ckpt_select` / `checkpoint`: Select best checkpoint

### GPU Monitoring

```bash
# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Or check periodically during training
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
```

### Quick vs Full Experiments

**Quick validation** (for CI/testing):
```bash
python experiments/quick_benchmark.py
# or
python main.py train --steps 100 --batch_size 32
```

**Full training** (for research):
```bash
python main.py train --config basic --outer-steps 5000 --batch_size 128 --improve attention bn
```

## CLI and improvements

main.py adds:
- Pre-run checks: Python>=3.9, numpy present, torch optional (warns if missing)
- Safer import validation: clear hints if modules missing
- --improve TAG [TAG ...]: unknown tags warn and are ignored; valid tags are listed

Supported tags (if real_agi_continuous_improvement exposes the symbols):
- attention/attn -> inject_attention(policy)
- nas/search -> neural_architecture_search(policy, budget)
- bn/batch_norm -> enable_batch_norm(policy)
- recursive/recursion -> enable_recursive_policies(policy)
- logger -> attach_performance_logger(policy, stage) and/or log_training_metrics(step, metrics)
- ckpt_select/checkpoint -> select_best_checkpoint(ckpt_dir)

Behavior:
- The improvement module is optional; missing module just warns and continues
- Each call is hasattr-guarded; per-tag failures are warned and skipped
- A small default NAS budget is used unless cfg.nas_budget is provided

## Module docs and examples

### core/ssm.py

- Class SSM(config: SSMConfig)
  - init_parameters(seed:int)->Dict[str,np.ndarray]
  - set_parameters(params:Dict[str,np.ndarray])->None
  - get_parameters()->Dict[str,np.ndarray]
  - reset()->None
  - act(obs: np.ndarray)->np.ndarray
- Dataclass SSMConfig: hyperparams for SSM

Example usage:
```python
from core.ssm import SSM, SSMConfig
config = SSMConfig(state_dim=16, action_dim=4, obs_dim=10)
policy = SSM(config)
params = policy.init_parameters(seed=42)
policy.set_parameters(params)
policy.reset()
action = policy.act(obs=np.random.randn(10))
```

### meta_rl/meta_maml.py

- Class MetaLearner(config: MetaConfig)
  - outer_step(policy, tasks, inner_steps:int)->Dict[str,np.ndarray]
  - compute_meta_gradient(policy, tasks, inner_steps:int)->Dict[str,np.ndarray]
- Dataclass MetaConfig: meta_lr, inner_lr, etc.

Example:
```python
from meta_rl.meta_maml import MetaLearner, MetaConfig
learner = MetaLearner(MetaConfig(meta_lr=1e-3, inner_lr=1e-2))
tasks = [task1, task2, ...]
updated_params = learner.outer_step(policy, tasks, inner_steps=5)
```

### env_runner/environment.py

- Class EnvBatch(config: EnvConfig)
  - reset()->np.ndarray
  - step(actions: np.ndarray)->(obs, rew, done, info)
  - sample_tasks(n:int)->List[Task]
- Dataclass EnvConfig: env_name, num_envs, time_limit, seed

Example:
```python
from env_runner.environment import EnvBatch, EnvConfig
envs = EnvBatch(EnvConfig(env_name="MetaGymToy-v0", num_envs=8, time_limit=200, seed=0))
obs = envs.reset()
obs, rew, done, info = envs.step(actions=np.zeros((8,)))
```

### adaptation/test_time_adaptation.py

- Class TestTimeAdapter(config: AdaptConfig)
  - adapt(policy, envs, steps:int)->Dict[str,np.ndarray]
  - online_step(policy, obs, rew)->Dict[str,np.ndarray]
- Dataclass AdaptConfig: steps, lr, online(optional bool)

Example:
```python
from adaptation.test_time_adaptation import TestTimeAdapter, AdaptConfig
adapter = TestTimeAdapter(AdaptConfig(steps=5, lr=1e-2))
new_params = adapter.adapt(policy, envs, steps=5)
```

## Checkpoints

- Format: NumPy .npz created by numpy.savez. Keys are parameter names; values are ndarrays.
- Save: save_checkpoint(path, params)
- Load: load_checkpoint(path)->Dict[str,np.ndarray]
- Size: roughly the sum of array sizes; no pickling; portable; only ndarrays supported.
- Limitations: dtype and shape must match model expectations; no optimizer state stored.

## Extending the project

### Add a new environment (EnvBatch)

- Implement the EnvBatch interface and EnvConfig. Required methods: reset(), step(), sample_tasks().
- Ensure step() returns (obs, reward, done, info) with vectorized shapes for num_envs.
- Add your module under env_runner/ and adjust import if placed elsewhere.

### Add a new policy (SSM subclass)

- Subclass or implement a compatible interface with SSM:
  - init_parameters, set_parameters, get_parameters, reset, act
- If your architecture is reconfigurable, implement reconfigure(new_arch) for NAS integration.

### Add a new adaptation strategy

- Provide a class with the TestTimeAdapter interface: adapt(policy, envs, steps) and optionally online_step.
- Wire it in by replacing build_adapter() or adding a new CLI mode/flag.

## Security notes for dynamic improvement module

- Whitelist enforcement: only expected attributes are accessible (inject_attention, neural_architecture_search, enable_batch_norm, enable_recursive_policies, attach_performance_logger, select_best_checkpoint)
- Imported via importlib with errors soft-failing; unknown attributes are pruned from the accessible namespace
- All calls are guarded by hasattr and try/except; failures are logged and skipped
- No file/network/exec permissions are granted by this pipeline; external code must not perform side-effectful operations without explicit user review

## Tests (minimal smoke)

- core/test_ssm.py: imports SSM/SSMConfig, initializes params, runs reset/act
- meta_rl/test_meta_rl.py: imports MetaLearner/MetaConfig and runs a dummy outer_step with placeholder tasks
- adaptation/test_adaptation.py: imports TestTimeAdapter/AdaptConfig and calls adapt() with a stub policy/envs

Run tests:
```bash
python -m pytest -q  # if you add pytest
# or simply run each test file with python to check imports/execution.
```
