# SSM-MetaRL-TestCompute
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
- tests: minimal smoke/unit tests (added below)

## Quickstart
- Install: pip install numpy [torch]
- Train: python main.py train --config basic --outer-steps 100 --improve attention bn
- Eval: python main.py eval --config basic --checkpoint checkpoints/latest.npz --episodes 10
- Adapted eval: python main.py adapt --config basic --checkpoint checkpoints/latest.npz --episodes 10 --adapt-steps 5

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
from core.ssm import SSM, SSMConfig
cfg = SSMConfig()
policy = SSM(cfg)
params = policy.init_parameters(seed=0)
policy.set_parameters(params)
action = policy.act(obs=np.zeros((1,))))

### meta_rl/meta_maml.py
- Class MetaLearner(config: MetaConfig)
  - outer_step(policy_class, init_params, tasks, ssm_cfg)->Tuple[Dict[str,np.ndarray], Dict[str,float]]
- Dataclass MetaConfig: outer_steps, tasks_per_batch, inner_steps, inner_lr

Example:
from meta_rl.meta_maml import MetaLearner, MetaConfig
from core.ssm import SSM, SSMConfig
meta = MetaLearner(MetaConfig())
params0 = SSM(SSMConfig()).init_parameters(seed=0)
params1, metrics = meta.outer_step(SSM, params0, tasks=[...], ssm_cfg=SSMConfig())

### env_runner/environment.py
- Class EnvBatch(config: EnvConfig)
  - sample_tasks(n:int)->List[Any]
  - reset()->np.ndarray
  - step(actions: np.ndarray)->Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]
- Dataclass EnvConfig: env_name, num_envs, time_limit, seed

Example:
from env_runner.environment import EnvBatch, EnvConfig
envs = EnvBatch(EnvConfig(env_name="MetaGymToy-v0", num_envs=8, time_limit=200, seed=0))
obs = envs.reset()
obs, rew, done, info = envs.step(actions=np.zeros((8,)))

### adaptation/test_time_adaptation.py
- Class TestTimeAdapter(config: AdaptConfig)
  - adapt(policy, envs, steps:int)->Dict[str,np.ndarray]
  - online_step(policy, obs, rew)->Dict[str,np.ndarray]
- Dataclass AdaptConfig: steps, lr, online(optional bool)

Example:
from adaptation.test_time_adaptation import TestTimeAdapter, AdaptConfig
adapter = TestTimeAdapter(AdaptConfig(steps=5, lr=1e-2))
new_params = adapter.adapt(policy, envs, steps=5)

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
python -m pytest -q  # if you add pytest
or simply run each test file with python to check imports/execution.
