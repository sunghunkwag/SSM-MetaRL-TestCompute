# SSM-MetaRL-TestCompute

A minimal Python package for State Space Models (SSM) with Meta Reinforcement Learning and Test-Time Adaptation.

## Installation

### Install from GitHub

You can install the package directly from this GitHub repository:

```bash
pip install git+https://github.com/sunghunkwag/SSM-MetaRL-TestCompute.git
```

### Install for Development

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/sunghunkwag/SSM-MetaRL-TestCompute.git
cd SSM-MetaRL-TestCompute
pip install -e .
```

### Build and Install from Wheel

To build the package as a wheel:

```bash
pip install build
python -m build
pip install dist/ssm_metarl-0.1.0-py3-none-any.whl
```

## Usage

### Import the Package

```python
import core.ssm as ssm
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import TestTimeAdapter
from env_runner.environment import EnvironmentRunner

# Initialize State Space Model
config = ssm.SSMConfig(state_dim=10, obs_dim=5, action_dim=2)
model = ssm.StateSpaceModel(config)

# Initialize parameters
params = model.init_parameters()

# Reset and act
state = model.reset(params)
action = model.act(params, state, observation)
```

### Example: Meta-Learning with MAML

```python
from meta_rl.meta_maml import MetaMAML, MetaConfig

# Configure meta-learning
meta_config = MetaConfig(
    inner_lr=0.01,
    outer_lr=0.001,
    num_inner_steps=5
)

meta_learner = MetaMAML(meta_config)

# Train on multiple tasks
meta_learner.train(tasks)
```

### Example: Test-Time Adaptation

```python
from adaptation.test_time_adaptation import TestTimeAdapter, AdaptConfig

# Configure adaptation
adapt_config = AdaptConfig(
    adapt_lr=0.001,
    adapt_steps=10
)

adapter = TestTimeAdapter(adapt_config)

# Adapt policy at test time
adapted_policy = adapter.adapt(policy, test_envs, steps=10)
```

## Package Structure

- `core/` - State Space Model (SSM) implementation
  - `ssm.py` - Core SSM classes and configuration
- `meta_rl/` - Meta Reinforcement Learning algorithms
  - `meta_maml.py` - MAML implementation for meta-learning
- `adaptation/` - Test-time adaptation strategies
  - `test_time_adaptation.py` - Online adaptation algorithms
- `env_runner/` - Environment runner utilities
  - `environment.py` - Environment interface and batch processing

## Features

- **State Space Models**: Efficient SSM implementation for sequential decision making
- **Meta-Learning**: MAML-based meta-learning for fast adaptation
- **Test-Time Adaptation**: Online adaptation strategies for deployment
- **Modular Design**: Easy to extend and customize components

## Size and Limitations

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

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
