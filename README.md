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
from core.ssm import SSM
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter
from env_runner.environment import Environment

# Initialize State Space Model
model = SSM()
params = model.init_parameters()

# Reset and act
state = model.reset(params)
action = model.act(params, state, observation)
```

### Meta-Learning with MetaMAML
```python
from meta_rl.meta_maml import MetaMAML

meta_learner = MetaMAML()
meta_learner.outer_step(tasks)
```

### Test-Time Adaptation
```python
from adaptation.test_time_adaptation import Adapter
from env_runner.environment import Environment

adapter = Adapter()
envs = Environment()
adapter.adapt(policy, envs, steps=100)
```

## Structure
The package is organized into four main components:

### core/
Core SSM implementation:
- `SSM`: Base State Space Model class with init_parameters, reset, and act methods

### meta_rl/
Meta-learning algorithms:
- `MetaMAML`: Meta-learning with Model-Agnostic Meta-Learning

### adaptation/
Test-time adaptation strategies:
- `Adapter`: Test-time adaptation for policies

### env_runner/
Environment management:
- `Environment`: Vectorized environment runner

## Extending

### Add a new environment
- Ensure step() returns (obs, reward, done, info) with vectorized shapes for num_envs.
- Add your module under env_runner/ and adjust import if placed elsewhere.

### Add a new policy (SSM subclass)
- Subclass or implement a compatible interface with SSM:
  - init_parameters, set_parameters, get_parameters, reset, act
- If your architecture is reconfigurable, implement reconfigure(new_arch) for NAS integration.

### Add a new adaptation strategy
- Provide a class with the Adapter interface: adapt(policy, envs, steps) and optionally online_step.
- Wire it in by replacing build_adapter() or adding a new CLI mode/flag.

## Security notes for dynamic improvement module
- Whitelist enforcement: only expected attributes are accessible (inject_attention, neural_architecture_search, enable_batch_norm, enable_recursive_policies, attach_performance_logger, select_best_checkpoint)
- Imported via importlib with errors soft-failing; unknown attributes are pruned from the accessible namespace
- All calls are guarded by hasattr and try/except; failures are logged and skipped
- No file/network/exec permissions are granted by this pipeline; external code must not perform side-effectful operations without explicit user review

## Tests (minimal smoke)
- core/test_ssm.py: imports SSM, initializes params, runs reset/act
- meta_rl/test_meta_rl.py: imports MetaMAML and runs a dummy outer_step with placeholder tasks
- adaptation/test_adaptation.py: imports Adapter and calls adapt() with a stub policy/envs

Run tests:
```bash
python -m pytest -q  # if you add pytest
# or simply run each test file with python to check imports/execution.
```

## License
MIT License

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
