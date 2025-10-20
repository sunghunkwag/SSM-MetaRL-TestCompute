# SSM-MetaRL-TestCompute
A research framework combining State Space Models (SSM), Meta-Learning (MAML), and Test-Time Adaptation for reinforcement learning.

## Project Structure
- **core/**: Core model implementations
  - `ssm.py`: State Space Model implementation
- **meta_rl/**: Meta-learning algorithms
  - `meta_maml.py`: MetaMAML implementation
- **adaptation/**: Test-time adaptation
  - `test_time_adaptation.py`: Adapter class with comprehensive configuration
- **env_runner/**: Environment utilities
  - `environment.py`: Gym environment wrapper and utilities
- **experiments/**: Experiment scripts and benchmarks
  - `quick_benchmark.py`: Quick benchmark suite
- **tests/**: Test suite for all components

## Core Components

### State Space Model (SSM)
The SSM implementation in `core/ssm.py` uses a single constructor argument `state_dim` and its `forward` returns a single tensor output (not a tuple). Any earlier examples that used an `input_dim` argument or showed a tuple `(output, hidden)` are outdated and have been removed to match the actual implementation.

Example usage (matches core/ssm.py):
```python
import torch
from core.ssm import SSM

state_dim = 128
model = SSM(state_dim=state_dim)

x = torch.randn(32, state_dim)  # batch_size x state_dim
out = model(x)                  # out is a single tensor
print(out.shape)                # expected: (32, state_dim)
```

## Installation
```bash
pip install torch numpy gymnasium
```

## Debug & Development Mode
### Enabling Debug Mode
For comprehensive error logging and debugging, set the `DEBUG` environment variable:
```bash
# Linux/macOS
export DEBUG=True
python main.py

# Windows (PowerShell)
$env:DEBUG="True"
python main.py

# Windows (CMD)
set DEBUG=True
python main.py
```

### Running Scripts with Debug Mode
```bash
# Main training script
DEBUG=True python main.py

# Quick benchmark
DEBUG=True python experiments/quick_benchmark.py

# Run tests with verbose error output
DEBUG=True python -m pytest -v --tb=long

# Run specific test with full traceback
DEBUG=True python -m pytest core/test_ssm.py -v --tb=long -s
```

### Development Guidelines
**For Contributors and Developers:**
1. **Always enable DEBUG mode when investigating errors:**
   - Full stack traces will be printed
   - All intermediate values logged
   - Error messages include context and state information
2. **Follow the git workflow:**
   - Create a feature branch for changes
   - Write unit tests for new features and bug fixes
   - Ensure all tests pass locally before opening a PR
   - Keep commits atomic with clear messages
   - Reference issues in commit messages when applicable
