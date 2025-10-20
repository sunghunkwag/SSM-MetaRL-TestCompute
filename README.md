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

The SSM implementation in `core/ssm.py` uses constructor arguments `state_dim`, `input_dim`, and `output_dim`. The `forward` method returns a single tensor output (not a tuple).

Example usage (matches core/ssm.py):

```python
import torch
from core.ssm import SSM

state_dim = 128
input_dim = 128
output_dim = 64

# Create SSM with explicit input_dim and output_dim
model = SSM(state_dim=state_dim, input_dim=input_dim, output_dim=output_dim)

# Forward pass
x = torch.randn(32, input_dim)  # batch_size x input_dim
out = model(x)                   # out is a single tensor
print(out.shape)                 # expected: (32, output_dim)
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

### Debug Features

When debug mode is enabled, you'll see:
- Detailed error stack traces
- Component initialization logs
- Configuration validation messages
- Warning for missing dependencies

## Usage

### Quick Test

```bash
python main.py
```

### Quick Benchmark

```bash
python experiments/quick_benchmark.py
```

### Run Tests

```bash
pytest tests/ -v
```

## Features

- **State Space Models (SSM)**: PyTorch implementation with torch.nn.Module integration
- **Meta-Learning (MAML)**: Model-Agnostic Meta-Learning with PyTorch 2.0+ support
- **Test-Time Adaptation**: Configurable adaptation strategies
- **Environment Integration**: Gymnasium-compatible environment wrappers
- **Comprehensive Testing**: Unit tests for all components

## License

MIT License - See LICENSE file for details
