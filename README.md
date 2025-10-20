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
The SSM implementation in `core/ssm.py` requires all of the following constructor arguments to be explicitly specified:
- `state_dim` (int): Internal state dimension
- `input_dim` (int): Input feature dimension
- `output_dim` (int): Output feature dimension
- `hidden_dim` (int): Hidden layer dimension
- `device` (str or torch.device): Device for computation (e.g., 'cpu', 'cuda')

The `forward` method accepts input of shape `(batch_size, input_dim)` and returns a single tensor output of shape `(batch_size, output_dim)` (not a tuple).

Example usage (matches core/ssm.py):
```python
import torch
from core.ssm import SSM

# Create SSM with all required parameters explicitly specified
model = SSM(
    state_dim=128,
    input_dim=64,
    output_dim=32,
    hidden_dim=128,
    device='cpu'
)

# Forward pass
x = torch.randn(32, 64)  # shape: (batch_size, input_dim)
out = model(x)           # out is a single tensor, shape: (batch_size, output_dim)
print(out.shape)         # expected: torch.Size([32, 32])
```

## Installation
