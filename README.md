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

### MetaMAML

The MetaMAML implementation in `meta_rl/meta_maml.py` provides meta-learning capabilities using Model-Agnostic Meta-Learning (MAML).

#### Implementation Details

**MetaMAML uses ONLY `torch.func.functional_call` for all gradient computations**, including second-order derivatives. This ensures:
- Clean, consistent API across all operations
- Proper support for higher-order gradients required by MAML
- No fallback mechanisms or alternative code paths

#### Key Features

- **Pure `torch.func.functional_call` approach**: All forward passes use `torch.func.functional_call` with explicit parameter dictionaries
- **Second-order gradient support**: Enables true MAML updates with gradients through the adaptation process
- **Task adaptation**: Fast adaptation to new tasks with few gradient steps
- **Meta-optimization**: Updates base parameters to improve adaptation performance

#### API Reference

**Constructor**:
```python
MetaMAML(
    model: nn.Module,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    num_inner_steps: int = 1
)
```

Parameters:
- `model`: Base neural network model (e.g., SSM)
- `inner_lr`: Learning rate for task-specific adaptation
- `outer_lr`: Learning rate for meta-optimization
- `num_inner_steps`: Number of gradient steps during adaptation

**Methods**:

- `adapt(task_data, task_labels)`: Adapts model to a specific task
  - Returns adapted parameters as a dictionary
  - Uses `torch.func.functional_call` with `create_graph=True` for gradient tracking

- `meta_update(tasks)`: Performs meta-learning update across multiple tasks
  - Computes meta-gradient through adaptation process
  - Updates base model parameters

#### Usage Example

```python
import torch
import torch.nn as nn
from meta_rl.meta_maml import MetaMAML
from core.ssm import SSM

# Create base model
base_model = SSM(
    state_dim=128,
    input_dim=64,
    output_dim=32,
    hidden_dim=128,
    device='cpu'
)

# Initialize MetaMAML
maml = MetaMAML(
    model=base_model,
    inner_lr=0.01,
    outer_lr=0.001,
    num_inner_steps=5
)

# Task adaptation
task_x = torch.randn(16, 64)
task_y = torch.randn(16, 32)

# Adapt to task using torch.func.functional_call internally
adapted_params = maml.adapt(task_x, task_y)

# Use adapted model with torch.func.functional_call
test_x = torch.randn(8, 64)
test_output = torch.func.functional_call(
    base_model,
    adapted_params,
    test_x
)

# Meta-learning across tasks
tasks = [(task_x1, task_y1), (task_x2, task_y2), ...]
maml.meta_update(tasks)
```

#### Important Notes

- **Only `torch.func.functional_call` is used**: There are no alternative implementations, fallback paths, or manual gradient computations
- **Second-order gradients**: MAML requires computing gradients of gradients, which is fully supported by the `torch.func.functional_call` approach with `create_graph=True`
- **Parameter dictionaries**: All adapted parameters are maintained as dictionaries compatible with `torch.func.functional_call`
- **PyTorch 2.0+**: Requires PyTorch 2.0 or later for proper `torch.func` support

#### Implementation Overview

The core adaptation step uses:
```python
# Inside adapt() method - simplified conceptual view
params = {name: param.clone() for name, param in self.model.named_parameters()}

for step in range(self.num_inner_steps):
    # Forward pass using torch.func.functional_call
    output = torch.func.functional_call(self.model, params, task_data)
    
    # Compute loss
    loss = loss_fn(output, task_labels)
    
    # Compute gradients with create_graph=True for second-order derivatives
    grads = torch.autograd.grad(
        loss,
        params.values(),
        create_graph=True
    )
    
    # Update parameters
    params = {name: param - self.inner_lr * grad 
              for (name, param), grad in zip(params.items(), grads)}

return params
```

All forward passes go through `torch.func.functional_call`, enabling clean gradient flow for meta-learning.

## Installation
