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
````

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
# [FIXED] Updated constructor to match meta_maml.py
MetaMAML(
    model: nn.Module,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    first_order: bool = False
)
```

Parameters:

  - `model`: Base neural network model (e.g., SSM)
  - `inner_lr`: Learning rate for task-specific adaptation
  - `outer_lr`: Learning rate for meta-optimization
  - `first_order`: If True, use first-order MAML (no second-order gradients)

**Methods**:

  - `adapt(support_x: torch.Tensor, support_y: torch.Tensor, loss_fn=None, num_steps: int = 1)`: Adapts model to a specific task

      - [FIXED] Updated signature to match meta\_maml.py
      - Returns adapted parameters as an `OrderedDict`
      - Uses `torch.func.functional_call` with `create_graph=not self.first_order` for gradient tracking

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
    first_order=False
)

# Task adaptation
task_x = torch.randn(16, 64)
task_y = torch.randn(16, 32)

# [FIXED] Correct adapt call with 'num_steps'
# Adapt to task using torch.func.functional_call internally
adapted_params = maml.adapt(task_x, task_y, num_steps=5)

# Use adapted model with torch.func.functional_call
test_x = torch.randn(8, 64)
test_output = torch.func.functional_call(
    base_model,
    adapted_params,
    test_x
)

# Meta-learning across tasks (assuming tasks are (sx, sy, qx, qy) tuples)
# tasks = [(task_x1, task_y1, query_x1, query_y1), ...]
# maml.meta_update(tasks)
```

#### Important Notes

  - **Only `torch.func.functional_call` is used**: There are no alternative implementations, fallback paths, or manual gradient computations
  - **Second-order gradients**: MAML requires computing gradients of gradients, which is fully supported by the `torch.func.functional_call` approach (`create_graph=not self.first_order`)
  - **Parameter dictionaries**: All adapted parameters are maintained as `OrderedDict` compatible with `torch.func.functional_call`
  - **PyTorch 2.0+**: Requires PyTorch 2.0 or later for proper `torch.func` support

#### Implementation Overview

The core adaptation step uses:

```python
# [FIXED] Updated overview to match meta_maml.py
# Inside adapt() method - simplified conceptual view
params = OrderedDict((name, param.clone())
         for name, param in self.model.named_parameters())

# 'num_steps' is passed into the adapt method
for step in range(num_steps):
    # Forward pass using torch.func.functional_call
    output = torch.func.functional_call(self.model, params, support_x)
    
    # Compute loss
    loss = loss_fn(output, support_y)
    
    # Compute gradients with create_graph based on 'first_order' flag
    grads = torch.autograd.grad(
        loss,
        params.values(),
        create_graph=not self.first_order
    )
    
    # Update parameters
    params = OrderedDict((name, param - self.inner_lr * grad)
             for (name, param), grad in zip(params.items(), grads))

return params
```

All forward passes go through `torch.func.functional_call`, enabling clean gradient flow for meta-learning.

## Installation

```bash
# Clone the repository
git clone [https://github.com/sunghunkwag/SSM-MetaRL-TestCompute.git](https://github.com/sunghunkwag/SSM-MetaRL-TestCompute.git)
cd SSM-MetaRL-TestCompute

# Install the package
pip install .

# To install in editable mode with development dependencies
pip install -e .[dev]
```

```
```
