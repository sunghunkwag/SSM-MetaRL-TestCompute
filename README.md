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
  - `environment.py`: Gymnasium environment wrapper and utilities
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

  - `adapt_task(support_x: torch.Tensor, support_y: torch.Tensor, loss_fn=None, num_steps: int = 1)`: Adapts model to a specific task

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
    input_dim=64, # Note: main.py uses a separate input_dim argument
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

# Task adaptation data
task_x = torch.randn(16, 64)
task_y = torch.randn(16, 32)

# Adapt to task using torch.func.functional_call internally
adapted_params = maml.adapt_task(task_x, task_y, num_steps=5)

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

#### Implementation Overview

The core adaptation step in `adapt_task` uses:

```python
# Inside adapt_task() method - simplified conceptual view
params = OrderedDict((name, param.clone())
         for name, param in self.model.named_parameters())

# 'num_steps' is passed into the adapt_task method
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

### Test-Time Adaptation (Adapter)

The `Adapter` class in `adaptation/test_time_adaptation.py` handles online updates to the model during inference.

**Key Method**:

  - `update_step(loss_fn: Callable, batch: Mapping, fwd_fn: Optional[Callable] = None) -> Dict`:
      - Performs one or more gradient steps based on the provided batch and loss function.
      - Returns a `dict` containing log information (e.g., `{'loss': 0.123, 'steps': 1, 'updated': True}`).

**Example Usage**:

```python
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
import torch.nn as nn

# Assume 'model' is an initialized SSM instance
# Create config and adapter
config = AdaptationConfig(lr=0.01, max_steps_per_call=5)
adapter = Adapter(model, config)

# Define forward and loss wrappers (necessary if batch keys don't match forward args)
def fwd_fn(batch):
    return adapter.target(batch['x']) # Assuming model's forward uses 'x'

def loss_fn_wrapper(outputs, batch):
    return nn.MSELoss()(outputs, batch['targets'])

# Create data batch matching model's input_dim and output_dim
# Example: input_dim=64, output_dim=32
batch_dict = {'x': torch.randn(8, 64), 'targets': torch.randn(8, 32)}

# Run an adaptation step
info_dict = adapter.update_step(loss_fn_wrapper, batch_dict, fwd_fn=fwd_fn)
print(info_dict) # Example output: {'updated': True, 'steps': 5, 'loss': ...}
```

## Installation

This project uses `pyproject.toml` for packaging. The `setup.py` file is deprecated and can be ignored or removed.

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
