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

## Installation

```bash
pip install torch numpy gymnasium
```

## API Reference

### SSM (State Space Model)

**Class:** `core.ssm.SSM`

**Constructor:**
```python
SSM(state_dim: int, hidden_dim: int = 128, output_dim: int = 32, device: str = 'cpu')
```

**Parameters:**
- `state_dim` (int): Dimension of input state/observation space
- `hidden_dim` (int, default=128): Hidden layer dimension for internal representations
- `output_dim` (int, default=32): Dimension of output (e.g., action space)
- `device` (str, default='cpu'): Device for computation - 'cpu' or 'cuda'

**Methods:**
- `forward(x, hidden_state=None)`: Forward pass through the model
  - `x` (Tensor): Input tensor of shape (batch_size, state_dim)
  - `hidden_state` (Tensor, optional): Previous hidden state
  - Returns: Single tensor output (output_dim)

**Example:**
```python
from core.ssm import SSM
import torch

# Initialize SSM with exact default values
model = SSM(state_dim=10, hidden_dim=128, output_dim=32, device='cpu')

# Forward pass
state = torch.randn(32, 10)  # batch_size=32, state_dim=10
output = model.forward(state)  # Returns single tensor
assert output.shape == (32, 32)  # (batch_size, output_dim)

# With hidden state
output = model.forward(state, hidden_state=None)
```

---

### MetaMAML

**Class:** `meta_rl.meta_maml.MetaMAML`

**Constructor:**
```python
MetaMAML(model: nn.Module, inner_lr: float, outer_lr: float, first_order: bool = False)
```

**Parameters:**
- `model` (nn.Module): Base model to meta-train (e.g., SSM)
- `inner_lr` (float): Learning rate for inner loop adaptation
- `outer_lr` (float): Learning rate for outer loop meta-update
- `first_order` (bool, default=False): Whether to use first-order approximation (faster but less accurate)

**Methods:**

1. `adapt(support_x, support_y, loss_fn, num_steps)` -> adapted_model
   - Adapts model to a specific task using support data
   - **Parameters:**
     - `support_x` (Tensor): Support set inputs
     - `support_y` (Tensor): Support set targets
     - `loss_fn` (callable): Loss function
     - `num_steps` (int): Number of adaptation steps
   - **Returns:** Adapted model (copy with updated parameters)

2. `meta_update(tasks: List[Tuple[support_x, support_y, query_x, query_y]], loss_fn)` -> meta_loss
   - Performs meta-update across multiple tasks
   - **Parameters:**
     - `tasks` (List[Tuple]): List of task tuples, each containing:
       - `support_x` (Tensor): Support inputs
       - `support_y` (Tensor): Support targets
       - `query_x` (Tensor): Query inputs
       - `query_y` (Tensor): Query targets
     - `loss_fn` (callable): Loss function
   - **Returns:** Meta-loss (float)

**Example:**
```python
from meta_rl.meta_maml import MetaMAML
from core.ssm import SSM
import torch
import torch.nn as nn

# Create base model
model = SSM(state_dim=4, hidden_dim=128, output_dim=1, device='cpu')

# Create MetaMAML with exact signature
meta_learner = MetaMAML(model, inner_lr=0.01, outer_lr=0.001, first_order=False)

# Prepare task data
support_x = torch.randn(10, 4)
support_y = torch.randn(10, 1)
query_x = torch.randn(10, 4)
query_y = torch.randn(10, 1)

loss_fn = nn.MSELoss()

# Adapt to single task
adapted_model = meta_learner.adapt(support_x, support_y, loss_fn, num_steps=5)

# Meta-update across multiple tasks
tasks = [(support_x, support_y, query_x, query_y) for _ in range(8)]
meta_loss = meta_learner.meta_update(tasks, loss_fn)
```

---

### Test-Time Adaptation

**Class:** `adaptation.test_time_adaptation.Adapter`

**Config Class:** `adaptation.test_time_adaptation.AdaptationConfig`

**AdaptationConfig Fields:**
```python
AdaptationConfig(
    lr: float,                    # Learning rate for adaptation
    grad_clip_norm: float,        # Gradient clipping norm
    trust_region_eps: float,      # Trust region epsilon for parameter updates
    ema_decay: float,             # EMA decay rate for statistics
    entropy_weight: float,        # Weight for entropy regularization
    max_steps_per_call: int       # Maximum adaptation steps per call
)
```

**Adapter Constructor:**
```python
Adapter(model: nn.Module, cfg: AdaptationConfig)
```

**Parameters:**
- `model` (nn.Module): Model to adapt (e.g., SSM)
- `cfg` (AdaptationConfig): Adaptation configuration

**Methods:**

- `adapt(loss_fn, batch_dict)` -> loss
  - Performs test-time adaptation on a batch
  - **Parameters:**
    - `loss_fn` (callable): Loss function
    - `batch_dict` (dict): Dictionary containing batch data (e.g., {'states': ..., 'targets': ...})
  - **Returns:** Adaptation loss (float)

**Example:**
```python
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from core.ssm import SSM
import torch
import torch.nn as nn

# Create model
model = SSM(state_dim=4, hidden_dim=128, output_dim=1, device='cpu')

# Create config with exact fields from implementation
config = AdaptationConfig(
    lr=0.01,
    grad_clip_norm=1.0,
    trust_region_eps=0.01,
    ema_decay=0.99,
    entropy_weight=0.01,
    max_steps_per_call=5
)

# Create Adapter with exact signature
adapter = Adapter(model, config)

# Prepare batch
states = torch.randn(16, 4)
targets = torch.randn(16, 1)
batch_dict = {'states': states, 'targets': targets}

loss_fn = nn.MSELoss()

# Perform adaptation
loss = adapter.adapt(loss_fn, batch_dict)
print(f"Adaptation loss: {loss:.4f}")
```

---

## Usage Example (main.py)

```python
from core.ssm import SSM
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
import torch
import torch.nn as nn

# 1. Create SSM with exact defaults
model = SSM(state_dim=4, hidden_dim=128, output_dim=1, device='cpu')

# 2. Meta-training with MetaMAML
meta_learner = MetaMAML(model, inner_lr=0.01, outer_lr=0.001, first_order=False)

# Prepare tasks: List[Tuple[support_x, support_y, query_x, query_y]]
tasks = []
for _ in range(8):
    support_x = torch.randn(10, 4)
    support_y = torch.randn(10, 1)
    query_x = torch.randn(10, 4)
    query_y = torch.randn(10, 1)
    tasks.append((support_x, support_y, query_x, query_y))

loss_fn = nn.MSELoss()
meta_loss = meta_learner.meta_update(tasks, loss_fn)

# 3. Test-time adaptation with Adapter
config = AdaptationConfig(
    lr=0.01,
    grad_clip_norm=1.0,
    trust_region_eps=0.01,
    ema_decay=0.99,
    entropy_weight=0.01,
    max_steps_per_call=5
)

adapter = Adapter(model, config)

batch_dict = {
    'states': torch.randn(16, 4),
    'targets': torch.randn(16, 1)
}

adapt_loss = adapter.adapt(loss_fn, batch_dict)
```

## Running Tests

```bash
# Test SSM
python -m pytest core/test_ssm.py -v

# Test MetaMAML
python -m pytest meta_rl/test_meta_rl.py -v

# Test Adaptation
python -m pytest adaptation/test_adaptation.py -v

# Run all tests
python -m pytest -v
```

## Quick Benchmark

```bash
python experiments/quick_benchmark.py
```

## Key API Guarantees

1. **SSM**: Always returns single tensor from forward(), never tuple
2. **MetaMAML**: 
   - `adapt()` returns adapted model
   - `meta_update()` expects List[Tuple[4 tensors]] format
3. **Adapter**: 
   - Uses exact AdaptationConfig fields (no extras)
   - `adapt()` signature is `(loss_fn, batch_dict)`

All examples, tests, and documentation are synchronized to these exact APIs.
