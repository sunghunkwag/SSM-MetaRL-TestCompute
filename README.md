# SSM-MetaRL-TestCompute

A research framework combining State Space Models (SSM), Meta-Learning (MAML), and Test-Time Adaptation for reinforcement learning.

## Project Structure

- **core/**: Core model implementations
  - `ssm.py`: State Space Model implementation
- **meta_rl/**: Meta-learning algorithms
  - `meta_maml.py`: MetaMAML implementation with adapt/meta_update methods
- **adaptation/**: Test-time adaptation
  - `test_time_adaptation.py`: Adapter class with comprehensive configuration
- **env_runner/**: Environment utilities
  - `environment.py`: Gym environment wrapper and utilities
- **experiments/**: Experiment scripts and benchmarks
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
SSM(state_dim: int, hidden_dim: int, output_dim: int, device: str = 'cpu')
```

**Parameters:**
- `state_dim` (int): Dimension of input state/observation space
- `hidden_dim` (int): Hidden layer dimension for internal representations
- `output_dim` (int): Dimension of output (e.g., action space)
- `device` (str): Device for computation - 'cpu' or 'cuda' (default: 'cpu')

**Methods:**
- `forward(x, hidden_state=None)`: Forward pass through the model
  - `x` (Tensor): Input tensor of shape (batch_size, state_dim)
  - `hidden_state` (Tensor, optional): Previous hidden state
  - Returns: Tuple of (output, new_hidden_state)

**Example:**
```python
from core.ssm import SSM
import torch

# Initialize SSM
model = SSM(state_dim=10, hidden_dim=64, output_dim=4, device='cpu')

# Forward pass
state = torch.randn(32, 10)  # batch_size=32, state_dim=10
output, hidden = model.forward(state)
print(f"Output shape: {output.shape}")  # [32, 4]
print(f"Hidden shape: {hidden.shape}")  # [32, 64]
```

---

### MetaMAML (Meta-Learning)

**Class:** `meta_rl.meta_maml.MetaMAML`

**Constructor:**
```python
MetaMAML(model: nn.Module, inner_lr: float = 0.01, outer_lr: float = 0.001, 
         first_order: bool = False, device: str = 'cpu')
```

**Parameters:**
- `model` (nn.Module): The policy model to meta-train (e.g., SSM instance)
- `inner_lr` (float): Learning rate for task-specific adaptation (inner loop) (default: 0.01)
- `outer_lr` (float): Learning rate for meta-optimizer (outer loop) (default: 0.001)
- `first_order` (bool): Whether to use first-order MAML approximation (default: False)
- `device` (str): Device for computation - 'cpu' or 'cuda' (default: 'cpu')

**Methods:**

- `adapt(task_batch, loss_fn, num_steps=1)`: Performs inner-loop adaptation on support set
  - `task_batch` (dict): Dictionary with 'observations' and 'actions'/'targets' keys
  - `loss_fn` (Callable): Loss function that takes (predictions, targets)
  - `num_steps` (int): Number of gradient steps for adaptation (default: 1)
  - Returns: `fast_weights` (OrderedDict) - adapted model parameters

- `meta_update(tasks, loss_fn)`: Performs outer-loop meta-update across multiple tasks
  - `tasks` (List[dict]): List of task dictionaries, each with 'support' and 'query' batches
  - `loss_fn` (Callable): Loss function for computing meta-loss
  - Returns: `meta_loss` (Tensor) - aggregated loss across all tasks

- `forward(x, params=None)`: Forward pass with optional custom parameters
  - `x` (Tensor): Input tensor
  - `params` (OrderedDict, optional): Custom parameters (e.g., from adapt())
  - Returns: Model output tensor

**Example:**
```python
from meta_rl.meta_maml import MetaMAML
from core.ssm import SSM
import torch
import torch.nn.functional as F

# Initialize model and meta-learner
model = SSM(state_dim=10, hidden_dim=64, output_dim=4)
meta_learner = MetaMAML(
    model=model,
    inner_lr=0.01,
    outer_lr=0.001,
    first_order=False
)

# Task adaptation (inner loop)
support_batch = {
    'observations': torch.randn(16, 10),
    'actions': torch.randn(16, 4)
}
fast_weights = meta_learner.adapt(
    task_batch=support_batch,
    loss_fn=F.mse_loss,
    num_steps=5
)

# Meta-update (outer loop) across multiple tasks
tasks = [
    {
        'support': {'observations': torch.randn(16, 10), 'actions': torch.randn(16, 4)},
        'query': {'observations': torch.randn(16, 10), 'actions': torch.randn(16, 4)}
    }
    for _ in range(8)  # 8 tasks
]
meta_loss = meta_learner.meta_update(tasks, F.mse_loss)
print(f"Meta-loss: {meta_loss.item():.4f}")
```

---

### Adapter (Test-Time Adaptation)

**Class:** `adaptation.test_time_adaptation.Adapter`

**Constructor:**
```python
Adapter(target: nn.Module, cfg: AdaptationConfig, strategy: str = 'none')
```

**Parameters:**
- `target` (nn.Module): The model to adapt during test time
- `cfg` (AdaptationConfig): Configuration object controlling adaptation behavior
- `strategy` (str): Adaptation strategy - "none", "online", or "meta" (default: "none")

**AdaptationConfig Fields:**

`AdaptationConfig` is a dataclass with the following fields:

- `lr` (float): Learning rate for test-time adaptation (default: 0.001)
- `max_steps_per_call` (int): Maximum number of gradient steps per adapt() call (default: 1)
- `grad_clip_norm` (float): Maximum gradient norm for clipping; None disables clipping (default: 1.0)
- `trust_region_eps` (float): Trust region constraint for parameter updates (default: 0.01)
- `ema_decay` (float): Exponential moving average decay for running statistics (default: 0.99)
- `entropy_weight` (float): Weight for entropy regularization in adaptation loss (default: 0.01)
- `min_samples` (int): Minimum samples required before adaptation begins (default: 10)
- `adaptation_interval` (int): Number of steps between adaptation updates (default: 1)
- `use_batch_norm` (bool): Whether to adapt batch normalization statistics (default: True)
- `reset_on_new_task` (bool): Whether to reset adapter state for new tasks (default: False)

**Methods:**

- `adapt(batch, loss_fn)`: Performs test-time adaptation on a data batch
  - `batch` (Mapping[str, Any]): Dictionary with 'observations' and 'targets' keys
  - `loss_fn` (Callable): Loss function that takes (predictions, targets)
  - Returns: `adaptation_info` (dict) with keys:
    - `'loss'`: Final adaptation loss (float)
    - `'steps'`: Number of gradient steps performed (int)
    - `'grad_norm'`: Gradient norm before clipping (float)
    - `'param_change'`: L2 norm of parameter change (float)

- `observe(context)`: Records context/meta-features for future adaptation
  - `context` (Mapping[str, Any]): Dictionary with context information (e.g., task metadata)
  - No return value; updates internal state

- `reset()`: Resets adapter to initial state (clears history, buffers)
  - No parameters or return value

**Example:**
```python
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from core.ssm import SSM
import torch
import torch.nn.functional as F

# Initialize model
model = SSM(state_dim=10, hidden_dim=64, output_dim=4)

# Create adapter with comprehensive configuration
config = AdaptationConfig(
    lr=0.01,
    max_steps_per_call=10,
    grad_clip_norm=1.0,
    trust_region_eps=0.01,
    ema_decay=0.99,
    entropy_weight=0.01,
    min_samples=10,
    adaptation_interval=1,
    use_batch_norm=True,
    reset_on_new_task=False
)

adapter = Adapter(
    target=model,
    cfg=config,
    strategy="online"
)

# Provide context for meta-learning strategies
context = {'task_id': 'task_001', 'domain': 'navigation'}
adapter.observe(context)

# Adapt on new test data - batch MUST be a dictionary
batch = {
    'observations': torch.randn(32, 10),
    'targets': torch.randn(32, 4)
}

info = adapter.adapt(
    batch=batch,
    loss_fn=F.mse_loss
)

print(f"Adaptation complete:")
print(f"  Loss: {info['loss']:.4f}")
print(f"  Steps: {info['steps']}")
print(f"  Grad norm: {info['grad_norm']:.4f}")
print(f"  Param change: {info['param_change']:.4f}")

# Reset for new task
adapter.reset()
```

---

## Complete Usage Example

Here's a complete example combining SSM, MetaMAML, and Adapter:

```python
import torch
import torch.nn.functional as F
from core.ssm import SSM
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter, AdaptationConfig

# 1. Create SSM model
model = SSM(state_dim=10, hidden_dim=64, output_dim=4, device='cpu')

# 2. Meta-train with MetaMAML
meta_learner = MetaMAML(
    model=model,
    inner_lr=0.01,
    outer_lr=0.001,
    first_order=False
)

# Generate meta-training tasks
meta_tasks = [
    {
        'support': {
            'observations': torch.randn(16, 10),
            'actions': torch.randn(16, 4)
        },
        'query': {
            'observations': torch.randn(16, 10),
            'actions': torch.randn(16, 4)
        }
    }
    for _ in range(10)
]

# Meta-training loop
for epoch in range(5):
    meta_loss = meta_learner.meta_update(meta_tasks, F.mse_loss)
    print(f"Epoch {epoch+1}, Meta-loss: {meta_loss.item():.4f}")

# 3. Test-time adaptation with Adapter
config = AdaptationConfig(
    lr=0.01,
    max_steps_per_call=10,
    grad_clip_norm=1.0,
    trust_region_eps=0.01
)

adapter = Adapter(target=model, cfg=config, strategy="online")

# Adapt to new test distribution
test_batch = {
    'observations': torch.randn(32, 10),
    'targets': torch.randn(32, 4)
}

info = adapter.adapt(batch=test_batch, loss_fn=F.mse_loss)
print(f"Test adaptation - Loss: {info['loss']:.4f}, Steps: {info['steps']}")
```

---

## Key Implementation Notes

### SSM (core/ssm.py)
- Constructor signature: `(state_dim, hidden_dim, output_dim, device)`
- Forward method: `forward(x, hidden_state=None)` returns `(output, new_hidden_state)`
- Device management: explicitly pass device parameter

### MetaMAML (meta_rl/meta_maml.py)
- `adapt()` returns `fast_weights` (OrderedDict), NOT loss
- `meta_update()` expects list of task dicts with 'support' and 'query' keys
- Use `forward(x, params=fast_weights)` for inference with adapted parameters

### Adapter (adaptation/test_time_adaptation.py)
- Batch input MUST be a dictionary with 'observations' and 'targets' keys
- AdaptationConfig has 10+ fields beyond just lr and max_steps_per_call
- Returns detailed adaptation_info dict with loss, steps, grad_norm, param_change
- Key stabilization features: gradient clipping, trust region, EMA, entropy regularization

---

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Key test files:
- `tests/test_ssm.py`: Tests SSM forward pass and hidden states
- `tests/test_meta_maml.py`: Tests adapt() and meta_update() methods
- `tests/test_adaptation.py`: Tests Adapter with all AdaptationConfig parameters

---

## License

MIT License
