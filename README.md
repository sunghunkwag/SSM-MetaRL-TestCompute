# SSM-MetaRL-TestCompute
  
- `ssm.py`: State Space Model (nn.Module with Linear layers)
- **meta_rl/**: Meta-learning algorithms
  - `meta_maml.py`: MetaMAML implementation with inner/outer loop optimization
- **adaptation/**: Test-time adaptation
  - `test_time_adaptation.py`: Adapter class for online policy adaptation
- **env_runner/**: Environment utilities
  - `environment.py`: Gym environment wrapper and utilities

## API Reference

### SSM(state_dim, action_dim, hidden_dim)

**Parameters:**
- `state_dim` (int): Dimension of state/observation space
- `action_dim` (int): Dimension of action space
- `hidden_dim` (int): Hidden layer dimension (default: 64)

**Methods:**
- `forward(state)`: Returns action logits for given state tensor

### MetaMAML(model, inner_lr, outer_lr, first_order)

**Parameters:**
- `model` (nn.Module): The policy model to meta-train
- `inner_lr` (float): Task-level adaptation learning rate for inner loop
- `outer_lr` (float): Meta-level learning rate for outer loop optimizer
- `first_order` (bool): Whether to use first-order approximation (default: False)

**Methods:**
- `inner_loop(support_x, support_y)`: Performs inner loop adaptation on support set
  - `support_x` (Tensor): Support set observations
  - `support_y` (Tensor): Support set targets
  - Returns adapted_params (dict of tensors)
  
- `outer_loop(query_x, query_y, adapted_params)`: Computes meta-loss on query set
  - `query_x` (Tensor): Query set observations
  - `query_y` (Tensor): Query set targets
  - `adapted_params` (dict): Adapted parameters from inner_loop
  - Returns meta_loss (Tensor)

**Example:**
```python
from meta_rl.meta_maml import MetaMAML
from core.ssm import SSM

model = SSM(state_dim=10, action_dim=4, hidden_dim=64)
meta_maml = MetaMAML(
    model=model,
    inner_lr=0.01,
    outer_lr=0.001,
    first_order=False
)

# Training loop
for task_batch in task_loader:
    for task in task_batch:
        # Inner loop: adapt to support set
        adapted_params = meta_maml.inner_loop(
            support_x=task['support_x'],
            support_y=task['support_y']
        )
        
        # Outer loop: compute meta-loss on query set
        meta_loss = meta_maml.outer_loop(
            query_x=task['query_x'],
            query_y=task['query_y'],
            adapted_params=adapted_params
        )
        
        meta_loss.backward()
    meta_maml.outer_optimizer.step()
```

### Adapter(target, cfg, strategy)

**Parameters:**
- `target` (nn.Module): The model to adapt
- `cfg` (AdaptationConfig): Configuration for adaptation behavior
- `strategy` (str): Adaptation strategy - "none", "online", or "meta" (default: "none")

**AdaptationConfig fields:**
- `lr` (float): Learning rate for adaptation (default: 0.001)
- `max_steps_per_call` (int): Maximum adaptation steps per adapt() call (default: 1)

**Methods:**
- `adapt(batch, loss_fn)`: Adapts model to new data batch
  - `batch` (Mapping[str, Any]): Dictionary with 'observations' and 'targets' keys
  - `loss_fn` (Callable): Loss function that takes (predictions, targets)
  - Returns adaptation_info (dict with loss, steps, etc.)
  
- `observe(context)`: Records meta-features from context for adaptation
  - `context` (Mapping[str, Any]): Dictionary containing context information

**Example:**
```python
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from core.ssm import SSM
import torch
import torch.nn.functional as F

model = SSM(state_dim=10, action_dim=4, hidden_dim=64)

# Create adapter with configuration
adapter = Adapter(
    target=model,
    cfg=AdaptationConfig(
        lr=0.01,
        max_steps_per_call=10
    ),
    strategy="online"
)

# Adapt on new data - batch MUST be a dictionary
batch = {
    'observations': torch.randn(32, 10),
    'targets': torch.randn(32, 4)
}
info = adapter.adapt(
    batch=batch,
    loss_fn=F.mse_loss
)
print(f"Adapted with loss: {info['loss']}, steps: {info['steps']}")
```

## License

MIT License
