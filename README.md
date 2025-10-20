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

### MetaMAML(model, inner_lr, outer_lr, inner_steps, first_order)

**Parameters:**
- `model` (nn.Module): The policy model to meta-train
- `inner_lr` (float): Task-level adaptation learning rate for inner loop
- `outer_lr` (float): Meta-level learning rate for outer loop optimizer
- `inner_steps` (int): Number of gradient steps per task adaptation
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
    inner_steps=5,
    first_order=False
)

# Training loop
for task in tasks:
    support_x, support_y = task['support']
    query_x, query_y = task['query']
    
    # Inner loop: adapt to support set
    adapted_params = meta_maml.inner_loop(support_x, support_y)
    
    # Outer loop: compute meta-loss on query set
    meta_loss = meta_maml.outer_loop(query_x, query_y, adapted_params)
    meta_loss.backward()
    meta_maml.meta_optimizer.step()
```

### Adapter(target, cfg, strategy)

**Parameters:**
- `target` (nn.Module): The model to adapt at test time
- `cfg` (AdaptationConfig, optional): Configuration object with adaptation settings
- `strategy` (str): Adaptation strategy - "none", "online", or "meta" (default: "none")

**AdaptationConfig fields:**
- `lr` (float): Learning rate for adaptation (default: 0.001)
- `max_steps_per_call` (int): Maximum adaptation steps per adapt() call (default: 1)
- `loss_fn` (str): Loss function name - "mse" or "cross_entropy" (default: "mse")
- `optimizer` (str): Optimizer type - "sgd" or "adam" (default: "adam")

**Methods:**
- `adapt(loss_fn, batch, **kwargs)`: Adapts model to new data batch
  - `loss_fn` (Callable): Loss function that takes (predictions, targets)
  - `batch` (Mapping[str, Any]): Dictionary with 'observations' and 'targets' keys
  - `**kwargs`: Additional adaptation parameters
  - Returns adaptation_info (dict with loss, steps, etc.)
  
- `observe(context)`: Records meta-features from context for adaptation
  - `context` (Mapping[str, Any]): Dictionary containing context information

**Example:**
```python
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from core.ssm import SSM
import torch.nn.functional as F

model = SSM(state_dim=10, action_dim=4, hidden_dim=64)

# Create adapter with configuration
adapter = Adapter(
    target=model,
    cfg=AdaptationConfig(
        lr=0.01,
        max_steps_per_call=10,
        loss_fn="mse",
        optimizer="adam"
    ),
    strategy="online"
)

# Adapt on new data - batch MUST be a dictionary
batch = {
    'observations': torch.randn(32, 10),
    'targets': torch.randn(32, 4)
}

info = adapter.adapt(
    loss_fn=F.mse_loss,
    batch=batch
)

print(f"Adapted with loss: {info['loss']}, steps: {info['steps']}")
```

## License

MIT License
