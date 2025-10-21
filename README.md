# SSM-MetaRL-TestCompute
A research framework combining State Space Models (SSM), Meta-Learning (MAML), and Test-Time Adaptation for reinforcement learning.

## Project Structure

- **core/**: Core model implementations
  - `ssm.py`: State Space Model implementation (now returns state)
- **meta_rl/**: Meta-learning algorithms
  - `meta_maml.py`: MetaMAML implementation (handles stateful models and time series input)
- **adaptation/**: Test-time adaptation
  - `test_time_adaptation.py`: Adapter class (manages hidden state updates)
- **env_runner/**: Environment utilities
  - `environment.py`: Gymnasium environment wrapper
- **experiments/**: Experiment scripts and benchmarks
  - `quick_benchmark.py`: Quick benchmark suite
- **tests/**: Test suite for all components (includes parameter mutation verification)

## Core Components

### State Space Model (SSM)

The SSM implementation in `core/ssm.py` models state transitions.

**Key Changes**:
- `forward(x, hidden_state)` now returns a tuple: `(output, next_hidden_state)`. It no longer modifies internal state directly.
- A new method `init_hidden(batch_size)` is provided to get the initial hidden state.

Constructor Arguments:
- `state_dim` (int): Internal state dimension
- `input_dim` (int): Input feature dimension
- `output_dim` (int): Output feature dimension
- `hidden_dim` (int): Hidden layer dimension within networks
- `device` (str or torch.device)

Example usage:
```python
import torch
from core.ssm import SSM

model = SSM(state_dim=128, input_dim=64, output_dim=32, device='cpu')
batch_size = 4
input_x = torch.randn(batch_size, 64)
current_hidden = model.init_hidden(batch_size)

# Forward pass requires current state and returns next state
output, next_hidden = model(input_x, current_hidden)
print(output.shape)       # torch.Size([4, 32])
print(next_hidden.shape)  # torch.Size([4, 128])
```

### MetaMAML

The `MetaMAML` class in `meta_rl/meta_maml.py` implements MAML.

**Key Changes**:
- Now correctly handles **stateful models** (like the modified SSM) where `forward` takes `hidden_state` and returns `(output, next_state)`.
- Supports **time series input** in `(B, T, D)` format where B=batch, T=time steps, D=features.
- During inner loop adaptation, properly manages hidden state across time steps.

**Time Series Input Handling**:
When passing time series data to MetaMAML, the input should be shaped as `(batch_size, time_steps, features)`. The MAML algorithm will:
1. Initialize hidden state for each task
2. Process each time step sequentially
3. Update hidden state at each step
4. Compute loss over the entire sequence

Example with time series:
```python
import torch
from meta_rl.meta_maml import MetaMAML
from core.ssm import SSM

model = SSM(state_dim=64, input_dim=32, output_dim=16, device='cpu')
maml = MetaMAML(model, inner_lr=0.01, outer_lr=0.001)

# Time series input: (batch=4, time_steps=10, features=32)
support_x = torch.randn(4, 10, 32)
support_y = torch.randn(4, 10, 16)
query_x = torch.randn(4, 10, 32)
query_y = torch.randn(4, 10, 16)

# MetaMAML handles time series internally
tasks = [(support_x[i], support_y[i], query_x[i], query_y[i]) for i in range(4)]
loss = maml.meta_update(tasks, num_inner_steps=5)
```

Constructor Arguments:
- `model`: The base model to meta-train (e.g., SSM)
- `inner_lr` (float): Inner loop learning rate
- `outer_lr` (float): Outer loop (meta) learning rate
- `first_order` (bool): If True, use first-order MAML approximation

### Adapter (Test-Time Adaptation)

The `Adapter` class in `adaptation/test_time_adaptation.py` performs test-time adaptation.

**Key Changes - Hidden State Management**:
- The `update_step()` method now accepts a `hidden_state` parameter for stateful models.
- User code must manage hidden state updates by providing a `fwd_fn` that:
  1. Takes `(model, x, hidden_state)` as input
  2. Returns `(output, next_hidden_state)`
  3. Properly updates hidden state across adaptation steps

**Important**: The Adapter does not internally manage hidden states. You must:
1. Initialize hidden state before calling `update_step()`
2. Provide a `fwd_fn` that handles state updates
3. Pass the current `hidden_state` to each `update_step()` call
4. Update your local hidden state variable after each step

Constructor Arguments:
- `model`: The model to adapt
- `learning_rate` (float): Learning rate for adaptation
- `loss_fn`: Loss function (default: MSELoss)

Example usage with hidden state:
```python
import torch
from adaptation.test_time_adaptation import Adapter
from core.ssm import SSM

model = SSM(state_dim=64, input_dim=32, output_dim=16, device='cpu')
adapter = Adapter(model, learning_rate=0.01)

# Define forward function that manages hidden state
def fwd_fn(m, x, h):
    output, next_h = m(x, h)
    return output, next_h

# Initialize hidden state
hidden_state = model.init_hidden(batch_size=1)

# Adaptation loop with hidden state management
for step in range(10):
    x = torch.randn(1, 32)
    y_target = torch.randn(1, 16)
    
    # Pass current hidden state and get updated state
    loss, hidden_state = adapter.update_step(
        x, y_target, 
        fwd_fn=fwd_fn,
        hidden_state=hidden_state
    )
    print(f"Step {step}, Loss: {loss.item()}")
```

See `main.py` and `experiments/quick_benchmark.py` for complete examples of hidden state management in environment interaction contexts.

### Environment Runner

The `env_runner/environment.py` module now exclusively uses `gymnasium` and related wrappers. The old `gym` fallback has been removed.

## Main Script (`main.py`)

The `main.py` script demonstrates the complete workflow:

1. Initializes a `gymnasium` environment using `env_runner`.
2. Determines SSM `input_dim` and `output_dim` from the environment.
3. Includes a `collect_data` function that:
   - Manages hidden state during data collection
   - Resets hidden state at episode boundaries
   - Collects (observation, action, reward) tuples
4. Performs meta-training using `MetaMAML` with time series data (B, T, D format).
5. Performs test-time adaptation using `Adapter` with proper hidden state management:
   - Initializes hidden state before adaptation
   - Provides `fwd_fn` for state updates
   - Updates hidden state after each step
6. Accepts `--input_dim` and `--state_dim` as separate arguments.

**Hidden State Management in main.py**:
```python
# During data collection
hidden_state = model.init_hidden(batch_size=1)
for step in range(max_steps):
    obs, _ = env.reset() if done else (obs, None)
    if done:
        hidden_state = model.init_hidden(batch_size=1)
    
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    action_pred, hidden_state = model(obs_tensor, hidden_state)
    # ... rest of episode logic

# During test-time adaptation
def fwd_fn(m, x, h):
    return m(x, h)

hidden_state = model.init_hidden(batch_size=1)
for step in range(adaptation_steps):
    loss, hidden_state = adapter.update_step(
        x, y, fwd_fn=fwd_fn, hidden_state=hidden_state
    )
```

## Experiments

### Quick Benchmark (`experiments/quick_benchmark.py`)

The quick benchmark demonstrates:
1. **Time series input to MAML** without flattening (preserves B, T, D shape)
2. **Hidden state management** in Adapter with `fwd_fn`
3. Comparison between meta-trained and baseline models

Key implementation details:
```python
# Time series input (B=num_tasks, T=seq_length, D=input_dim)
support_x = torch.randn(num_tasks, seq_length, input_dim)
support_y = torch.randn(num_tasks, seq_length, output_dim)

# Pass to MAML without reshaping
tasks = [(support_x[i], support_y[i], query_x[i], query_y[i]) 
         for i in range(num_tasks)]
loss = meta_learner.meta_update(tasks, num_inner_steps=5)

# Adapter with hidden state
def fwd_fn(m, x, h):
    return m(x, h)

hidden = model.init_hidden(batch_size=1)
for i in range(test_steps):
    loss, hidden = adapter.update_step(
        test_x[i:i+1], test_y[i:i+1],
        fwd_fn=fwd_fn, hidden_state=hidden
    )
```

## Running Tests

Ensure development dependencies are installed (`pip install -e .[dev]`).

```bash
pytest
```

### Test Suite Highlights

**Parameter Mutation Verification** (`tests/test_adaptation.py`):
- Tests verify that `Adapter.update_step()` actually mutates model parameters
- Uses `copy.deepcopy` to save initial parameters
- Compares parameters after adaptation using `torch.equal`
- Ensures gradient-based updates are working correctly

Example from tests:
```python
import copy
import torch

# Save initial parameters
initial_params = {name: copy.deepcopy(param) 
                  for name, param in model.named_parameters()}

# Perform adaptation
for step in range(num_steps):
    loss, hidden = adapter.update_step(
        x, y, fwd_fn=fwd_fn, hidden_state=hidden
    )

# Verify parameters changed
for name, param in model.named_parameters():
    assert not torch.equal(param, initial_params[name]), \
        f"Parameter {name} was not updated"
```

**Other Test Files**:
- `tests/test_ssm.py`: Tests SSM state transitions and output shapes
- `tests/test_meta_rl.py`: Tests MetaMAML with time series input

The `core/test_ssm.py` file has been moved to `tests/test_ssm.py`.

## Installation & Setup

This project uses `pyproject.toml` for packaging. The `setup.py` file is deprecated and has been removed.

**Python Version**: Requires Python >= 3.8. CI tests run on 3.8, 3.9, 3.10, 3.11.

```bash
git clone https://github.com/sunghunkwag/SSM-MetaRL-TestCompute.git
cd SSM-MetaRL-TestCompute
pip install .

# For development:
pip install -e .[dev]
```

## Docker Usage

The `Dockerfile` now uses a multi-stage build for efficiency and no longer sets a default `ENTRYPOINT`.

**Build the image:**
```bash
docker build -t ssm-metarl .
```

**Run experiments:**
```bash
# Run the main example script with a specific environment
docker run ssm-metarl python main.py --env_name Pendulum-v1 --num_epochs 10

# Run the benchmark script
docker run ssm-metarl python experiments/quick_benchmark.py
```

## Summary of Key Changes

1. **SSM**: Returns `(output, next_hidden_state)` instead of modifying internal state
2. **MetaMAML**: Handles time series input `(B, T, D)` and stateful models correctly
3. **Adapter**: Requires user to manage hidden state via `fwd_fn` and `hidden_state` parameter
4. **Tests**: Include parameter mutation verification to ensure updates work correctly
5. **main.py & quick_benchmark.py**: Show complete examples of hidden state management

All user code (`main.py`, `experiments/quick_benchmark.py`, `tests/test_adaptation.py`) has been updated to match these patterns.
