# SSM-MetaRL-TestCompute
A research framework combining State Space Models (SSM), Meta-Learning (MAML), and Test-Time Adaptation for reinforcement learning.

## Project Structure

- **core/**: Core model implementations
  - `ssm.py`: State Space Model implementation (returns state)
- **meta_rl/**: Meta-learning algorithms
  - `meta_maml.py`: MetaMAML implementation (handles stateful models and time series input)
- **adaptation/**: Test-time adaptation
  - `test_time_adaptation.py`: Adapter class (API updated, manages hidden state updates internally)
- **env_runner/**: Environment utilities
  - `environment.py`: Gymnasium environment wrapper
- **experiments/**: Experiment scripts and benchmarks
  - `quick_benchmark.py`: Quick benchmark suite (updated MAML API calls)
- **tests/**: Test suite for all components (includes parameter mutation verification)

## Core Components

### State Space Model (SSM)

The SSM implementation in `core/ssm.py` models state transitions.

**API**:
- `forward(x, hidden_state)` returns a tuple: `(output, next_hidden_state)`.
- `init_hidden(batch_size)` provides the initial hidden state.

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
current_hidden = model.init_hidden(batch_size) #

# Forward pass requires current state and returns next state
output, next_hidden = model(input_x, current_hidden) #
print(output.shape)       # torch.Size([4, 32])
print(next_hidden.shape)  # torch.Size([4, 128])
````

### MetaMAML

The `MetaMAML` class in `meta_rl/meta_maml.py` implements MAML.

**Key Changes**:

  - Correctly handles **stateful models** (like SSM).
  - Supports **time series input** `(B, T, D)`.
  - **API Update**: `meta_update` now takes `tasks` (a list of tuples) and `initial_hidden_state` as arguments.

**Time Series Input Handling**:
Input data should be shaped `(batch_size, time_steps, features)`. MAML processes sequences internally.

Example with time series (Updated API):

```python
import torch
import torch.nn.functional as F
from meta_rl.meta_maml import MetaMAML
from core.ssm import SSM

model = SSM(state_dim=64, input_dim=32, output_dim=16, device='cpu')
maml = MetaMAML(model, inner_lr=0.01, outer_lr=0.001)

# Time series input: (batch=4, time_steps=10, features=32)
support_x = torch.randn(4, 10, 32)
support_y = torch.randn(4, 10, 16) # Example target
query_x = torch.randn(4, 10, 32)
query_y = torch.randn(4, 10, 16) # Example target

# Prepare tasks as a list of tuples
tasks = []
for i in range(4): # Batch size is 4
    tasks.append((support_x[i:i+1], support_y[i:i+1], query_x[i:i+1], query_y[i:i+1]))

# Initialize hidden state (assuming same init state for all tasks in batch)
initial_hidden = model.init_hidden(batch_size=4) #

# Correctly call meta_update with tasks list and initial state
loss = maml.meta_update(tasks=tasks, initial_hidden_state=initial_hidden, loss_fn=F.mse_loss) #
print(f"Meta Loss: {loss:.4f}")
```

Constructor Arguments:

  - `model`: The base model.
  - `inner_lr` (float): Inner loop learning rate.
  - `outer_lr` (float): Outer loop learning rate.
  - `first_order` (bool): Use first-order MAML.

### Adapter (Test-Time Adaptation)

The `Adapter` class in `adaptation/test_time_adaptation.py` performs test-time adaptation.

**Key Changes - API and Hidden State Management**:

  - **API Update**: `update_step` now takes `x`, `y` (target), and `hidden_state` directly as arguments. It no longer requires `loss_fn` or `fwd_fn`.
  - The `Adapter` internally performs `config.num_steps` of gradient updates per `update_step` call, correctly managing the hidden state across these internal steps.
  - Returns `(loss, steps_taken)`.

Constructor Arguments:

  - `model`: The model to adapt.
  - `config`: An `AdaptationConfig` object containing `learning_rate` and `num_steps` (internal steps per call).
  - `device`: Device string ('cpu' or 'cuda').

Example usage with hidden state (Updated API):

```python
import torch
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from core.ssm import SSM

# Model output dim must match target 'y'
model = SSM(state_dim=64, input_dim=32, output_dim=32, device='cpu') # Predict next state
config = AdaptationConfig(learning_rate=0.01, num_steps=5) # 5 internal steps per call
adapter = Adapter(model=model, config=config, device='cpu')

# Initialize hidden state
hidden_state = model.init_hidden(batch_size=1) # state_t

# Adaptation loop
for step in range(10): # Total adaptation calls
    x = torch.randn(1, 32)         # obs_t
    y_target = torch.randn(1, 32)  # target_t+1 (e.g., next_obs)
    
    # Store current state for adaptation call
    current_hidden_state_for_adapt = hidden_state # state_t
    
    # Get next state prediction (optional, for environment interaction)
    with torch.no_grad():
        output, hidden_state = model(x, current_hidden_state_for_adapt) # Update hidden_state to state_t+1
    
    # Call update_step with x, target, and state_t
    loss, steps_taken = adapter.update_step(
        x=x,
        y=y_target,
        hidden_state=current_hidden_state_for_adapt # Pass state_t
    )
    print(f"Adapt Call {step}, Loss: {loss:.4f}, Internal Steps: {steps_taken}")
    
    # hidden_state is now state_t+1 for the next loop iteration
```

### Environment Runner

Uses `gymnasium`.

## Main Script (`main.py`)

Demonstrates the complete workflow using the updated APIs.

  - Collects data and returns it as a dictionary of tensors.
  - Correctly calls `MetaMAML.meta_update` with `tasks` list and `initial_hidden_state`.
  - Correctly calls `Adapter.update_step` with `x`, `y` (target), and the correct `hidden_state` (state\_t).
  - Sets SSM `output_dim` to match the target dimension (e.g., `input_dim` if predicting next observation).

## Experiments

### Quick Benchmark (`experiments/quick_benchmark.py`)

Updated to use the correct API calls for `MetaMAML` and `Adapter`.

## Running Tests

Ensure development dependencies are installed (`pip install -e .[dev]`).

```bash
pytest
```

### Test Suite Highlights

  - `tests/test_adaptation.py`: Includes parameter mutation verification using `torch.equal`.
  - `tests/test_ssm.py`: Tests SSM API.
  - `tests/test_meta_rl.py`: Tests MetaMAML API.

## Installation & Setup

Uses `pyproject.toml`. Requires Python \>= 3.8.

```bash
git clone [https://github.com/sunghunkwag/SSM-MetaRL-TestCompute.git](https://github.com/sunghunkwag/SSM-MetaRL-TestCompute.git)
cd SSM-MetaRL-TestCompute
pip install .

# For development:
pip install -e .[dev]
```

## Docker Usage

Uses multi-stage build.

**Build:**

```bash
docker build -t ssm-metarl .
```

**Run:**

```bash
# Run main script
docker run ssm-metarl python main.py --env_name Pendulum-v1 --num_epochs 10

# Run benchmark
docker run ssm-metarl python experiments/quick_benchmark.py
```

## Summary of Key Changes (Latest Version)

1.  **SSM**: API unchanged (`forward` returns `(output, next_state)`).
2.  **MetaMAML**: API unchanged (`meta_update` takes `tasks` list and `initial_hidden_state`).
3.  **Adapter**: API changed (`update_step` takes `x`, `y`, `hidden_state`). Manages state updates internally across `num_steps`.
4.  **main.py**: Updated to use correct MetaMAML and Adapter API calls and logic.
5.  **experiments/quick\_benchmark.py**: Updated to use correct MetaMAML and Adapter API calls.
6.  **Tests**: Updated `test_adaptation.py` to match new Adapter API. Includes parameter mutation check.

<!-- end list -->

```
```
