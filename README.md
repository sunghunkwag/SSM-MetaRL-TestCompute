# SSM-MetaRL-TestCompute
A research framework combining State Space Models (SSM), Meta-Learning (MAML), and Test-Time Adaptation for reinforcement learning.

## Project Structure
- **core/**: Core model implementations
  - `ssm.py`: State Space Model implementation (now returns state)
- **meta_rl/**: Meta-learning algorithms
  - `meta_maml.py`: MetaMAML implementation (handles stateful models)
- **adaptation/**: Test-time adaptation
  - `test_time_adaptation.py`: Adapter class
- **env_runner/**: Environment utilities
  - `environment.py`: Gymnasium environment wrapper
- **experiments/**: Experiment scripts and benchmarks
  - `quick_benchmark.py`: Quick benchmark suite
- **tests/**: Test suite for all components (includes `test_ssm.py` now)

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
current_hidden = model.init_hidden(batch_size) #

# Forward pass requires current state and returns next state
output, next_hidden = model(input_x, current_hidden) #

print(output.shape)       # torch.Size([4, 32])
print(next_hidden.shape)  # torch.Size([4, 128])
````

### MetaMAML

The `MetaMAML` class in `meta_rl/meta_maml.py` implements MAML.

**Key Changes**:

  - Now correctly handles **stateful models** (like the modified SSM) where `forward` takes `hidden_state` and returns `(output, next_state)`.
  - `functional_forward` now accepts `hidden_state` and returns `(output, next_state)` if the model is stateful.
  - `adapt_task` requires an `initial_hidden_state` argument for stateful models and manages state propagation during the inner loop.
  - `meta_update` similarly requires `initial_hidden_state` for stateful models.

#### API Reference

**Constructor**:

```python
MetaMAML(model, inner_lr=0.01, outer_lr=0.001, first_order=False) #
```

**Methods**:

  - `adapt_task(support_x, support_y, initial_hidden_state=None, loss_fn=None, num_steps=1)`
  - `meta_update(tasks, initial_hidden_state=None, loss_fn=None)`
  - `functional_forward(x, hidden_state, params=None)`

#### Usage Example (Stateful Model like SSM)

```python
# Assume base_model is an initialized SSM instance
maml = MetaMAML(model=base_model, inner_lr=0.01)

# Task adaptation data (Batch, Time, Dim)
support_x_seq = torch.randn(16, 10, 64) # (B, T, D_in)
support_y_seq = torch.randn(16, 10, 32) # (B, T, D_out)
initial_hidden = base_model.init_hidden(batch_size=16) #

# Adapt to task, providing initial hidden state
adapted_params = maml.adapt_task( #
    support_x_seq,
    support_y_seq,
    initial_hidden_state=initial_hidden,
    num_steps=5
)

# Use adapted model over a sequence, managing state manually
test_x_seq = torch.randn(8, 10, 64)
current_hidden_test = base_model.init_hidden(batch_size=8)
outputs_seq = []
with torch.no_grad():
    for t in range(test_x_seq.shape[1]):
        step_input = test_x_seq[:, t, :]
        step_output, current_hidden_test = maml.functional_forward( #
            step_input,
            current_hidden_test,
            params=adapted_params
        )
        outputs_seq.append(step_output)
final_output = torch.stack(outputs_seq, dim=1) # Shape: (8, 10, 32)
```

### Test-Time Adaptation (Adapter)

The `Adapter` class in `adaptation/test_time_adaptation.py` handles online updates.

**Key Method**: `update_step(...)` returns a log dictionary.

**Note**: The current `Adapter.update_step` does not explicitly manage hidden states for stateful models within its loop. The example in `main.py` shows how to handle state outside the `update_step` call when interacting with an environment.

### Environment Runner

The `env_runner/environment.py` module now exclusively uses `gymnasium` and related wrappers. The old `gym` fallback has been removed.

## Main Script (`main.py`)

The `main.py` script now demonstrates an example workflow:

1.  Initializes a `gymnasium` environment using `env_runner`.
2.  Determines SSM `input_dim` and `output_dim` from the environment.
3.  Includes a basic `collect_data` function using the environment.
4.  Performs meta-training using `MetaMAML` with data collected from the environment.
5.  Performs test-time adaptation using `Adapter` while interacting with the environment.
6.  Accepts `--input_dim` and `--state_dim` as separate arguments.

## Installation & Setup

This project uses `pyproject.toml` for packaging. The `setup.py` file is deprecated and has been removed.

**Python Version**: Requires Python \>= 3.8. CI tests run on 3.8, 3.9, 3.10, 3.11.

```bash
git clone [https://github.com/sunghunkwag/SSM-MetaRL-TestCompute.git](https://github.com/sunghunkwag/SSM-MetaRL-TestCompute.git)
cd SSM-MetaRL-TestCompute
pip install .
# For development:
pip install -e .[dev]
```

## Running Tests

Ensure development dependencies are installed (`pip install -e .[dev]`).

```bash
pytest
```

The `core/test_ssm.py` file has been moved to `tests/test_ssm.py`.

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

```
```
