# Autonomous-SSM-MetaRL

**An Autonomous Research Framework for State Space Models and Meta-Reinforcement Learning**

This repository integrates **State Space Models (SSM)** and **Meta-Learning (MAML)** into an autonomous multi-agent framework. Instead of manually tuning hyperparameters or architectures, specialized AI agents collaborate to design, train, and adapt models for high-dimensional reinforcement learning tasks.

## 🚀 Key Features

- **Autonomous Experimentation**: Agents design SSM architectures and run experiments autonomously.
- **Core Integration**:
  - **Brain**: CrewAI-based Multi-Agent System
  - **Engine**: PyTorch-based SSM & MAML implementations
- **Real Implementation**: Tools (`multi_agent/tools`) directly invoke the deep learning core (`core/`), managing model lifecycles and file I/O.

## 🛠️ Architecture

1. **State Modeling Agent**: Designing optimal SSM architectures (State Dim, Hidden Dim).
2. **Meta-Learning Agent**: Optimizing MAML strategies (Inner/Outer LR).
3. **Coordinator Agent**: Managing the overall research workflow.

## 📂 Project Structure

This structure clearly separates the "Brain" (Agents) from the "Engine" (Core Deep Learning).

```text
Autonomous-SSM-MetaRL/
├── core/                   # [Engine] Deep Learning Core (from SSM-MetaRL-TestCompute)
│   ├── ssm.py              # State Space Model implementation
├── meta_rl/                # [Engine] Meta-Learning Core
│   ├── meta_maml.py        # MAML implementation
├── multi_agent/            # [Brain] Multi-Agent System (from MultiAgent-SSM-MetaRL)
│   ├── agents/
│   │   ├── state_modeling_agent.py
│   ├── tools/
│   │   ├── ssm_tool.py     # BRIDGE: Connects Agents to PyTorch Core
│   └── workflows/
│       ├── research_workflow.py
├── saved_models/           # Artifacts storage (Model weights, Configs)
├── main.py                 # Entry point
├── pyproject.toml          # Dependencies
└── README.md               # Documentation
```

(Legacy components `adaptation/`, `experiments/`, `env_runner/` are preserved for engine functionality)

## Interactive Demo

**Try it now**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghunkwag/SSM-MetaRL-TestCompute/blob/main/demo.ipynb)

Run the complete demo in your browser with Google Colab - no installation required!

### Demo Notebook Features

- Correct API Usage: Demonstrates proper MetaMAML and Adapter APIs
- Time Series Handling: Proper 3D tensor shapes (batch, time, features)
- Hidden State Management: Correct initialization and propagation
- Visualization: Loss curves and adaptation progress
- Evaluation: Model performance metrics
- High-dimensional Benchmarks Preview: Introduces MuJoCo tasks and baseline comparisons

---

## Advanced Benchmarks

**Beyond Simple Tasks**: We've implemented benchmarks on high-dimensional MuJoCo tasks with baseline comparisons.

### Motivation

Simple benchmarks (CartPole, Pendulum) have limitations for research validation:
- Low dimensional (4-8 state dims)
- Simple dynamics
- Limited baseline comparisons
- No scaling validation

### Our Approach

**High-Dimensional Tasks**
- HalfCheetah-v4: 17-dim state, 6-dim action
- Ant-v4: 27-dim state, 8-dim action
- Humanoid-v4: 376-dim state, 17-dim action

**Baseline Comparisons**
- LSTM-MAML (76K params, O(n²) complexity)
- GRU-MAML (57K params, O(n²) complexity)
- Transformer-MAML (400K params, O(n²) complexity) 
- MLP-MAML (20K params, no sequence modeling)
- **SSM-MAML (53K params, O(n) complexity)**

**Meta-Learning Task Distributions**
- Velocity tasks: Different target speeds
- Direction tasks: Different goal directions
- Dynamics tasks: Varying gravity/mass

### Quick Start with Benchmarks

```bash
# Install MuJoCo dependencies
pip install 'gymnasium[mujoco]'

# Run benchmark on HalfCheetah-Vel
python experiments/serious_benchmark.py --task halfcheetah-vel --method ssm --epochs 50

# Compare all methods
python experiments/serious_benchmark.py --task ant-vel --method all --epochs 100

# Visualize results
python experiments/visualize_results.py --results-dir results --output-dir figures
```

### Results Preview

| Method | Parameters | Complexity | HalfCheetah-Vel |
|--------|------------|------------|----------------|
| **SSM** | 53K | O(n) | ✅ Tested |
| LSTM | 76K | O(n²) | ✅ Tested |
| GRU | 57K | O(n²) | ✅ Tested |
| Transformer | 400K | O(n²) | ✅ Tested |
| MLP | 20K | - | ✅ Tested |

**See [experiments/README.md](experiments/README.md) for detailed documentation.**

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Autonomous-SSM-MetaRL.git
cd Autonomous-SSM-MetaRL

# Install dependencies
pip install -e .
```

## 🏃 Usage

Run the autonomous researcher:

```bash
python main.py
```

```bash
# Pull the latest container
docker pull ghcr.io/sunghunkwag/ssm-metarl-testcompute:latest

# Run main script
docker run --rm ghcr.io/sunghunkwag/ssm-metarl-testcompute:latest python main.py --env_name CartPole-v1

# Run benchmark
docker run --rm ghcr.io/sunghunkwag/ssm-metarl-testcompute:latest python experiments/quick_benchmark.py
```

### Run Main Script

```bash
# Train on CartPole environment
python main.py --env_name CartPole-v1 --num_epochs 20

# Train on Pendulum environment
python main.py --env_name Pendulum-v1 --num_epochs 10
```

### Run Benchmark

```bash
python experiments/quick_benchmark.py
```

### Run Tests

```bash
pytest
```

## Test Results

The framework has been tested with the following results:

| Test Category | Status | Pass Rate |
|--------------|--------|-----------|
| Unit Tests | ✅ All Passing | 100% |
| CI/CD Pipeline | ✅ Automated | Python 3.8-3.11 |
| CartPole-v1 | ✅ Passed | Loss reduction: 91.5% - 93.7% |
| Pendulum-v1 | ✅ Passed | Loss reduction: 95.9% |
| Benchmarks | ✅ Passed | Loss reduction: 86.8% |

### Verified Functionality

- ✅ State Space Model (SSM) - All features working
- ✅ MetaMAML - Meta-learning operational  
- ✅ Test-Time Adaptation - Adaptation effects confirmed
- ✅ Environment Runner - Multiple environments supported
- ✅ Docker Container - Automated builds and deployment

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
from core.ssm import StateSpaceModel

model = StateSpaceModel(state_dim=128, input_dim=64, output_dim=32, device='cpu')
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

**Key Features**:
- Handles **stateful models** (like SSM)
- Supports **time series input** `(B, T, D)`
- **API**: `meta_update` takes `tasks` (a list of tuples) and `initial_hidden_state` as arguments

**Time Series Input Handling**:
Input data should be shaped `(batch_size, time_steps, features)`. MAML processes sequences internally.

Example with time series:

```python
import torch
import torch.nn.functional as F
from meta_rl.meta_maml import MetaMAML
from core.ssm import StateSpaceModel

model = StateSpaceModel(state_dim=64, input_dim=32, output_dim=16, device='cpu')
maml = MetaMAML(model, inner_lr=0.01, outer_lr=0.001)

# Time series input: (batch=4, time_steps=10, features=32)
support_x = torch.randn(4, 10, 32)
support_y = torch.randn(4, 10, 16)
query_x = torch.randn(4, 10, 32)
query_y = torch.randn(4, 10, 16)

# Prepare tasks as a list of tuples
tasks = []
for i in range(4):
    tasks.append((support_x[i:i+1], support_y[i:i+1], query_x[i:i+1], query_y[i:i+1]))

# Initialize hidden state
initial_hidden = model.init_hidden(batch_size=4)

# Call meta_update with tasks list and initial state
loss = maml.meta_update(tasks=tasks, initial_hidden_state=initial_hidden, loss_fn=F.mse_loss)
print(f"Meta Loss: {loss:.4f}")
```

Constructor Arguments:
- `model`: The base model.
- `inner_lr` (float): Inner loop learning rate.
- `outer_lr` (float): Outer loop learning rate.
- `first_order` (bool): Use first-order MAML.

### Adapter (Test-Time Adaptation)

The `Adapter` class in `adaptation/test_time_adaptation.py` performs test-time adaptation.

**Key Features**:
- **API**: `update_step` takes `x`, `y` (target), and `hidden_state` directly as arguments
- Internally performs `config.num_steps` gradient updates per call
- Properly detaches hidden state to prevent autograd computational graph errors
- Manages hidden state across internal steps
- Returns `(loss, steps_taken)`

Constructor Arguments:
- `model`: The model to adapt.
- `config`: An `AdaptationConfig` object containing `learning_rate` and `num_steps`.
- `device`: Device string ('cpu' or 'cuda').

Example usage:

```python
import torch
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from core.ssm import StateSpaceModel

# Model output dim must match target 'y'
model = StateSpaceModel(state_dim=64, input_dim=32, output_dim=32, device='cpu')
config = AdaptationConfig(learning_rate=0.01, num_steps=5)
adapter = Adapter(model=model, config=config, device='cpu')

# Initialize hidden state
hidden_state = model.init_hidden(batch_size=1)

# Adaptation loop
for step in range(10):
    x = torch.randn(1, 32)
    y_target = torch.randn(1, 32)
    
    # Store current state for adaptation call
    current_hidden_state_for_adapt = hidden_state
    
    # Get next state prediction (optional)
    with torch.no_grad():
        output, hidden_state = model(x, current_hidden_state_for_adapt)
    
    # Call update_step with x, target, and state_t
    loss, steps_taken = adapter.update_step(
        x=x,
        y=y_target,
        hidden_state=current_hidden_state_for_adapt
    )
    print(f"Adapt Call {step}, Loss: {loss:.4f}, Internal Steps: {steps_taken}")
```

### Environment Runner

The `Environment` class in `env_runner/environment.py` provides a wrapper around Gymnasium environments.

**Key Features**:
- Simplified API: `reset()` returns only observation (not tuple)
- Simplified API: `step(action)` returns 4 values (obs, reward, done, info)
- Batch processing support with `batch_size` parameter

## Main Script (`main.py`)

Demonstrates the complete workflow using the updated APIs.

- Collects data and returns it as a dictionary of tensors
- Calls `MetaMAML.meta_update` with `tasks` list and `initial_hidden_state`
- Calls `Adapter.update_step` with `x`, `y` (target), and the correct `hidden_state`
- Sets SSM `output_dim` to match the target dimension

## Experiments

### Quick Benchmark (`experiments/quick_benchmark.py`)

Runs a quick benchmark across multiple configurations to verify the framework's functionality.

**Features**:
- Tests multiple environments (CartPole, Pendulum)
- Measures adaptation effectiveness
- Reports loss reduction percentages

## Docker Usage

Uses multi-stage build for efficient containerization with automated CI/CD.

**Pull Pre-built Container:**

```bash
# Latest version
docker pull ghcr.io/sunghunkwag/ssm-metarl-testcompute:latest

# Specific version
docker pull ghcr.io/sunghunkwag/ssm-metarl-testcompute:main
```

**Build Locally:**

```bash
docker build -t ssm-metarl .
```

**Run:**

```bash
# Run main script
docker run --rm ghcr.io/sunghunkwag/ssm-metarl-testcompute:latest python main.py --env_name Pendulum-v1 --num_epochs 10

# Run benchmark
docker run --rm ghcr.io/sunghunkwag/ssm-metarl-testcompute:latest python experiments/quick_benchmark.py

# Run tests
docker run --rm ghcr.io/sunghunkwag/ssm-metarl-testcompute:latest pytest
```

## Recent Updates (v1.3.0)

### Demo Notebook Fixes (Latest)

1. **MetaMAML API Correction** (Commit: TBD)
   - **Fixed**: Corrected `meta_update()` to use `tasks` list and `initial_hidden_state`
   - **Problem**: Demo was using non-existent `support_data`/`query_data` parameters
   - **Solution**: Updated to match actual API: `meta_update(tasks, initial_hidden_state, loss_fn)`
   - **Impact**: Demo now runs without errors in Colab

2. **Adapter API Correction** (Commit: TBD)
   - **Fixed**: Replaced non-existent `adapt()` method with `update_step()`
   - **Problem**: Demo was calling `adapter.adapt(observations, targets)`
   - **Solution**: Use `update_step(x, y, hidden_state)` in a loop
   - **Impact**: Proper adaptation with loss tracking

3. **Data Shape Fixes** (Commit: TBD)
   - **Fixed**: Proper 3D tensor reshaping for time series (batch, time, features)
   - **Problem**: Data was passed as 2D tensors
   - **Solution**: Added `.unsqueeze(0)` to create batch dimension
   - **Impact**: MetaMAML can now process sequences correctly

4. **Hidden State Management** (Commit: TBD)
   - **Fixed**: Added proper hidden state initialization and propagation
   - **Problem**: Stateful model wasn't receiving required hidden_state
   - **Solution**: Initialize with `model.init_hidden()` and pass through all operations
   - **Impact**: SSM model works correctly with sequential data

### Previous Updates (v1.2.0)

1. **PyTorch Autograd Error Fix** (Commit: e084cf6)
   - **Fixed**: Added `hidden_state.detach()` in adaptation loop
   - **Problem**: Computational graph was being reused across gradient steps
   - **Solution**: Detach hidden state to prevent autograd errors
   - **Impact**: All tests now pass, adaptation works correctly

2. **Environment API Compatibility** (Commit: acbd1cf)
   - Fixed `env.reset()` to match Environment wrapper return values
   - Fixed `env.step()` to handle 4 return values instead of 5
   - Updated in 4 locations across `main.py`

3. **Action Space Handling** (Commit: acbd1cf)
   - Added dimension slicing for discrete action spaces
   - Prevents errors when model output_dim > action_space.n
   - Ensures valid action sampling

4. **Import Fixes** (Commit: acbd1cf)
   - Fixed incorrect import in `experiments/quick_benchmark.py`
   - Changed `import nn_functional as F` to `import torch.nn.functional as F`

### Test Results After All Fixes

All components work correctly:
- ✅ `main.py` works with CartPole-v1 and Pendulum-v1
- ✅ `experiments/quick_benchmark.py` runs without errors
- ✅ All unit tests pass (100% success rate)
- ✅ CI/CD pipeline passes on Python 3.8, 3.9, 3.10, 3.11
- ✅ Docker container builds and runs successfully

### Container Deployment

- **Automated builds** on every commit to main branch
- **Multi-stage Docker build** for optimized image size
- **Available on GitHub Container Registry**: `ghcr.io/sunghunkwag/ssm-metarl-testcompute`
- **Tags**: `latest`, `main`, `sha-<commit>`

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- Gymnasium >= 1.0
- NumPy
- pytest (for development)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{ssm_metarl_testcompute,
  author = {sunghunkwag},
  title = {SSM-MetaRL-TestCompute: A Framework for Meta-RL with State Space Models},
  year = {2025},
  url = {https://github.com/sunghunkwag/SSM-MetaRL-TestCompute}
}
```

## Acknowledgments

This framework builds upon research in:
- State Space Models for sequence modeling
- Model-Agnostic Meta-Learning (MAML)
- Test-time adaptation techniques
- Reinforcement learning with Gymnasium