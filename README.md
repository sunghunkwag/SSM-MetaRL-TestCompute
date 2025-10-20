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

## Debug & Development Mode
### Enabling Debug Mode
For comprehensive error logging and debugging, set the `DEBUG` environment variable:

```bash
# Linux/macOS
export DEBUG=True
python main.py

# Windows (PowerShell)
$env:DEBUG="True"
python main.py

# Windows (CMD)
set DEBUG=True
python main.py
```

### Running Scripts with Debug Mode
```bash
# Main training script
DEBUG=True python main.py

# Quick benchmark
DEBUG=True python experiments/quick_benchmark.py

# Run tests with verbose error output
DEBUG=True python -m pytest -v --tb=long

# Run specific test with full traceback
DEBUG=True python -m pytest core/test_ssm.py -v --tb=long -s
```

### Development Guidelines
**For Contributors and Developers:**
1. **Always enable DEBUG mode when investigating errors:**
   - Full stack traces will be printed
   - All intermediate values logged
   - Error messages include context and state information

2. **Follow the git workflow:**
   - Create feature branches
   - Write meaningful commit messages
   - Test thoroughly before pushing

3. **Code quality standards:**
   - All code must pass type checking (mypy)
   - Add docstrings for all public functions
   - Write unit tests for new functionality

## Core Components

### State Space Model (SSM)
**Location:** `core/ssm.py`

A flexible state space model implementation for sequential data processing.

```python
from core.ssm import StateSpaceModel

# Initialize model
ssm = StateSpaceModel(
    state_dim=64,
    input_dim=10,
    output_dim=5
)

# Forward pass
import torch
x = torch.randn(32, 20, 10)  # (batch, seq_len, input_dim)
output, hidden = ssm(x)
```

**Key Features:**
- Efficient sequential processing
- Supports variable-length sequences
- GPU-accelerated computations

### MetaMAML
**Location:** `meta_rl/meta_maml.py`

Model-Agnostic Meta-Learning algorithm for rapid adaptation.

```python
from meta_rl.meta_maml import MetaMAML
from collections import OrderedDict

# Initialize MetaMAML
maml = MetaMAML(
    model=your_model,
    inner_lr=0.01,
    meta_lr=0.001
)

# Adapt to new task (returns OrderedDict of fast_weights)
fast_weights = maml.adapt(
    task_data,
    n_steps=5
)

# fast_weights is an OrderedDict containing updated parameters
assert isinstance(fast_weights, OrderedDict)

# Use adapted weights for prediction
predictions = maml.forward_with_weights(test_data, fast_weights)
```

**Key Methods:**
- `adapt(task_data, n_steps)`: Returns `fast_weights` (OrderedDict) - adapted model parameters
- `meta_update(batch_tasks)`: Update meta-parameters across tasks
- `forward_with_weights(data, weights)`: Forward pass with custom weights

**Important:** The `adapt()` method returns an `OrderedDict` containing the fast-adapted weights, NOT a loss value or info dict.

### Test-Time Adaptation
**Location:** `adaptation/test_time_adaptation.py`

Adaptive learning at test time for dynamic environments.

```python
from adaptation.test_time_adaptation import Adapter

# Initialize adapter
adapter = Adapter(
    model=your_model,
    lr=0.001,
    max_steps=10
)

# Adapt model (returns dict with info)
info = adapter.adapt(
    observation,
    target=None  # Can be None for unsupervised adaptation
)

# info is a dictionary containing adaptation details
assert isinstance(info, dict)
loss = info['loss']  # Extract the loss value
steps = info['steps']  # Number of adaptation steps taken
converged = info.get('converged', False)  # Whether adaptation converged

print(f"Adaptation loss: {loss:.4f}, steps: {steps}")
```

**Key Methods:**
- `adapt(observation, target)`: Returns `dict` with keys:
  - `'loss'`: Final adaptation loss (float)
  - `'steps'`: Number of steps taken (int)
  - `'updated'`: Whether model was updated (bool)
  - `'converged'`: Whether adaptation converged (bool, optional)

**Important:** The `adapt()` method returns a `dict` with adaptation info, NOT just a loss value or OrderedDict.

## Usage Examples

### Basic Training Pipeline
```python
import torch
from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter
from collections import OrderedDict

# Setup models
ssm = StateSpaceModel(state_dim=64, input_dim=10, output_dim=5)
maml = MetaMAML(model=ssm, inner_lr=0.01, meta_lr=0.001)

# Meta-training
for task_batch in meta_train_loader:
    # Adapt returns OrderedDict of fast_weights
    adapted_weights_list = []
    for task in task_batch:
        fast_weights = maml.adapt(task.support_data, n_steps=5)
        assert isinstance(fast_weights, OrderedDict)
        adapted_weights_list.append(fast_weights)
    
    # Meta-update
    maml.meta_update(task_batch)

# Test-time adaptation (different from meta-learning)
adapter = Adapter(model=ssm, lr=0.001, max_steps=10)
for test_obs in test_loader:
    # Adapt returns dict with info
    info = adapter.adapt(test_obs)
    assert isinstance(info, dict)
    
    # Extract values from info dict
    loss = info['loss']
    steps = info['steps']
    print(f"Test adaptation: loss={loss:.4f}, steps={steps}")
```

### Quick Benchmark
Run comprehensive benchmarks:

```bash
python experiments/quick_benchmark.py
```

The benchmark tests both meta-learning and test-time adaptation:
- **MetaMAML benchmark**: Validates that `adapt()` returns OrderedDict
- **Adapter benchmark**: Validates that `adapt()` returns dict with 'loss' key

## Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Test Individual Components
```bash
# Test SSM
python -m pytest tests/test_ssm.py -v

# Test MetaMAML (expects OrderedDict from adapt)
python -m pytest tests/test_meta_rl.py -v

# Test Adapter (expects dict from adapt)
python -m pytest tests/test_adaptation.py -v
```

### Test Coverage
```bash
python -m pytest --cov=. --cov-report=html
```

## API Reference

### MetaMAML.adapt() - Returns OrderedDict
```python
def adapt(self, task_data, n_steps: int = 5) -> OrderedDict:
    """
    Adapt model to a specific task using gradient descent.
    
    Args:
        task_data: Task-specific training data
        n_steps: Number of adaptation steps
    
    Returns:
        OrderedDict: Fast-adapted weights (parameter dictionary)
    
    Example:
        >>> fast_weights = maml.adapt(task_data, n_steps=5)
        >>> assert isinstance(fast_weights, OrderedDict)
        >>> predictions = maml.forward_with_weights(test_data, fast_weights)
    """
```

### Adapter.adapt() - Returns Dict
```python
def adapt(self, observation, target=None) -> Dict[str, Any]:
    """
    Adapt model at test time using current observation.
    
    Args:
        observation: Current observation
        target: Optional target for supervised adaptation
    
    Returns:
        dict: Adaptation information with keys:
            - 'loss' (float): Final adaptation loss
            - 'steps' (int): Number of adaptation steps
            - 'updated' (bool): Whether model was updated
            - 'converged' (bool): Whether adaptation converged
    
    Example:
        >>> info = adapter.adapt(observation)
        >>> assert isinstance(info, dict)
        >>> loss = info['loss']
        >>> print(f"Loss: {loss:.4f}")
    """
```

## Troubleshooting

### Common Issues

#### Type Errors with adapt() Methods
**Problem:** Getting unexpected types from `adapt()` calls.

**Solution:**
- `MetaMAML.adapt()` returns `OrderedDict` (fast_weights)
- `Adapter.adapt()` returns `dict` (info with 'loss', 'steps', etc.)

Make sure you're using the correct method for your use case:
```python
# For meta-learning (MetaMAML)
from collections import OrderedDict
fast_weights = maml.adapt(task_data)
assert isinstance(fast_weights, OrderedDict)

# For test-time adaptation (Adapter)
info = adapter.adapt(observation)
assert isinstance(info, dict)
loss = info['loss']
```

#### Import Errors
**Problem:** Cannot import modules.

**Solution:**
```bash
# Ensure you're in the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### CUDA Out of Memory
**Problem:** GPU memory errors.

**Solution:**
- Reduce batch size
- Use gradient checkpointing
- Clear cache: `torch.cuda.empty_cache()`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the development guidelines
4. Ensure all tests pass
5. Commit with descriptive message (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License
MIT License - see LICENSE file for details

## Citation
If you use this code in your research, please cite:
```bibtex
@software{ssm_metarl_testcompute,
  title={SSM-MetaRL-TestCompute: A Framework for Meta-Learning with State Space Models},
  author={Your Name},
  year={2025},
  url={https://github.com/sunghunkwag/SSM-MetaRL-TestCompute}
}
```

## Contact
For questions or issues, please open a GitHub issue or contact the maintainers.
