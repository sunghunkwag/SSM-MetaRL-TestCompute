# SSM-MetaRL-TestCompute

A research implementation combining State Space Models (SSM) with Meta-Reinforcement Learning for fast adaptation to new tasks.

## Overview

This repository implements a hybrid approach that integrates:

- **State Space Models**: For efficient sequence modeling and long-range dependencies
- **Meta-Learning Loop**: Outer loop for meta-training across task distributions
- **Test-Time Adaptation**: Online weight updates during inference using gradient-based adaptation

## Core SSM Module

The `core/` directory contains the foundational State Space Model implementation:

### `core/ssm.py`

A full-featured State Space Model class that implements:

- **Linear state transition dynamics**: Follows standard SSM formulation with learnable matrices A, B, C, D
- **Recurrent cell processing**: Maintains hidden state across sequence steps
- **Observation and output modules**: Transforms inputs to outputs through state space
- **Model persistence**: Save and load methods for model checkpoints
- **Comprehensive test suite**: Standalone tests for all functionality

#### Testing the Core Module

To run the SSM module tests:

```bash
python core/ssm.py
```

This will execute all test cases including:

- Basic initialization and forward pass
- Sequence processing capabilities
- State persistence and reset functionality
- Model save/load operations
- Parameter access and modification

The module is written entirely in English with clear educational comments for learning and extension.

## Meta-RL Module

The `meta_rl/` directory contains the meta-reinforcement learning implementation:

### `meta_rl/meta_maml.py`

A comprehensive MAML (Model-Agnostic Meta-Learning) implementation for SSM-based policies that includes:

- **SSMMetaPolicy**: Neural network policy that integrates State Space Model components for sequential decision making
- **MetaMAML**: Complete meta-learning orchestrator that manages:
  - Inner loop adaptation (task-specific fine-tuning)
  - Outer loop meta-updates (learning to learn)
  - Gradient computation through adaptation steps
- **Integration hooks**: Placeholder connections for SSM-based architectures
- **Test-time adaptation**: Methods for online weight updates during inference

#### Key Components

1. **Policy Network**: Combines SSM for sequence modeling with action prediction heads
2. **Inner Loop**: Fast adaptation to new tasks using few-shot gradient updates
3. **Outer Loop**: Meta-gradient computation for improving adaptation efficiency
4. **Task Sampling**: Support for multi-task training distributions

#### Testing the Meta-RL Module

Run the module tests:

```bash
python meta_rl/meta_maml.py
```

#### References

This implementation follows patterns from:

- MAML paper: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
- RL² (Fast RL via Slow RL): Duan et al., "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning"
- Meta-World benchmark: Yu et al., "Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning"

## Environment Runner Module

The `env_runner/` directory contains the RL environment management system for meta-RL tasks:

### `env_runner/environment.py`

A comprehensive environment runner implementing interfaces compatible with Meta-World and RLlib:

- **Environment**: Base environment class providing unified RL interfaces
  - Standard `reset()` and `step()` methods following Gym API
  - Task identification for multi-task scenarios
  - Configurable episode length and seeding
  
- **BatchedEnvironment**: Parallel environment wrapper for efficient training
  - Manages multiple environment instances simultaneously
  - Vectorized operations for batched `reset()` and `step()`
  - Reduces training time through parallelization
  
- **MultiTaskEnvironment**: Multi-task support following Meta-World patterns
  - Task sampling and switching capabilities
  - Maintains task distributions for meta-learning
  - Supports both random sampling and explicit task specification
  
- **SSMPolicyIntegration**: Integration utilities for SSM-based policies
  - Observation sequence preparation for SSM input
  - Hidden state extraction from policy outputs
  - Meta-feature computation from trajectories
  - Bridges environment observations with SSM policy requirements

#### Key Features

1. **Gym-Compatible API**: Standard interfaces ensure compatibility with existing RL codebases
2. **Batched Operations**: Efficient parallel environment execution for faster training
3. **Multi-Task Support**: Native support for task distributions in meta-learning
4. **SSM Integration Hooks**: Purpose-built utilities for State Space Model policies
5. **Flexible Factory Pattern**: Easy environment creation with `create_environment()` function

#### Usage Example

```python
from env_runner.environment import create_environment, BatchedEnvironment

# Create single environment
env = create_environment('MetaWorld-reach-v2', batch_size=1)
obs = env.reset()
obs, reward, done, info = env.step(action)

# Create batched multi-task environment
env = create_environment(
    'MetaWorld-MT10',
    batch_size=16,
    multi_task=True,
    tasks=list(range(10))
)
observations = env.reset()  # Shape: (16, obs_dim)
observations, rewards, dones, infos = env.step(actions)  # Batched
```

#### Architecture

The environment runner is designed to work seamlessly with:

- **Meta-World**: Robotics manipulation tasks for meta-RL research
- **RLlib**: Scalable RL library with distributed training support
- **Custom environments**: Extensible base classes for domain-specific tasks

Environments provide the interaction layer between SSM-based policies (from `core/` and `meta_rl/`) and the task distribution, enabling:

1. Data collection during inner-loop adaptation
2. Multi-task training for meta-learning
3. Test-time evaluation on novel tasks
4. Trajectory batching for efficient SSM sequence processing

#### References

This implementation follows standards from:

- Meta-World: https://github.com/Farama-Foundation/Metaworld
- RLlib Documentation: https://docs.ray.io/en/latest/rllib/index.html
- OpenAI Gym: Standard RL environment API specification

## Future Extensions

Key areas for expansion:

- Implement proper RL loss functions (PPO, A2C, etc.)
- Add environment-specific adaptations
- Optimize gradient computation for large models

## Usage

This is an experimental implementation, not production-ready code. To run the minimal example:

```bash
# Clone the repository
git clone https://github.com/sunghunkwag/SSM-MetaRL-TestCompute.git
cd SSM-MetaRL-TestCompute

# Run the main script
python main.py

# Test individual modules
python core/ssm.py
python meta_rl/meta_maml.py
```

### Requirements

- Python 3.8+
- NumPy
- PyTorch (recommended for future extensions)

### What to Expect

The code provides:

- A minimal SSM implementation with state transitions
- Complete MAML meta-learning framework structure
- SSM-based policy integration patterns
- Placeholder functions for test-time weight adaptation
- Detailed comments explaining where full logic would be implemented
- Comprehensive research references and integration guides

This is designed for educational and experimental purposes to understand the integration of these three concepts: State Space Models, Meta-Learning, and Reinforcement Learning.
