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
- **MetaMAMLTrainer**: Complete meta-learning framework with outer/inner loop structure
- **TaskBatch**: Task management system for meta-training episodes
- **Fast Adaptation**: Gradient-based adaptation for quick learning on new tasks

#### Key Features

- **MAML-style Meta-Learning**: Implements the classic gradient-based meta-learning approach with inner and outer loops
- **SSM Integration**: Combines policy networks with SSM layers for enhanced sequential processing
- **Task Distribution Support**: Handles batched task training following Meta-World and similar benchmarks
- **Research References**: Extensively documented with references to key research implementations

#### Source References and Integration Approach

This implementation draws from several key research codebases:

1. **State-Spaces Repository** (`https://github.com/state-spaces/s4`)
   - SSM architecture patterns and layer implementations
   - S4 and Mamba integration points for policy learning
   - Efficient state space computations

2. **Original MAML Implementation** (`https://github.com/cbfinn/maml`)
   - Gradient-based meta-learning structure
   - Inner/outer loop optimization patterns
   - Fast adaptation mechanisms

3. **PyTorch MAML** (`https://github.com/tristandeleu/pytorch-maml`)
   - Modern PyTorch implementation patterns
   - Functional API usage for gradient computation
   - Efficient batched operations

4. **Distributionally Adaptive Meta-RL**
   - Task distribution management
   - Adaptive sampling strategies
   - Environment integration patterns

5. **Meta-World Benchmark** (`https://github.com/rlworkgroup/metaworld`)
   - Task batch structure and management
   - Environment interface standards
   - Evaluation protocols

#### Integration Strategy

The module follows a modular design that allows for:
- **Plug-and-play SSM components**: Easy integration of different SSM architectures (S4, Mamba, etc.)
- **Environment flexibility**: Support for various RL environments and task distributions
- **Research extensibility**: Clear integration points for new meta-learning algorithms
- **Performance optimization**: Efficient batched operations and gradient computations

#### Usage Example

```python
from meta_rl.meta_maml import SSMMetaPolicy, MetaMAMLTrainer, TaskBatch

# Create SSM-based meta-policy
policy = SSMMetaPolicy(state_dim=84, action_dim=4)

# Initialize meta-trainer
trainer = MetaMAMLTrainer(policy, meta_lr=1e-3, inner_lr=1e-2)

# Create task batch for meta-training
task_batch = TaskBatch(tasks, batch_size=16)

# Run meta-training
meta_losses = trainer.meta_train(task_batch, num_meta_iterations=1000)

# Adapt to new task
adapted_policy = trainer.adapt_to_new_task(new_task_data)
```

#### Current Implementation Status

The current implementation provides:
- Complete MAML framework structure
- SSM integration placeholders and patterns
- Comprehensive documentation and research references
- Extensible architecture for various SSM types
- Task management and batch processing

**Integration TODOs** (marked in code):
- Connect with actual SSM layers from `core/ssm.py`
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
