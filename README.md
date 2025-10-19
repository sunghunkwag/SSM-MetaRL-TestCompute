# SSM-MetaRL-TestCompute
This repository implements a minimal SSM (State Space Model) architecture integrated with meta-reinforcement learning and test-time compute for weight adaptation.

## Overview
This project combines three key concepts:

### State Space Models (SSM)
State Space Models provide an efficient way to model sequential data through state transition dynamics. Our implementation uses a minimal SSM architecture that maintains hidden states and applies linear transformations for sequence processing.

### Meta-Reinforcement Learning (Meta-RL)
Meta-RL enables agents to learn how to learn across multiple tasks. This implementation follows a MAML-like (Model-Agnostic Meta-Learning) structure where the model learns initialization parameters that can be quickly adapted to new tasks with minimal gradient steps.

### Test-Time Compute
Test-time compute refers to the ability to perform additional computation during inference to adapt model weights online. This allows the model to fine-tune its parameters based on test-time observations, improving performance on new or shifted distributions.

## Architecture
The system consists of:
- **SSM Core**: Minimal state space model for sequence encoding
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

## Usage
This is an experimental implementation, not production-ready code. To run the minimal example:
```bash
# Clone the repository
git clone https://github.com/sunghunkwag/SSM-MetaRL-TestCompute.git
cd SSM-MetaRL-TestCompute

# Run the main script
python main.py
```

### Requirements
- Python 3.8+
- NumPy
- PyTorch (recommended for future extensions)

### What to Expect
The code provides:
- A minimal SSM implementation with state transitions
- Skeleton structure for meta-RL training (MAML-style)
- Placeholder functions for test-time weight adaptation
- Detailed comments explaining where full logic would be implemented

This is designed for educational and experimental purposes to understand the integration of these three concepts.
