# SSM-MetaRL-TestCompute

A research implementation combining State Space Models (SSM) with Meta-Reinforcement Learning for fast adaptation to new tasks.

## Overview

This repository implements a hybrid approach that integrates:
- State Space Models: For efficient sequence modeling and long-range dependencies
- Meta-Learning Loop: Outer loop for meta-training across task distributions
- Test-Time Adaptation: Online weight updates during inference using gradient-based adaptation

## Core SSM Module

The `core/` directory contains the foundational State Space Model implementation:

### `core/ssm.py`

A full-featured State Space Model class that implements:
- Linear state transition dynamics: standard SSM formulation with learnable matrices A, B, C, D
- Recurrent cell processing: maintains hidden state across sequence steps
- Observation and output modules: transforms inputs to outputs through state space
- Model persistence: save and load methods for model checkpoints
- Comprehensive test suite: standalone tests for all functionality

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
- SSMMetaPolicy: Neural network policy that integrates State Space Model components for sequential decision making
- MetaMAML: Complete meta-learning orchestrator that manages inner-loop updates, task sampling, and meta-optimization

## Test-Time Adaptation Module

The `adaptation/` directory provides mechanisms to adapt models online during inference.

### Architecture

- Adapter: Controls parameter selection, optimizer, stability guards (grad clipping, max step), EMA, and trust region bounding.
- Strategy: Modular augmentation to the loss for adaptation (e.g., Tent-style entropy minimization). Strategies are registered via a registry pattern for easy extension.
- Hooks: Integration points to SSMs and Meta-RL components for feature extraction and policy/value losses.

### File: `adaptation/test_time_adaptation.py`

Key capabilities:
- Online Weight Updates: Safe, bounded gradient steps during inference with Adam optimizer.
- Meta-Feature Monitoring: Predictive entropy, confidence, feature-shift tracking with exponential moving statistics.
- Uncertainty Estimation: Entropy and variance metrics, plus placeholders for epistemic proxies.
- Stability Guards: Gradient clipping, per-parameter step bounding, optional trust region via EMA anchor.
- EMA and Trust Region: Optional exponential moving average of parameters used to constrain drift.
- Strategy Registry: Built-in `none` and `tent` strategies; register new ones with `register_adaptation_strategy`.
- Integration Hooks: `SSMHook` for feature extraction, `MetaRLHook` for RL policy losses.

### Minimal Usage Example
```python
from adaptation.test_time_adaptation import create_adapter, AdaptationConfig, SSMHook
adapter = create_adapter(model, AdaptationConfig(lr=5e-5, entropy_weight=0.01), strategy="tent")
ssm_hook = SSMHook(adapter).attach(model)

def fwd(batch):
    return model(**batch)

def loss_fn(outputs, batch):
    y = batch["labels"]
    logits = outputs["logits"]
    return F.cross_entropy(logits, y)

with adapter.context(model):
    outputs = fwd(batch)
    feats = ssm_hook.features(batch)
    mf = adapter.observe(outputs, batch, feats)
    info = adapter.adapt(loss_fn, batch, fwd_fn=fwd)
```

### Extending Adaptation Strategies
```python
from adaptation.test_time_adaptation import register_adaptation_strategy, AdaptationStrategy

@register_adaptation_strategy("my_strategy")
class MyStrategy(AdaptationStrategy):
    def post_loss(self, loss, outputs, batch):
        reg = some_regularizer(outputs, batch)
        return loss + 0.1 * reg
```

## Integration & Pipeline

The unified `main.py` serves as the orchestration script that wires together all repository modules into a cohesive end-to-end pipeline.

### Architecture Overview

`main.py` integrates four core components:
1. **SSM Policy** (`core/ssm.py`): State Space Model for sequential decision-making
2. **Meta-Learner** (`meta_rl/meta_maml.py`): MAML-based meta-training orchestrator
3. **Environment Batch** (`env_runner/environment.py`): Parallel environment execution
4. **Test-Time Adapter** (`adaptation/test_time_adaptation.py`): Online adaptation during inference

### Pipeline Orchestration Logic

#### Training Flow
1. **Initialization**: Create SSM policy with random parameters
2. **Task Sampling**: Sample batch of tasks from environment distribution
3. **Inner Loop**: For each task, adapt policy parameters using MAML inner updates
4. **Outer Loop**: Aggregate gradients across tasks and update meta-initialization
5. **Checkpointing**: Save parameters periodically and at end of training

#### Evaluation Flow
1. **Load Checkpoint**: Restore trained meta-initialization parameters
2. **Policy Creation**: Instantiate SSM policy with loaded parameters
3. **Rollout**: Execute policy in environment without adaptation
4. **Metrics**: Collect and report mean/std returns across episodes

#### Adaptation Flow
1. **Load Checkpoint**: Restore trained meta-initialization
2. **Per-Episode Adaptation**: 
   - Clone base parameters for each episode
   - Pre-rollout adaptation: Run test-time adapter for K steps
   - Online adaptation (optional): Update parameters during rollout based on observations
3. **Rollout with Adapted Policy**: Execute adapted policy in environment
4. **Metrics**: Report performance improvements from adaptation

### Command-Line Interface

The pipeline provides three operational modes accessible via CLI:

#### Train Mode
Meta-learning training to find good initialization parameters:
```bash
python main.py train --config basic --outer-steps 100 --tasks-per-batch 8
```

**Arguments:**
- `--config`: Experiment configuration (default: `basic`)
- `--seed`: Random seed for reproducibility
- `--outer-steps`: Number of meta-learning outer loop iterations
- `--tasks-per-batch`: Tasks per meta-batch for gradient aggregation
- `--ckpt-dir`: Directory for saving checkpoints

#### Eval Mode
Evaluate trained policy without test-time adaptation:
```bash
python main.py eval --checkpoint checkpoints/latest.npz --episodes 20
```

**Arguments:**
- `--checkpoint`: Path to checkpoint file (required)
- `--config`: Experiment configuration (default: `basic`)
- `--episodes`: Number of evaluation episodes

#### Adapt Mode
Evaluate policy with test-time adaptation:
```bash
python main.py adapt --checkpoint checkpoints/latest.npz --adapt-steps 10 --online
```

**Arguments:**
- `--checkpoint`: Path to checkpoint file (required)
- `--config`: Experiment configuration
- `--episodes`: Number of evaluation episodes
- `--adapt-steps`: Number of adaptation gradient steps before rollout
- `--online`: Enable online adaptation during rollout (flag)

### Example Configurations

The pipeline includes built-in experiment configs:

**Basic Config**:
```python
ExperimentConfig(
    seed=42,
    env_name="MetaGymToy-v0",
    num_envs=8,
    time_limit=200,
    meta=MetaConfig(outer_steps=50, tasks_per_batch=8, inner_steps=1, inner_lr=1e-2),
    adapt=AdaptConfig(steps=5, lr=1e-2),
)
```

### Example Runs

**Complete workflow:**
```bash
# 1. Train meta-initialization
python main.py train --config basic --outer-steps 100

# 2. Evaluate without adaptation
python main.py eval --checkpoint checkpoints/latest.npz --episodes 50

# 3. Evaluate with test-time adaptation
python main.py adapt --checkpoint checkpoints/latest.npz --adapt-steps 5 --episodes 50

# 4. Compare adaptation benefits
# Observe performance delta between eval and adapt modes
```

**Custom hyperparameters:**
```bash
# Train with more tasks per batch
python main.py train --outer-steps 200 --tasks-per-batch 16

# Adapt with stronger online updates
python main.py adapt --checkpoint checkpoints/step_100.npz --adapt-steps 20 --online
```

### Extensibility

The pipeline is designed for easy extension:
- **New environments**: Implement `EnvConfig` and `EnvBatch` interfaces
- **New architectures**: Subclass `SSM` with custom state transitions
- **New adaptation strategies**: Register via `@register_adaptation_strategy`
- **New meta-learning algorithms**: Implement `MetaLearner` interface

### References
- Wang et al., Tent: Fully Test-Time Adaptation by Entropy Minimization (ICLR 2021)
- Sun et al., Test-Time Training with Self-Supervision for Generalization under Distribution Shifts (ICML 2020)
- Nado et al., T3A: Test-time Template Adjustments (NeurIPS 2020 Workshop)
- Finn et al., Model-Agnostic Meta-Learning (MAML) (ICML 2017)
- Gu et al., Low-Rank State-Space Models (NeurIPS 2021); Smith et al., HiPPO (NeurIPS 2020)
