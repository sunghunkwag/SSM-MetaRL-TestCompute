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
```
bash
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
```
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
```
from adaptation.test_time_adaptation import register_adaptation_strategy, AdaptationStrategy

@register_adaptation_strategy("my_strategy")
class MyStrategy(AdaptationStrategy):
    def post_loss(self, loss, outputs, batch):
        reg = some_regularizer(outputs, batch)
        return loss + 0.1 * reg
```

### References
- Wang et al., Tent: Fully Test-Time Adaptation by Entropy Minimization (ICLR 2021)
- Sun et al., Test-Time Training with Self-Supervision for Generalization under Distribution Shifts (ICML 2020)
- Nado et al., T3A: Test-time Template Adjustments (NeurIPS 2020 Workshop)
- Finn et al., Model-Agnostic Meta-Learning (MAML) (ICML 2017)
- Gu et al., Low-Rank State-Space Models (NeurIPS 2021); Smith et al., HiPPO (NeurIPS 2020)

