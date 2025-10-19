# SSM-MetaRL-TestCompute
A research implementation combining State Space Models (SSM) with Meta-Reinforcement Learning for fast adaptation to new tasks.

## Overview
This repository implements a hybrid approach that integrates:
- State Space Models: For efficient sequence modeling and long-range dependencies
- Meta-Learning Loop: Outer loop for meta-training across task distributions
- Test-Time Adaptation: Online weight updates during inference using gradient-based adaptation

## Integration & Pipeline
The main entrypoint is main.py which orchestrates training, evaluation, and test-time adaptation. As of this release, an optional improvement pipeline can be enabled to selectively apply techniques defined in an external module, real_agi_continuous_improvement.py.

- Optional external module: real_agi_continuous_improvement.py (place at repo root)
- When present and enabled with --improve TAGS, the pipeline will call into the module to apply selected techniques.
- Improvements are applied:
  - post_meta: After meta-training (architecture/policy-level injections)
  - pre_adapt: Before evaluation or test-time adaptation (runtime/policy augmentations)

Supported tags (if the module exposes corresponding symbols):
- attention: inject_attention(policy)
- nas: neural_architecture_search(policy, budget)
- bn: enable_batch_norm(policy)
- recursive: enable_recursive_policies(policy)
- logger: attach_performance_logger(policy, stage) and/or log_training_metrics(step, metrics)
- ckpt_select: select_best_checkpoint(ckpt_dir)

Safety and behavior:
- The module is optional. If not found, the pipeline continues without improvements.
- Each tag is checked with hasattr before calling; errors are caught and skipped per-tag.
- NAS defaults to a small budget unless cfg.nas_budget is provided in the config.

### CLI
Common examples:
- Train with improvements
  python main.py train --config basic --outer-steps 100 --improve attention nas bn recursive
- Evaluate with improvements
  python main.py eval --checkpoint checkpoints/latest.npz --episodes 20 --improve attention
- Adapt with improvements
  python main.py adapt --checkpoint checkpoints/latest.npz --episodes 20 --adapt-steps 10 --improve bn

Flags added:
- --improve TAG [TAG ...]: list of improvement tags to apply. Recognized: attention, nas, bn, recursive, logger, ckpt_select
- --online: enables online test-time adaptation during rollout (existing)
- --adapt-steps N: number of pre-rollout adaptation steps (existing)

### Performance logging and checkpoint selection
- If real_agi_continuous_improvement exposes log_training_metrics(step, metrics), the training loop will forward metrics each outer step.
- If attach_performance_logger(policy, stage) is present, it is invoked at post_meta or pre_adapt stages.
- The training loop tracks best checkpoints by mean_return if provided by the meta-learner.
- If select_best_checkpoint(ckpt_dir) is present, it is called at the end of training to recommend the best .npz.

### Where improvements are applied
- After meta-training: apply_improvements(stage='post_meta', policy, meta)
- Before eval/adapt: apply_improvements(stage='pre_adapt', policy)

## Core SSM Module
The core/ directory contains the foundational State Space Model implementation.

## Meta-RL Module
The meta_rl/ directory contains the MAML-based meta-learner for SSM policies.

## Test-Time Adaptation Module
The adaptation/ directory provides mechanisms to adapt models online during inference.
