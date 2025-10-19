#!/usr/bin/env python3
"""
Unified Main Pipeline for SSM-MetaRL-TestCompute
Orchestrates the end-to-end pipeline by integrating:
- core/ssm.py (SSM policy)
- meta_rl/meta_maml.py (Meta-RL training)
- env_runner/environment.py (environment batching)
- adaptation/test_time_adaptation.py (test-time adaptation)
- Optional improvements via real_agi_continuous_improvement.py when --improve flag is used
Usage:
  python main.py train --config basic --outer-steps 100 [--improve attention nas bn recursive]
  python main.py eval --checkpoint checkpoints/latest.npz --episodes 20 [--improve ...]
  python main.py adapt --checkpoint checkpoints/latest.npz --adapt-steps 10 [--improve ...]
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np

# Import all repo modules
try:
    from core.ssm import SSM, SSMConfig
    from meta_rl.meta_maml import MetaLearner, MetaConfig
    from env_runner.environment import EnvBatch, EnvConfig
    from adaptation.test_time_adaptation import TestTimeAdapter, AdaptConfig
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure all modules exist: core/ssm.py, meta_rl/meta_maml.py, env_runner/environment.py, adaptation/test_time_adaptation.py")
    sys.exit(1)

# Optional external improvement module
IMPROVE_MODULE = None
IMPROVE_AVAILABLE = False
try:
    # Expect this file to be placed at repo root as real_agi_continuous_improvement.py
    import importlib
    IMPROVE_MODULE = importlib.import_module('real_agi_continuous_improvement')
    IMPROVE_AVAILABLE = True
except Exception as e:
    # Soft-fail: improvements are optional
    print(f"[Improve] Optional module not available: {e}")

# Experiment configuration
@dataclass
class ExperimentConfig:
    """Top-level configuration for experiments"""
    seed: int = 0
    env_name: str = "MetaGymToy-v0"
    num_envs: int = 8
    time_limit: int = 200
    ssm: SSMConfig = field(default_factory=SSMConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)
    log_dir: str = "runs"
    ckpt_dir: str = "checkpoints"
    # improvement flags selected by user
    improve: List[str] = field(default_factory=list)

# Built-in example configs
EXAMPLE_CONFIGS = {
    "basic": ExperimentConfig(
        seed=42,
        env_name="MetaGymToy-v0",
        num_envs=8,
        time_limit=200,
        meta=MetaConfig(outer_steps=50, tasks_per_batch=8, inner_steps=1, inner_lr=1e-2),
        adapt=AdaptConfig(steps=5, lr=1e-2),
    ),
}

# Utilities
def set_seed(seed: int):
    np.random.seed(seed)

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def save_checkpoint(path: str, params: Dict[str, np.ndarray]):
    ensure_dirs(os.path.dirname(path))
    np.savez(path, **params)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}

# Component builders
def build_env(cfg: ExperimentConfig) -> EnvBatch:
    env_cfg = EnvConfig(env_name=cfg.env_name, num_envs=cfg.num_envs, time_limit=cfg.time_limit, seed=cfg.seed)
    return EnvBatch(env_cfg)

def build_policy(cfg: ExperimentConfig) -> SSM:
    return SSM(cfg.ssm)

def build_meta_learner(cfg: ExperimentConfig) -> MetaLearner:
    return MetaLearner(cfg.meta)

def build_adapter(cfg: ExperimentConfig) -> TestTimeAdapter:
    return TestTimeAdapter(cfg.adapt)

# Improvement application
def apply_improvements(stage: str, cfg: ExperimentConfig, policy: SSM, meta: MetaLearner = None):
    """
    Apply selected improvements by referencing functions/classes from real_agi_continuous_improvement module.
    stage: 'post_meta' (after meta-training) or 'pre_adapt' (before test-time adaptation)
    """
    if not cfg.improve or not IMPROVE_AVAILABLE:
        return
    print(f"[Improve] Applying improvements at {stage}: {cfg.improve}")

    # Map friendly flags to module callables if present
    # We only call if attribute exists to keep integration selective/safe.
    for tag in cfg.improve:
        try:
            if tag in ("attention", "attn") and hasattr(IMPROVE_MODULE, 'inject_attention'):
                IMPROVE_MODULE.inject_attention(policy)
            elif tag in ("nas", "search") and hasattr(IMPROVE_MODULE, 'neural_architecture_search'):
                # We pass a lightweight search budget default
                new_arch = IMPROVE_MODULE.neural_architecture_search(policy, budget=getattr(cfg, 'nas_budget', 10))
                if hasattr(policy, 'reconfigure'):
                    policy.reconfigure(new_arch)
            elif tag in ("bn", "batch_norm") and hasattr(IMPROVE_MODULE, 'enable_batch_norm'):
                IMPROVE_MODULE.enable_batch_norm(policy)
            elif tag in ("recursive", "recursion") and hasattr(IMPROVE_MODULE, 'enable_recursive_policies'):
                IMPROVE_MODULE.enable_recursive_policies(policy)
            elif tag in ("logger",) and hasattr(IMPROVE_MODULE, 'attach_performance_logger'):
                IMPROVE_MODULE.attach_performance_logger(policy, stage=stage)
            elif tag in ("ckpt_select", "checkpoint") and hasattr(IMPROVE_MODULE, 'select_best_checkpoint'):
                # This tag is handled elsewhere with metrics, but allow here as no-op
                pass
        except Exception as e:
            print(f"[Improve] Skipped {tag} due to error: {e}")

# Training: Meta-learning outer loop
def train(cfg: ExperimentConfig):
    """End-to-end training loop: initialization + task sampling + meta-learning"""
    set_seed(cfg.seed)
    ensure_dirs(cfg.log_dir, cfg.ckpt_dir)

    envs = build_env(cfg)
    policy = build_policy(cfg)
    meta = build_meta_learner(cfg)

    params = policy.init_parameters(seed=cfg.seed)
    print("[Train] Starting meta-learning...")

    best_metric = -np.inf
    best_ckpt_path = None

    start = time.time()
    for step in range(cfg.meta.outer_steps):
        tasks = envs.sample_tasks(cfg.meta.tasks_per_batch)
        params, metrics = meta.outer_step(policy_class=SSM, init_params=params, tasks=tasks, ssm_cfg=cfg.ssm)

        # Performance logging via improvement module if available
        try:
            if IMPROVE_AVAILABLE and hasattr(IMPROVE_MODULE, 'log_training_metrics'):
                IMPROVE_MODULE.log_training_metrics(step=step, metrics=metrics)
        except Exception as e:
            print(f"[Improve] log_training_metrics error: {e}")

        # Periodic checkpointing
        if (step + 1) % 10 == 0 or step == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"step_{step+1}.npz")
            save_checkpoint(ckpt_path, params)
            print(f"[Train] step={step+1}/{cfg.meta.outer_steps} metrics={metrics}")

        # Track best by mean_return if available
        mean_ret = float(metrics.get('mean_return', -np.inf)) if isinstance(metrics, dict) else -np.inf
        if mean_ret > best_metric:
            best_metric = mean_ret
            best_ckpt_path = os.path.join(cfg.ckpt_dir, f"best_step_{step+1}.npz")
            save_checkpoint(best_ckpt_path, params)

    # After meta-training, optionally apply improvements to policy architecture
    policy.set_parameters(params)
    apply_improvements(stage='post_meta', cfg=cfg, policy=policy, meta=meta)
    # Save latest and possibly improved parameters
    latest = os.path.join(cfg.ckpt_dir, "latest.npz")
    save_checkpoint(latest, policy.get_parameters() if hasattr(policy, 'get_parameters') else params)

    # Use improvement module to select best checkpoint if available
    try:
        if IMPROVE_AVAILABLE and hasattr(IMPROVE_MODULE, 'select_best_checkpoint'):
            chosen = IMPROVE_MODULE.select_best_checkpoint(cfg.ckpt_dir)
            if chosen:
                print(f"[Improve] Selected best checkpoint: {chosen}")
    except Exception as e:
        print(f"[Improve] select_best_checkpoint error: {e}")

    print(f"[Train] Complete in {time.time()-start:.1f}s")
    return policy.get_parameters() if hasattr(policy, 'get_parameters') else params

# Evaluation: Rollouts without adaptation
def evaluate(cfg: ExperimentConfig, params: Dict[str, np.ndarray], episodes: int = 10):
    """Evaluate policy without test-time adaptation"""
    set_seed(cfg.seed)
    envs = build_env(cfg)
    policy = build_policy(cfg)
    policy.set_parameters(params)

    # Allow applying improvements before plain evaluation
    apply_improvements(stage='pre_adapt', cfg=cfg, policy=policy)

    print(f"[Eval] Running {episodes} episodes...")
    returns = []
    for ep in range(episodes):
        obs = envs.reset()
        policy.reset()
        ep_ret = np.zeros(cfg.num_envs)
        for _ in range(cfg.time_limit):
            actions = policy.act(obs)
            obs, rew, done, _ = envs.step(actions)
            ep_ret += rew
            if done.all():
                break
        returns.extend(ep_ret.tolist())

    metrics = {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns))}
    print(f"[Eval] Results: {metrics}")
    return metrics

# Adaptation: Test-time adaptation during inference
def evaluate_with_adaptation(cfg: ExperimentConfig, params: Dict[str, np.ndarray], episodes: int = 10):
    """Evaluate policy with test-time adaptation"""
    set_seed(cfg.seed)
    envs = build_env(cfg)
    adapter = build_adapter(cfg)

    print(f"[Adapt] Running {episodes} episodes with adaptation...")
    returns = []
    for ep in range(episodes):
        obs = envs.reset()
        ep_params = dict(params)
        policy = build_policy(cfg)
        policy.set_parameters(ep_params)
        policy.reset()

        # Allow applying improvements before adaptation
        apply_improvements(stage='pre_adapt', cfg=cfg, policy=policy)

        # Pre-rollout adaptation
        if cfg.adapt.steps > 0:
            ep_params = adapter.adapt(policy, envs, steps=cfg.adapt.steps)
            policy.set_parameters(ep_params)

        ep_ret = np.zeros(cfg.num_envs)
        for _ in range(cfg.time_limit):
            actions = policy.act(obs)
            obs, rew, done, _ = envs.step(actions)
            ep_ret += rew

            # Online adaptation during rollout
            if getattr(cfg.adapt, 'online', False):
                ep_params = adapter.online_step(policy, obs, rew)
                policy.set_parameters(ep_params)

            if done.all():
                break
        returns.extend(ep_ret.tolist())

    metrics = {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns))}
    print(f"[Adapt] Results: {metrics}")
    return metrics

# Command-line interface
def build_parser():
    parser = argparse.ArgumentParser(
        description="SSM-MetaRL-TestCompute: Unified pipeline for SSM-based Meta-RL with test-time adaptation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with basic config and improvements
  python main.py train --config basic --outer-steps 100 --improve attention nas bn recursive

  # Evaluate trained checkpoint with improvements
  python main.py eval --checkpoint checkpoints/latest.npz --episodes 20 --improve attention

  # Evaluate with test-time adaptation and improvements
  python main.py adapt --checkpoint checkpoints/latest.npz --episodes 20 --adapt-steps 10 --improve bn
"""
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operation mode", required=True)

    # Common improvement args for subcommands
    def add_improve_args(p):
        p.add_argument(
            "--improve",
            nargs='*',
            default=None,
            metavar='TAG',
            help="Optional improvement tags to apply: attention, nas, bn, recursive, logger, ckpt_select",
        )

    # Train mode
    train_parser = subparsers.add_parser("train", help="Train meta-initialization via meta-learning")
    train_parser.add_argument("--config", default="basic", choices=list(EXAMPLE_CONFIGS.keys()), help="Experiment config")
    train_parser.add_argument("--seed", type=int, help="Random seed")
    train_parser.add_argument("--outer-steps", type=int, help="Number of meta-learning outer steps")
    train_parser.add_argument("--tasks-per-batch", type=int, help="Tasks per meta-batch")
    train_parser.add_argument("--ckpt-dir", help="Checkpoint directory")
    add_improve_args(train_parser)

    # Eval mode
    eval_parser = subparsers.add_parser("eval", help="Evaluate policy without adaptation")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .npz file")
    eval_parser.add_argument("--config", default="basic", choices=list(EXAMPLE_CONFIGS.keys()), help="Experiment config")
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    add_improve_args(eval_parser)

    # Adapt mode
    adapt_parser = subparsers.add_parser("adapt", help="Evaluate policy with test-time adaptation")
    adapt_parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .npz file")
    adapt_parser.add_argument("--config", default="basic", choices=list(EXAMPLE_CONFIGS.keys()), help="Experiment config")
    adapt_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    adapt_parser.add_argument("--adapt-steps", type=int, help="Adaptation steps before rollout")
    adapt_parser.add_argument("--online", action="store_true", help="Enable online adaptation during rollout")
    add_improve_args(adapt_parser)

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load base config
    cfg = EXAMPLE_CONFIGS.get(getattr(args, 'config', 'basic'), EXAMPLE_CONFIGS["basic"])  # default fallback

    # Override with CLI args
    if hasattr(args, "seed") and args.seed is not None:
        cfg.seed = args.seed
    if hasattr(args, "outer_steps") and args.outer_steps is not None:
        cfg.meta.outer_steps = args.outer_steps
    if hasattr(args, "tasks_per_batch") and args.tasks_per_batch is not None:
        cfg.meta.tasks_per_batch = args.tasks_per_batch
    if hasattr(args, "ckpt_dir") and args.ckpt_dir is not None:
        cfg.ckpt_dir = args.ckpt_dir
    if hasattr(args, "adapt_steps") and args.adapt_steps is not None:
        cfg.adapt.steps = args.adapt_steps
    if hasattr(args, "online") and args.online:
        cfg.adapt.online = True
    # Improvement selection
    if hasattr(args, "improve") and args.improve is not None:
        cfg.improve = list(args.improve)

    print(f"{"=" * 60}")
    print(f"SSM-MetaRL-TestCompute Pipeline")
    print(f"Mode: {args.mode.upper()}")
    print(f"Config: {getattr(args, 'config', 'basic')}")
    print(f"{"="  * 60}")

    if args.mode == "train":
        train(cfg)
