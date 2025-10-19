#!/usr/bin/env python3
"""
Unified Main Pipeline for SSM-MetaRL-TestCompute

Orchestrates the end-to-end pipeline by integrating:
- core/ssm.py (SSM policy)
- meta_rl/meta_maml.py (Meta-RL training)
- env_runner/environment.py (environment batching)
- adaptation/test_time_adaptation.py (test-time adaptation)

Usage:
  python main.py train --config basic --outer-steps 100
  python main.py eval --checkpoint checkpoints/latest.npz --episodes 20
  python main.py adapt --checkpoint checkpoints/latest.npz --adapt-steps 10
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, Any
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
    
    start = time.time()
    for step in range(cfg.meta.outer_steps):
        tasks = envs.sample_tasks(cfg.meta.tasks_per_batch)
        params, metrics = meta.outer_step(policy_class=SSM, init_params=params, tasks=tasks, ssm_cfg=cfg.ssm)
        
        if (step + 1) % 10 == 0 or step == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"step_{step+1}.npz")
            save_checkpoint(ckpt_path, params)
            print(f"[Train] step={step+1}/{cfg.meta.outer_steps} metrics={metrics}")
    
    latest = os.path.join(cfg.ckpt_dir, "latest.npz")
    save_checkpoint(latest, params)
    print(f"[Train] Complete in {time.time()-start:.1f}s")
    return params


# Evaluation: Rollouts without adaptation
def evaluate(cfg: ExperimentConfig, params: Dict[str, np.ndarray], episodes: int = 10):
    """Evaluate policy without test-time adaptation"""
    set_seed(cfg.seed)
    envs = build_env(cfg)
    policy = build_policy(cfg)
    policy.set_parameters(params)
    
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
            if cfg.adapt.online:
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
  # Train with basic config
  python main.py train --config basic --outer-steps 100
  
  # Evaluate trained checkpoint
  python main.py eval --checkpoint checkpoints/latest.npz --episodes 20
  
  # Evaluate with test-time adaptation
  python main.py adapt --checkpoint checkpoints/latest.npz --episodes 20 --adapt-steps 10
"""
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode", required=True)
    
    # Train mode
    train_parser = subparsers.add_parser("train", help="Train meta-initialization via meta-learning")
    train_parser.add_argument("--config", default="basic", choices=list(EXAMPLE_CONFIGS.keys()), help="Experiment config")
    train_parser.add_argument("--seed", type=int, help="Random seed")
    train_parser.add_argument("--outer-steps", type=int, help="Number of meta-learning outer steps")
    train_parser.add_argument("--tasks-per-batch", type=int, help="Tasks per meta-batch")
    train_parser.add_argument("--ckpt-dir", help="Checkpoint directory")
    
    # Eval mode
    eval_parser = subparsers.add_parser("eval", help="Evaluate policy without adaptation")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .npz file")
    eval_parser.add_argument("--config", default="basic", choices=list(EXAMPLE_CONFIGS.keys()), help="Experiment config")
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    
    # Adapt mode
    adapt_parser = subparsers.add_parser("adapt", help="Evaluate policy with test-time adaptation")
    adapt_parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .npz file")
    adapt_parser.add_argument("--config", default="basic", choices=list(EXAMPLE_CONFIGS.keys()), help="Experiment config")
    adapt_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    adapt_parser.add_argument("--adapt-steps", type=int, help="Adaptation steps before rollout")
    adapt_parser.add_argument("--online", action="store_true", help="Enable online adaptation during rollout")
    
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # Load base config
    cfg = EXAMPLE_CONFIGS.get(args.config, EXAMPLE_CONFIGS["basic"])
    
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
    
    print(f"=" * 60)
    print(f"SSM-MetaRL-TestCompute Pipeline")
    print(f"Mode: {args.mode.upper()}")
    print(f"Config: {args.config}")
    print(f"="  * 60)
    
    if args.mode == "train":
        train(cfg)
    elif args.mode == "eval":
        params = load_checkpoint(args.checkpoint)
        evaluate(cfg, params, episodes=args.episodes)
    elif args.mode == "adapt":
        params = load_checkpoint(args.checkpoint)
        evaluate_with_adaptation(cfg, params, episodes=args.episodes)
    else:
        parser.print_help()
        sys.exit(1)
    
    print(f"\nDone.")


if __name__ == "__main__":
    main()
