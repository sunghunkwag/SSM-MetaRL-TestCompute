#!/usr/bin/env python3
"""
SSM-MetaRL-TestCompute: Unified pipeline for SSM-based Meta-RL with test-time adaptation

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
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# ----------------------
# Pre-run system checks
# ----------------------
MIN_PY = (3, 9)

def _check_prereqs() -> None:
    py_ok = sys.version_info >= MIN_PY
    if not py_ok:
        warnings.warn(
            f"Python {MIN_PY[0]}.{MIN_PY[1]}+ recommended. Current: {sys.version.split()[0]}.")
    # numpy required, torch optional but recommended
    try:
        import numpy as _np  # noqa: F401
    except Exception:
        print("[Prereq] numpy not found. Install: pip install numpy")
        sys.exit(1)
    try:
        import torch  # noqa: F401
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    except Exception:
        warnings.warn("PyTorch not found. Some features may be disabled. Install: pip install torch --index-url https://download.pytorch.org/whl/cpu")

_check_prereqs()

import numpy as np

# ----------------------
# Import repo modules with validation/repair hints
# ----------------------
MISSING_HINT = (
    "Ensure these files exist with correct paths: "
    "core/ssm.py, meta_rl/meta_maml.py, env_runner/environment.py, adaptation/test_time_adaptation.py"
)

try:
    from core.ssm import SSM, SSMConfig
except Exception as e:
    print(f"[Import] Failed to import core.ssm: {e}\n{MISSING_HINT}")
    raise

try:
    from meta_rl.meta_maml import MetaLearner, MetaConfig
except Exception as e:
    print(f"[Import] Failed to import meta_rl.meta_maml: {e}\n{MISSING_HINT}")
    raise

try:
    from env_runner.environment import EnvBatch, EnvConfig
except Exception as e:
    print(f"[Import] Failed to import env_runner.environment: {e}\n{MISSING_HINT}")
    raise

try:
    from adaptation.test_time_adaptation import TestTimeAdapter, AdaptConfig
except Exception as e:
    print(f"[Import] Failed to import adaptation.test_time_adaptation: {e}\n{MISSING_HINT}")
    raise

# ----------------------
# Optional external improvement module (secured)
# ----------------------
IMPROVE_MODULE = None
IMPROVE_AVAILABLE = False
VALID_IMPROVE_TAGS = (
    "attention", "attn", "nas", "search", "bn", "batch_norm",
    "recursive", "recursion", "logger", "ckpt_select", "checkpoint",
)
IMPROVE_WHITELIST = {
    "inject_attention",
    "neural_architecture_search",
    "enable_batch_norm",
    "enable_recursive_policies",
    "attach_performance_logger",
    "select_best_checkpoint",
}
try:
    import importlib
    IMPROVE_MODULE = importlib.import_module('real_agi_continuous_improvement')
    # Whitelist enforcement: only allow expected attributes
    for attr in list(vars(IMPROVE_MODULE)):
        if attr.startswith("__"):
            continue
        if attr not in IMPROVE_WHITELIST:
            try:
                delattr(IMPROVE_MODULE, attr)
            except Exception:
                pass
    IMPROVE_AVAILABLE = True
except Exception as e:
    warnings.warn(f"[Improve] Optional module not available: {e}")

# ----------------------
# Experiment configuration
# ----------------------
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

# ----------------------
# Utilities
# ----------------------

def set_seed(seed: int):
    np.random.seed(seed)


def ensure_dirs(*paths):
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)


def save_checkpoint(path: str, params: Dict[str, np.ndarray]):
    """Save parameters as NumPy .npz (keys->arrays). Avoid pickles; portable.
    Limitations: arrays only; dtype/shape must match on load; size ~sum of arrays.
    """
    ensure_dirs(os.path.dirname(path))
    np.savez(path, **params)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path: str) -> Dict[str, np.ndarray]:
    """Load parameters from .npz created by save_checkpoint."""
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}

# ----------------------
# Component builders
# ----------------------

def build_env(cfg: ExperimentConfig) -> EnvBatch:
    env_cfg = EnvConfig(env_name=cfg.env_name, num_envs=cfg.num_envs, time_limit=cfg.time_limit, seed=cfg.seed)
    return EnvBatch(env_cfg)


def build_policy(cfg: ExperimentConfig) -> SSM:
    return SSM(cfg.ssm)


def build_meta_learner(cfg: ExperimentConfig) -> MetaLearner:
    return MetaLearner(cfg.meta)


def build_adapter(cfg: ExperimentConfig) -> TestTimeAdapter:
    return TestTimeAdapter(cfg.adapt)

# ----------------------
# Improvement application
# ----------------------

def _warn_unknown_improve_tags(tags: List[str]) -> List[str]:
    unknown = [t for t in tags if t not in VALID_IMPROVE_TAGS]
    if unknown:
        warnings.warn(f"Unknown --improve tags: {unknown}. Valid: {sorted(set(VALID_IMPROVE_TAGS))}")
    return [t for t in tags if t in VALID_IMPROVE_TAGS]


def apply_improvements(stage: str, cfg: ExperimentConfig, policy: SSM, meta: Optional[MetaLearner] = None):
    """Apply selected improvements safely. stage in {'post_meta','pre_adapt'}."""
    if not cfg.improve:
        return
    if not IMPROVE_AVAILABLE:
        warnings.warn("[Improve] Module not available; skipping improvements.")
        return
    cfg.improve = _warn_unknown_improve_tags(cfg.improve)
    if not cfg.improve:
        return
    print(f"[Improve] Applying improvements at {stage}: {cfg.improve}")
    for tag in cfg.improve:
        try:
            if tag in ("attention", "attn") and hasattr(IMPROVE_MODULE, 'inject_attention'):
                IMPROVE_MODULE.inject_attention(policy)
            elif tag in ("nas", "search") and hasattr(IMPROVE_MODULE, 'neural_architecture_search'):
                new_arch = IMPROVE_MODULE.neural_architecture_search(policy, budget=getattr(cfg, 'nas_budget', 10))
                if hasattr(policy, 'reconfigure') and new_arch is not None:
                    policy.reconfigure(new_arch)
            elif tag in ("bn", "batch_norm") and hasattr(IMPROVE_MODULE, 'enable_batch_norm'):
                IMPROVE_MODULE.enable_batch_norm(policy)
            elif tag in ("recursive", "recursion") and hasattr(IMPROVE_MODULE, 'enable_recursive_policies'):
                IMPROVE_MODULE.enable_recursive_policies(policy)
            elif tag in ("logger",) and hasattr(IMPROVE_MODULE, 'attach_performance_logger'):
                IMPROVE_MODULE.attach_performance_logger(policy, stage=stage)
            elif tag in ("ckpt_select", "checkpoint") and hasattr(IMPROVE_MODULE, 'select_best_checkpoint'):
                pass
        except Exception as e:
            warnings.warn(f"[Improve] Skipped {tag} due to error: {e}")

# ----------------------
# Training / Evaluation
# ----------------------

def train(cfg: ExperimentConfig):
    set_seed(cfg.seed)
    ensure_dirs(cfg.log_dir, cfg.ckpt_dir)
    envs = build_env(cfg)
    policy = build_policy(cfg)
    meta = build_meta_learner(cfg)
    params = policy.init_parameters(seed=cfg.seed)
    print("[Train] Starting meta-learning...")
    best_metric = -np.inf
    start = time.time()
    for step in range(cfg.meta.outer_steps):
        tasks = envs.sample_tasks(cfg.meta.tasks_per_batch)
        params, metrics = meta.outer_step(policy_class=SSM, init_params=params, tasks=tasks, ssm_cfg=cfg.ssm)
        try:
            if IMPROVE_AVAILABLE and hasattr(IMPROVE_MODULE, 'log_training_metrics'):
                IMPROVE_MODULE.log_training_metrics(step=step, metrics=metrics)
        except Exception as e:
            warnings.warn(f"[Improve] log_training_metrics error: {e}")
        if (step + 1) % 10 == 0 or step == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"step_{step+1}.npz")
            save_checkpoint(ckpt_path, params)
            print(f"[Train] step={step+1}/{cfg.meta.outer_steps} metrics={metrics}")
        mean_ret = float(metrics.get('mean_return', -np.inf)) if isinstance(metrics, dict) else -np.inf
        if mean_ret > best_metric:
            best_metric = mean_ret
            best_ckpt_path = os.path.join(cfg.ckpt_dir, f"best_step_{step+1}.npz")
            save_checkpoint(best_ckpt_path, params)
    policy.set_parameters(params)
    apply_improvements(stage='post_meta', cfg=cfg, policy=policy, meta=meta)
    latest = os.path.join(cfg.ckpt_dir, "latest.npz")
    save_checkpoint(latest, policy.get_parameters() if hasattr(policy, 'get_parameters') else params)
    try:
        if IMPROVE_AVAILABLE and hasattr(IMPROVE_MODULE, 'select_best_checkpoint'):
            chosen = IMPROVE_MODULE.select_best_checkpoint(cfg.ckpt_dir)
            if chosen:
                print(f"[Improve] Selected best checkpoint: {chosen}")
    except Exception as e:
        warnings.warn(f"[Improve] select_best_checkpoint error: {e}")
    print(f"[Train] Complete in {time.time()-start:.1f}s")
    return policy.get_parameters() if hasattr(policy, 'get_parameters') else params


def evaluate(cfg: ExperimentConfig, params: Dict[str, np.ndarray], episodes: int = 10):
    set_seed(cfg.seed)
    envs = build_env(cfg)
    policy = build_policy(cfg)
    policy.set_parameters(params)
    apply_improvements(stage='pre_adapt', cfg=cfg, policy=policy)
    print(f"[Eval] Running {episodes} episodes...")
    returns: List[float] = []
    for _ in range(episodes):
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


def evaluate_with_adaptation(cfg: ExperimentConfig, params: Dict[str, np.ndarray], episodes: int = 10):
    set_seed(cfg.seed)
    envs = build_env(cfg)
    adapter = build_adapter(cfg)
    print(f"[Adapt] Running {episodes} episodes with adaptation...")
    returns: List[float] = []
    for _ in range(episodes):
        obs = envs.reset()
        ep_params = dict(params)
        policy = build_policy(cfg)
        policy.set_parameters(ep_params)
        policy.reset()
        apply_improvements(stage='pre_adapt', cfg=cfg, policy=policy)
        if cfg.adapt.steps > 0:
            ep_params = adapter.adapt(policy, envs, steps=cfg.adapt.steps)
            policy.set_parameters(ep_params)
        ep_ret = np.zeros(cfg.num_envs)
        for _ in range(cfg.time_limit):
            actions = policy.act(obs)
            obs, rew, done, _ = envs.step(actions)
            ep_ret += rew
            if getattr(cfg.adapt, 'online', False):
                ep_params = adapter.online_step(policy, obs, rew)
                policy.set_parameters(ep_params)
            if done.all():
                break
        returns.extend(ep_ret.tolist())
    metrics = {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns))}
    print(f"[Adapt] Results: {metrics}")
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(
        description="SSM-MetaRL-TestCompute: Unified pipeline for SSM-based Meta-RL with test-time adaptation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Train with basic config and improvements\n"
            "  python main.py train --config basic --outer-steps 100 --improve attention nas bn recursive\n"
            "  # Evaluate trained checkpoint with improvements\n"
            "  python main.py eval --checkpoint checkpoints/latest.npz --episodes 20 --improve attention\n"
            "  # Evaluate with test-time adaptation and improvements\n"
            "  python main.py adapt --checkpoint checkpoints/latest.npz --episodes 20 --adapt-steps 10 --improve bn\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode", required=True)

    def add_improve_args(p):
        p.add_argument(
            "--improve",
            nargs='*',
            default=None,
            metavar='TAG',
            help=("Optional improvement tags to apply: " + ", ".join(sorted(set(VALID_IMPROVE_TAGS)))),
        )

    train_parser = subparsers.add_parser("train", help="Train meta-initialization via meta-learning")
    train_parser.add_argument("--config", default="basic", choices=list(EXAMPLE_CONFIGS.keys()), help="Experiment config")
    train_parser.add_argument("--seed", type=int, help="Random seed")
    train_parser.add_argument("--outer-steps", type=int, help="Number of meta-learning outer steps")
    train_parser.add_argument("--tasks-per-batch", type=int, help="Tasks per meta-batch")
    train_parser.add_argument("--ckpt-dir", help="Checkpoint directory")
    add_improve_args(train_parser)

    eval_parser = subparsers.add_parser("eval", help="Evaluate policy without adaptation")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .npz file")
    eval_parser.add_argument("--config", default="basic", choices=list(EXAMPLE_CONFIGS.keys()), help="Experiment config")
    eval
