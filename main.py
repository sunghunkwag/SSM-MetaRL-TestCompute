#!/usr/bin/env python3
"""
SSM-MetaRL-TestCompute (PyTorch): Unified pipeline for SSM-based Meta-RL with test-time adaptation
- models/ssm.py (SSM policy, nn.Module)
- meta/maml.py (Meta-MAML trainer)
- envs/vector_env.py (vectorized env wrapper)
- adaptation/tta.py (test-time adaptation utilities)

Usage:
  python main.py train --config basic --outer-steps 100 [--improve attention nas bn recursive]
  python main.py eval --checkpoint checkpoints/latest.pt --episodes 20 [--improve ...]
  python main.py adapt --checkpoint checkpoints/latest.pt --adapt-steps 10 [--improve ...]
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import warnings
from typing import Any, Dict, Optional

# ----------------------
# Pre-run system checks
# ----------------------
MIN_PY = (3, 9)

def _check_prereqs() -> None:
    if sys.version_info < MIN_PY:
        warnings.warn(
            f"Python {MIN_PY[0]}.{MIN_PY[1]}+ recommended. Current: {sys.version.split()[0]}."
        )
    try:
        import torch  # noqa: F401
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    except Exception:
        warnings.warn(
            "PyTorch not found. Install: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )

_check_prereqs()

import torch

# ----------------------
# Repo imports (PyTorch versions)
# ----------------------
try:
    from models.ssm import SSMPolicy  # nn.Module implementing policy with forward/act
except Exception as e:
    raise ImportError("models/ssm.py must define SSMPolicy(nn.Module)") from e

try:
    from meta.maml import MetaMAML  # meta-learner with train_step/eval_on_tasks
except Exception as e:
    raise ImportError("meta/maml.py must expose MetaMAML") from e

try:
    from envs.vector_env import make_vector_env  # returns vectorized env compatible with torch
except Exception as e:
    raise ImportError("envs/vector_env.py must expose make_vector_env") from e

try:
    from adaptation.tta import TTAdapter  # test-time adapter for online/episodic adaptation
except Exception as e:
    raise ImportError("adaptation/tta.py must expose TTAdapter") from e

# Optional improvements registry
VALID_IMPROVE_TAGS = {"attention", "nas", "bn", "recursive"}

def apply_improvements(policy: SSMPolicy, tags: Optional[list[str]]) -> SSMPolicy:
    if not tags:
        return policy
    for tag in tags:
        if tag not in VALID_IMPROVE_TAGS:
            warnings.warn(f"Unknown improvement tag: {tag}")
            continue
        # Hook points for optional modules; assume policy exposes enable_* methods
        if tag == "attention" and hasattr(policy, "enable_attention"):
            policy.enable_attention()
        elif tag == "bn" and hasattr(policy, "enable_batchnorm"):
            policy.enable_batchnorm()
        elif tag == "nas" and hasattr(policy, "apply_nas"):
            policy.apply_nas()
        elif tag == "recursive" and hasattr(policy, "enable_recursive"):
            policy.enable_recursive()
    return policy

# ----------------------
# Config helpers (dict-based)
# ----------------------
EXAMPLE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "basic": {
        "env_id": "CartPole-v1",
        "num_envs": 8,
        "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        "seed": 42,
        "inner_steps": 5,
        "outer_steps": 100,
        "tasks_per_batch": 4,
        "time_limit": 500,
        "lr": 3e-4,
        "ckpt_dir": "checkpoints",
    },
}

# ----------------------
# Checkpoint utils (torch-only)
# ----------------------

def save_checkpoint(path: str, policy: SSMPolicy, extra: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"policy": policy.state_dict(), "meta": extra or {}, "ts": time.time()}
    torch.save(payload, path)


def load_checkpoint(path: str, device: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "policy" not in payload:
        raise RuntimeError("Invalid checkpoint format: missing 'policy'")
    return payload

# ----------------------
# Core workflows
# ----------------------

def build_policy(obs_space: Any, act_space: Any, device: str, cfg: Dict[str, Any]) -> SSMPolicy:
    policy = SSMPolicy(obs_space, act_space, lr=cfg.get("lr", 3e-4), device=device)
    return policy.to(device)


def run_train(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    device = cfg["device"]
    envs = make_vector_env(cfg["env_id"], num_envs=cfg["num_envs"], seed=args.seed or cfg.get("seed", 0), device=device)
    obs_space, act_space = envs.observation_space, envs.action_space

    policy = build_policy(obs_space, act_space, device, cfg)
    policy = apply_improvements(policy, args.improve)

    meta = MetaMAML(policy=policy, inner_steps=cfg["inner_steps"], tasks_per_batch=args.tasks_per_batch or cfg["tasks_per_batch"], device=device)

    outer_steps = args.outer_steps or cfg["outer_steps"]
    best_ret = -float("inf")
    ckpt_dir = args.ckpt_dir or cfg.get("ckpt_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    for step in range(outer_steps):
        metrics = meta.train_step(envs)
        mean_ret = float(metrics.get("mean_return", 0.0))
        if mean_ret > best_ret:
            best_ret = mean_ret
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), policy, {"step": step, "metrics": metrics})
        if (step + 1) % 10 == 0:
            save_checkpoint(os.path.join(ckpt_dir, "latest.pt"), policy, {"step": step, "metrics": metrics})
        print(f"[Train] step={step} metrics={metrics}")

    # final save
    save_checkpoint(os.path.join(ckpt_dir, "final.pt"), policy, {"step": outer_steps})
    return {"best_return": best_ret, "ckpt_dir": ckpt_dir}


def run_eval(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    device = cfg["device"]
    envs = make_vector_env(cfg["env_id"], num_envs=cfg["num_envs"], seed=cfg.get("seed", 0), device=device, eval_mode=True)
    obs_space, act_space = envs.observation_space, envs.action_space

    policy = build_policy(obs_space, act_space, device, cfg)
    checkpoint = load_checkpoint(args.checkpoint, device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()

    returns = []
    episodes = int(args.episodes or 20)
    with torch.no_grad():
        obs = envs.reset()
        ep_ret = torch.zeros(cfg["num_envs"], device=device, dtype=torch.float32)
        for _ in range(cfg.get("time_limit", 500)):
            actions = policy.act(obs)
            obs, rew, done, _ = envs.step(actions)
            ep_ret += rew
            if done.all():
                returns.extend(ep_ret.detach().cpu().tolist())
                obs = envs.reset()
                ep_ret.zero_()
                if len(returns) >= episodes:
                    break
    metrics = {"mean_return": float(torch.tensor(returns).mean().item()) if len(returns) else 0.0,
               "std_return": float(torch.tensor(returns).std(unbiased=False).item()) if len(returns) else 0.0}
    print(f"[Eval] Results: {metrics}")
    return metrics


def run_adapt(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    device = cfg["device"]
    envs = make_vector_env(cfg["env_id"], num_envs=cfg["num_envs"], seed=cfg.get("seed", 0), device=device)
    obs_space, act_space = envs.observation_space, envs.action_space

    policy = build_policy(obs_space, act_space, device, cfg)
    checkpoint = load_checkpoint(args.checkpoint, device)
    policy.load_state_dict(checkpoint["policy"])
    policy = apply_improvements(policy, args.improve)
    policy.train()

    adapter = TTAdapter(policy=policy, steps=int(args.adapt_steps or 10), device=device)

    returns = []
    obs = envs.reset()
    ep_params = None
    for _ep in range(int(args.episodes or 20)):
        if adapter.steps > 0:
            ep_params = adapter.adapt(policy, envs)
            if ep_params is not None:
                policy.load_state_dict(ep_params, strict=False)
        ep_ret = torch.zeros(cfg["num_envs"], device=device, dtype=torch.float32)
        for _ in range(cfg.get("time_limit", 500)):
            actions = policy.act(obs)
            obs, rew, done, _ = envs.step(actions)
            ep_ret += rew
            if getattr(args, "online", False):
                ep_params = adapter.online_step(policy, obs, rew)
                if ep_params is not None:
                    policy.load_state_dict(ep_params, strict=False)
            if done.all():
                break
        returns.extend(ep_ret.detach().cpu().tolist())
    t = torch.tensor(returns)
    metrics = {"mean_return": float(t.mean().item()) if t.numel() else 0.0,
               "std_return": float(t.std(unbiased=False).item()) if t.numel() else 0.0}
    print(f"[Adapt] Results: {metrics}")
    return metrics

# ----------------------
# CLI
# ----------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SSM-MetaRL-TestCompute (PyTorch): Unified pipeline for SSM-based Meta-RL with test-time adaptation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py train --config basic --outer-steps 100 --improve attention nas bn recursive\n"
            "  python main.py eval --checkpoint checkpoints/latest.pt --episodes 20 --improve attention\n"
            "  python main.py adapt --checkpoint checkpoints/latest.pt --episodes 20 --adapt-steps 10 --improve bn\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode", required=True)

    def add_improve_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--improve", nargs='*', default=None, metavar='TAG',
            help=("Optional improvement tags to apply: " + ", ".join(sorted(VALID_IMPROVE_TAGS)))
        )

    train_parser = subparsers.add_parser("train", help="Train meta-initialization via meta-learning")
    train_parser.add_argument("--config", default="basic", choices=list(EXAMPLE_CONFIGS.keys()), help="Experiment config")
    train_parser.add_argument("--seed", type=int, help="Random seed")
    train_parser.add_argument("--outer-steps", type=int, help="Number of meta-learning outer steps")
    train_parser.add_argument("--tasks-per-batch", type=int, help="Tasks per meta-batch")
    train_parser.add_argument("--ckpt-dir", help="Checkpoint directory")
    add_improve_args(train_parser)

    eval_parser = subparsers.add_parser("eval", help="Evaluate policy without adaptation")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    eval_parser.add_argument("--config", default="basic", choices=list(EXAMPLE_CONFIGS.keys()), help="Experiment config")
    eval_parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    add_improve_args(eval_parser)

    adapt_parser = subparsers.add_parser("adapt", help="Evaluate with test-time adaptation")
    adapt_parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    adapt_parser.add_argument("--config", default="basic", choices=list(EXAMPLE_CONFIGS.keys()), help="Experiment config")
    adapt_parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    adapt_parser.add_argument("--adapt-steps", type=int, default=10, help="Adaptation steps per episode")
    adapt_parser.add_argument("--online", action="store_true", help="Enable online adaptation during rollout")
    add_improve_args(adapt_parser)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = dict(EXAMPLE_CONFIGS[args.config])
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.tasks_per_batch is not None:
        cfg["tasks_per_batch"] = args.tasks_per_batch
    if args.outer_steps is not None:
        cfg["outer_steps"] = args.outer_steps
    if getattr(args, "ckpt_dir", None):
        cfg["ckpt_dir"] = args.ckpt_dir

    os.makedirs(cfg.get("ckpt_dir", "checkpoints"), exist_ok=True)

    if args.mode == "train":
        run_train(cfg, args)
    elif args.mode == "eval":
        run_eval(cfg, args)
    elif args.mode == "adapt":
        run_adapt(cfg, args)
    else:
        parser.error(f"Unknown mode: {args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
