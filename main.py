# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
SSM-MetaRL Entry Point - Fully aligned with actual meta_maml.py and test_time_adaptation.py APIs.
All legacy parameters (inner_steps in MetaMAML.__init__, num_steps in Adapter.adapt,
optimizer_type, loss_type in AdaptationConfig) removed.
Only existing methods (adapt, meta_update) used. No inner_loop/outer_loop calls.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Any

from core.ssm import SSM
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter, AdaptationConfig


def create_dummy_batch(batch_size: int = 8, state_dim: int = 4) -> Dict[str, torch.Tensor]:
    """
    Create dummy batch as dict (required by Adapter.adapt and MetaMAML methods).
    """
    states = torch.randn(batch_size, state_dim)
    targets = torch.randn(batch_size, 2)  # Assuming output_dim=2
    return {"states": states, "targets": targets}


def train(args):
    """
    Meta-training workflow using MetaMAML with CORRECT signatures.
    
    Actual API from meta_maml.py:
    - __init__(model, inner_lr, outer_lr, first_order) — NO inner_steps!
    - adapt(support_batch, loss_fn) — batch is dict
    - meta_update(query_batch, loss_fn) — batch is dict
    """
    print(f"[TRAIN] Starting meta-training with {args.episodes} episodes...")
    
    # Initialize SSM
    ssm = SSM(
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        ssm_state_dim=args.ssm_state_dim,
        dt_rank=args.dt_rank,
        d_conv=args.d_conv
    )
    
    # Initialize MetaMAML with ONLY actual parameters (inner_lr, outer_lr, first_order)
    # NO inner_steps parameter!
    meta_learner = MetaMAML(
        model=ssm,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        first_order=args.first_order
    )
    
    # Define simple MSE loss function
    def loss_fn(batch: Dict[str, torch.Tensor], model: nn.Module) -> torch.Tensor:
        states = batch["states"]
        targets = batch["targets"]
        outputs = model(states)
        return nn.functional.mse_loss(outputs, targets)
    
    # Meta-training loop
    for ep in range(args.episodes):
        # Create support and query batches (both as dicts)
        support_batch = create_dummy_batch(args.batch_size, args.state_dim)
        query_batch = create_dummy_batch(args.batch_size, args.state_dim)
        
        # Inner adaptation on support set
        adapted_loss = meta_learner.adapt(support_batch, loss_fn)
        
        # Meta-update on query set
        meta_loss = meta_learner.meta_update(query_batch, loss_fn)
        
        if (ep + 1) % 10 == 0:
            print(f"[TRAIN] Episode {ep+1}/{args.episodes} | "
                  f"Adapted Loss: {adapted_loss:.4f} | Meta Loss: {meta_loss:.4f}")
    
    print("[TRAIN] Meta-training complete.")
    return meta_learner


def test(args, meta_learner):
    """
    Test-time adaptation using Adapter with CORRECT signatures.
    
    Actual API from test_time_adaptation.py:
    - AdaptationConfig(learning_rate, num_steps, temperature) — NO optimizer_type, loss_type!
    - Adapter.adapt(batch, loss_fn) — NO num_steps parameter! batch is dict!
    """
    print(f"\n[TEST] Starting test-time adaptation with {args.adapt_steps} steps...")
    
    # Create AdaptationConfig with ONLY actual parameters
    # NO optimizer_type, NO loss_type!
    adapt_config = AdaptationConfig(
        learning_rate=args.adapt_lr,
        num_steps=args.adapt_steps,
        temperature=args.temperature
    )
    
    # Initialize Adapter with the meta-learned model
    adapter = Adapter(
        model=meta_learner.model,
        config=adapt_config
    )
    
    # Define loss function
    def loss_fn(batch: Dict[str, torch.Tensor], model: nn.Module) -> torch.Tensor:
        states = batch["states"]
        targets = batch["targets"]
        outputs = model(states)
        return nn.functional.mse_loss(outputs, targets)
    
    # Create test batch (as dict)
    test_batch = create_dummy_batch(args.batch_size, args.state_dim)
    
    # Adapt on test batch — NO num_steps argument! Only batch and loss_fn!
    adapted_loss = adapter.adapt(test_batch, loss_fn)
    
    print(f"[TEST] Adaptation complete. Final loss: {adapted_loss:.4f}")
    return adapter


def main():
    parser = argparse.ArgumentParser(
        description="SSM-MetaRL: Fully corrected main.py aligned with actual APIs"
    )
    
    # SSM parameters
    parser.add_argument("--state_dim", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=2)
    parser.add_argument("--ssm_state_dim", type=int, default=16)
    parser.add_argument("--dt_rank", type=int, default=4)
    parser.add_argument("--d_conv", type=int, default=4)
    
    # MetaMAML parameters (NO inner_steps!)
    parser.add_argument("--inner_lr", type=float, default=0.01)
    parser.add_argument("--outer_lr", type=float, default=0.001)
    parser.add_argument("--first_order", type=bool, default=True)
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    
    # Test-time adaptation parameters
    parser.add_argument("--adapt_lr", type=float, default=0.001)
    parser.add_argument("--adapt_steps", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Execute workflow
    print("=" * 60)
    print("SSM-MetaRL Workflow - API-Aligned Version")
    print("=" * 60)
    
    # Meta-training phase
    meta_learner = train(args)
    
    # Test-time adaptation phase
    adapter = test(args, meta_learner)
    
    print("\n" + "=" * 60)
    print("Workflow complete: All APIs aligned with actual codebase.")
    print("=" * 60)


if __name__ == "__main__":
    main()
