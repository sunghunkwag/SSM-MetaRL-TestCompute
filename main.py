# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
SSM-MetaRL Entry Point - Refactored for actual module compatibility.

This file correctly initializes SSM, MetaMAML, and Adapter with proper
parameters and method calls matching the actual codebase signatures.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

from core.ssm import SSM
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter


def create_dummy_batch(batch_size: int = 8, state_dim: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create dummy data for minimal workflow demonstration."""
    states = torch.randn(batch_size, state_dim)
    targets = torch.randn(batch_size, 2)  # Assuming output_dim=2
    return states, targets


def train(args):
    """
    Meta-training workflow using MetaMAML with correct signatures.
    
    MetaMAML signature: __init__(model, inner_lr, outer_lr, inner_steps)
    - model: nn.Module (SSM instance)
    - inner_lr: float
    - outer_lr: float
    - inner_steps: int
    """
    print(f"[TRAIN] Starting meta-training with {args.episodes} episodes...")
    
    # Initialize SSM with explicit parameters (CORRECT)
    ssm = SSM(
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    )
    print(f"[TRAIN] Initialized SSM: state_dim={args.state_dim}, hidden_dim={args.hidden_dim}, output_dim={args.output_dim}")
    
    # Initialize MetaMAML with correct parameters (FIXED: no 'adapt_lr', use 'inner_lr' and 'outer_lr')
    meta_maml = MetaMAML(
        model=ssm,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        inner_steps=args.inner_steps
    )
    print(f"[TRAIN] Initialized MetaMAML: inner_lr={args.inner_lr}, outer_lr={args.outer_lr}, inner_steps={args.inner_steps}")
    
    # Minimal training loop with dummy data
    # Note: MetaMAML doesn't have a 'train' method, so we implement minimal logic
    optimizer = optim.Adam(meta_maml.model.parameters(), lr=args.outer_lr)
    criterion = nn.MSELoss()
    
    for episode in range(args.episodes):
        # Create support and query sets (simplified MAML workflow)
        support_states, support_targets = create_dummy_batch(args.batch_size, args.state_dim)
        query_states, query_targets = create_dummy_batch(args.batch_size, args.state_dim)
        
        # Inner loop: adapt on support set
        adapted_params = meta_maml.inner_loop(
            support_states,
            support_targets,
            criterion
        )
        
        # Outer loop: evaluate on query set with adapted params
        optimizer.zero_grad()
        meta_maml.model.load_state_dict(adapted_params)
        query_pred = meta_maml.model(query_states)
        loss = criterion(query_pred, query_targets)
        loss.backward()
        optimizer.step()
        
        if (episode + 1) % 10 == 0:
            print(f"[TRAIN] Episode {episode + 1}/{args.episodes}, Loss: {loss.item():.4f}")
    
    # Save checkpoint if specified
    if args.checkpoint:
        torch.save({
            'model_state_dict': meta_maml.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args)
        }, args.checkpoint)
        print(f"[TRAIN] Checkpoint saved to {args.checkpoint}")
    
    print("[TRAIN] Meta-training completed.")


def evaluate(args):
    """
    Evaluation workflow using trained SSM model.
    
    Note: Neither MetaMAML nor Adapter have an 'evaluate' method,
    so we implement basic evaluation logic directly.
    """
    print(f"[EVAL] Starting evaluation with {args.eval_episodes} episodes...")
    
    # Load checkpoint
    if not args.checkpoint:
        raise ValueError("Checkpoint path required for evaluation")
    
    checkpoint = torch.load(args.checkpoint)
    saved_args = checkpoint.get('args', {})
    
    # Reconstruct SSM with saved dimensions
    ssm = SSM(
        state_dim=saved_args.get('state_dim', args.state_dim),
        hidden_dim=saved_args.get('hidden_dim', args.hidden_dim),
        output_dim=saved_args.get('output_dim', args.output_dim)
    )
    ssm.load_state_dict(checkpoint['model_state_dict'])
    ssm.eval()
    print(f"[EVAL] Loaded model from {args.checkpoint}")
    
    # Evaluation loop with dummy data
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for episode in range(args.eval_episodes):
            states, targets = create_dummy_batch(args.batch_size, saved_args.get('state_dim', args.state_dim))
            predictions = ssm(states)
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            
            if (episode + 1) % 5 == 0:
                print(f"[EVAL] Episode {episode + 1}/{args.eval_episodes}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / args.eval_episodes
    print(f"[EVAL] Evaluation completed. Average Loss: {avg_loss:.4f}")


def adapt(args):
    """
    Test-time adaptation workflow using Adapter.
    
    Adapter signature: __init__(target: nn.Module, lr: float = 0.001, steps: int = 10)
    Adapter.adapt signature: adapt(loss_fn, batch, fwd_fn=None)
    """
    print(f"[ADAPT] Starting test-time adaptation with {args.adapt_steps} steps...")
    
    # Load checkpoint
    if not args.checkpoint:
        raise ValueError("Checkpoint path required for adaptation")
    
    checkpoint = torch.load(args.checkpoint)
    saved_args = checkpoint.get('args', {})
    
    # Reconstruct SSM (this is the target nn.Module for Adapter)
    ssm = SSM(
        state_dim=saved_args.get('state_dim', args.state_dim),
        hidden_dim=saved_args.get('hidden_dim', args.hidden_dim),
        output_dim=saved_args.get('output_dim', args.output_dim)
    )
    ssm.load_state_dict(checkpoint['model_state_dict'])
    print(f"[ADAPT] Loaded model from {args.checkpoint}")
    
    # Initialize Adapter with CORRECT signature (target: nn.Module)
    adapter = Adapter(
        target=ssm,  # CORRECT: passing nn.Module, not MetaMAML
        lr=args.adapt_lr,
        steps=args.adapt_steps
    )
    print(f"[ADAPT] Initialized Adapter: lr={args.adapt_lr}, steps={args.adapt_steps}")
    
    # Test-time adaptation with dummy batch
    states, targets = create_dummy_batch(args.batch_size, saved_args.get('state_dim', args.state_dim))
    criterion = nn.MSELoss()
    
    # Define forward function for Adapter
    def forward_fn(model, batch):
        states, targets = batch
        return model(states)
    
    # Perform adaptation (CORRECT: using actual Adapter.adapt signature)
    print(f"[ADAPT] Running adaptation on batch...")
    adapted_state = adapter.adapt(
        loss_fn=criterion,
        batch=(states, targets),
        fwd_fn=forward_fn
    )
    
    # Evaluate adapted model
    ssm.load_state_dict(adapted_state)
    ssm.eval()
    
    with torch.no_grad():
        test_states, test_targets = create_dummy_batch(args.batch_size, saved_args.get('state_dim', args.state_dim))
        predictions = ssm(test_states)
        final_loss = criterion(predictions, test_targets)
    
    print(f"[ADAPT] Adaptation completed. Final Loss: {final_loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="SSM-MetaRL: Meta-learning with State-Space Models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operating mode')
    
    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Meta-training mode')
    train_parser.add_argument('--state-dim', type=int, default=4, help='State dimension')
    train_parser.add_argument('--hidden-dim', type=int, default=16, help='Hidden dimension')
    train_parser.add_argument('--output-dim', type=int, default=2, help='Output dimension')
    train_parser.add_argument('--inner-lr', type=float, default=0.01, help='Inner loop learning rate')
    train_parser.add_argument('--outer-lr', type=float, default=0.001, help='Outer loop learning rate')
    train_parser.add_argument('--inner-steps', type=int, default=5, help='Inner loop adaptation steps')
    train_parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    train_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    train_parser.add_argument('--checkpoint', type=str, default=None, help='Path to save checkpoint')
    
    # Evaluate subcommand
    eval_parser = subparsers.add_parser('eval', help='Evaluation mode')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    eval_parser.add_argument('--eval-episodes', type=int, default=20, help='Number of evaluation episodes')
    eval_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    eval_parser.add_argument('--state-dim', type=int, default=4, help='State dimension (fallback)')
    eval_parser.add_argument('--hidden-dim', type=int, default=16, help='Hidden dimension (fallback)')
    eval_parser.add_argument('--output-dim', type=int, default=2, help='Output dimension (fallback)')
    
    # Adapt subcommand
    adapt_parser = subparsers.add_parser('adapt', help='Test-time adaptation mode')
    adapt_parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    adapt_parser.add_argument('--adapt-lr', type=float, default=0.001, help='Adaptation learning rate')
    adapt_parser.add_argument('--adapt-steps', type=int, default=10, help='Number of adaptation steps')
    adapt_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    adapt_parser.add_argument('--state-dim', type=int, default=4, help='State dimension (fallback)')
    adapt_parser.add_argument('--hidden-dim', type=int, default=16, help='Hidden dimension (fallback)')
    adapt_parser.add_argument('--output-dim', type=int, default=2, help='Output dimension (fallback)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)
    elif args.mode == 'adapt':
        adapt(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
