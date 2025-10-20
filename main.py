# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
SSM-MetaRL Entry Point - 100% aligned with core/ssm.py, meta_maml.py, test_time_adaptation.py.
"""
import argparse
import torch
import torch.nn as nn
from typing import List, Tuple, Dict

from core.ssm import SSM
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter, AdaptationConfig


def create_dummy_task(state_dim: int = 4, num_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create dummy task data: (support_x, support_y, query_x, query_y).
    """
    support_x = torch.randn(num_samples, state_dim)
    support_y = torch.randn(num_samples, 1)
    query_x = torch.randn(num_samples, state_dim)
    query_y = torch.randn(num_samples, 1)
    return support_x, support_y, query_x, query_y


def train(args):
    """
    Meta-training workflow using MetaMAML.
    
    API from meta_maml.py:
    - MetaMAML(model, inner_lr, outer_lr, first_order=False)
    - adapt(support_x, support_y, loss_fn, num_steps) -> adapted_model
    - meta_update(tasks: List[Tuple[support_x, support_y, query_x, query_y]], loss_fn)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create SSM: SSM(state_dim, hidden_dim=128, output_dim=32, device='cpu')
    model = SSM(state_dim=args.state_dim, hidden_dim=128, output_dim=1, device=device)
    
    # Create MetaMAML: MetaMAML(model, inner_lr, outer_lr, first_order=False)
    meta_learner = MetaMAML(model, inner_lr=args.inner_lr, outer_lr=args.outer_lr, first_order=False)
    
    loss_fn = nn.MSELoss()
    
    print(f"Training for {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        # Create batch of tasks for meta_update
        tasks = []
        for _ in range(args.batch_size):
            task = create_dummy_task(args.state_dim)
            tasks.append(task)
        
        # meta_update(tasks: List[Tuple[support_x, support_y, query_x, query_y]], loss_fn)
        meta_loss = meta_learner.meta_update(tasks, loss_fn)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Meta-loss: {meta_loss:.4f}")
    
    print("Training completed.")
    return model


def test_adaptation(args, model):
    """
    Test-time adaptation workflow using Adapter.
    
    API from test_time_adaptation.py:
    - AdaptationConfig(lr, grad_clip_norm, trust_region_eps, ema_decay, entropy_weight, max_steps_per_call)
    - Adapter(model, cfg)
    - Adapter.adapt(loss_fn, batch_dict) -> loss
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create AdaptationConfig with actual fields
    config = AdaptationConfig(
        lr=args.adapt_lr,
        grad_clip_norm=1.0,
        trust_region_eps=0.01,
        ema_decay=0.99,
        entropy_weight=0.01,
        max_steps_per_call=5
    )
    
    # Create Adapter: Adapter(model, cfg)
    adapter = Adapter(model, config)
    
    loss_fn = nn.MSELoss()
    
    print(f"Running adaptation for {args.num_adapt_steps} steps...")
    for step in range(args.num_adapt_steps):
        # Create batch_dict
        states = torch.randn(args.batch_size, args.state_dim).to(device)
        targets = torch.randn(args.batch_size, 1).to(device)
        batch_dict = {'states': states, 'targets': targets}
        
        # Adapter.adapt(loss_fn, batch_dict) -> loss
        loss = adapter.adapt(loss_fn, batch_dict)
        
        if step % 5 == 0:
            print(f"Adaptation step {step}, Loss: {loss:.4f}")
    
    print("Adaptation completed.")


def main():
    parser = argparse.ArgumentParser(description="SSM-MetaRL Training and Adaptation")
    parser.add_argument('--state_dim', type=int, default=4, help='State dimension')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner learning rate for MetaMAML')
    parser.add_argument('--outer_lr', type=float, default=0.001, help='Outer learning rate for MetaMAML')
    parser.add_argument('--adapt_lr', type=float, default=0.01, help='Learning rate for adaptation')
    parser.add_argument('--num_adapt_steps', type=int, default=20, help='Number of adaptation steps')
    
    args = parser.parse_args()
    
    # Train with MetaMAML
    model = train(args)
    
    # Test with Adapter
    test_adaptation(args, model)


if __name__ == "__main__":
    main()
