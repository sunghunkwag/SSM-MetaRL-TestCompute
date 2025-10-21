# -*- coding: utf-8 -*-
"""
Main training and adaptation script for SSM-MetaRL-TestCompute.
Demonstrates meta-learning with MetaMAML and test-time adaptation.
"""

import argparse
import torch
import torch.nn as nn
from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter, AdaptationConfig


def train(args):
    """
    Train model using MetaMAML.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = StateSpaceModel(
        state_dim=args.state_dim,
        input_dim=args.input_dim,  # Use separate input_dim
        output_dim=1
    ).to(device)
    
    # Initialize MetaMAML
    meta_maml = MetaMAML(
        model=model,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr
    )
    
    print(f"Training with MetaMAML for {args.num_epochs} epochs...")
    
    # Simplified meta-training loop
    for epoch in range(args.num_epochs):
        # Create dummy task data
        B, T, D_in = args.batch_size, 10, args.input_dim
        
        # Reshape data to be compatible with SSM (B*T, D_in)
        support_x = torch.randn(B, T, D_in).to(device).reshape(B * T, D_in)
        
        # Create dummy support_y (targets)
        support_y = torch.randn(B * T, 1).to(device) # model output_dim is 1
        
        # MetaMAML adapt returns OrderedDict of fast_weights
        from collections import OrderedDict
        
        # Call adapt_task with correct args
        fast_weights = meta_maml.adapt_task(support_x, support_y, num_steps=5)
        
        # Verify return type
        assert isinstance(fast_weights, OrderedDict), \
            f"Expected OrderedDict from meta_maml.adapt_task, got {type(fast_weights)}"
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Adapted weights obtained (OrderedDict with {len(fast_weights)} parameters)")
    
    print("Meta-training completed.")
    return model


def test_adaptation(args, model):
    """
    Test model adaptation using Adapter.
    Adapter.update_step returns a dict with 'loss', 'steps', etc.
    
    Args:
        args: Command-line arguments
        model: Pre-trained model
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
        states = torch.randn(args.batch_size, args.input_dim).to(device)
        targets = torch.randn(args.batch_size, 1).to(device)
        
        # The model's forward is 'x', not 'states'.
        batch_dict = {'x': states, 'targets': targets}
        
        # Define wrapper functions for fwd and loss
        def fwd_fn(batch):
            return adapter.target(batch['x'])
            
        def loss_fn_wrapper(outputs, batch):
            return loss_fn(outputs, batch['targets'])

        info = adapter.update_step(loss_fn_wrapper, batch_dict, fwd_fn=fwd_fn)
        
        # Verify return type and extract loss from dict
        assert isinstance(info, dict), \
            f"Expected dict from adapter.update_step, got {type(info)}"
        assert 'loss' in info, \
            f"Expected 'loss' key in adapter.update_step result, got keys: {info.keys()}"
        
        # Extract loss value from info dict
        loss = info['loss']
        steps = info.get('steps', 0)
        
        if step % 5 == 0:
            print(f"Adaptation step {step}, Loss: {loss:.4f}, Steps taken: {steps}")
    
    print("Adaptation completed.")


def main():
    parser = argparse.ArgumentParser(description="SSM-MetaRL Training and Adaptation")
    parser.add_argument('--state_dim', type=int, default=4, help='State dimension')
    parser.add_argument('--input_dim', type=int, default=4, help='Input dimension')
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
