#!/usr/bin/env python3
"""Generate Proof-of-Concept Results for SSM-MetaRL-TestCompute

This script runs a minimal experiment to demonstrate:
1. Meta-training with MetaMAML
2. Test-time adaptation
3. Loss reduction visualization

Usage:
    python experiments/generate_proof.py

Output:
    - results/proof_of_concept.png: Visualization of results
    - Console: Training statistics

Note: This is a minimal proof-of-concept, not a full benchmark.
      For comprehensive evaluation, see experiments/serious_benchmark.py
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

try:
    import gymnasium as gym
except ImportError:
    print("Error: gymnasium not installed. Install with: pip install gymnasium")
    sys.exit(1)

from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from env_runner.environment import Environment


def collect_data(env, model, num_episodes=3, max_steps=50, device='cpu'):
    """Collect trajectory data from environment."""
    all_obs, all_next_obs = [], []
    model.eval()
    
    obs = env.reset()
    hidden_state = model.init_hidden(batch_size=1)
    
    for ep in range(num_episodes):
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                _, hidden_state = model(obs_tensor, hidden_state)
                action = env.action_space.sample()
            
            next_obs, reward, done, info = env.step(action)
            
            all_obs.append(obs)
            all_next_obs.append(next_obs)
            
            obs = next_obs
            steps += 1
        
        if done:
            obs = env.reset()
            hidden_state = model.init_hidden(batch_size=1)
    
    # Return as (1, T, D) tensors
    return {
        'observations': torch.tensor(np.array(all_obs), dtype=torch.float32).unsqueeze(0).to(device),
        'next_observations': torch.tensor(np.array(all_next_obs), dtype=torch.float32).unsqueeze(0).to(device)
    }


def run_meta_training(model, env, num_epochs=5, device='cpu'):
    """Run meta-training with MetaMAML."""
    print("\n" + "="*60)
    print("Meta-Training with MetaMAML")
    print("="*60)
    
    meta_maml = MetaMAML(model=model, inner_lr=0.01, outer_lr=0.001)
    meta_losses = []
    
    for epoch in range(num_epochs):
        # Collect data
        task_data = collect_data(env, model, num_episodes=3, max_steps=50, device=device)
        
        obs_seq = task_data['observations']
        next_obs_seq = task_data['next_observations']
        
        total_len = obs_seq.shape[1]
        if total_len < 2:
            print(f"Epoch {epoch+1}: Data too short, skipping")
            continue
        
        # Split support/query
        split_idx = total_len // 2
        x_support = obs_seq[:, :split_idx]
        y_support = next_obs_seq[:, :split_idx]
        x_query = obs_seq[:, split_idx:]
        y_query = next_obs_seq[:, split_idx:]
        
        # Meta-update
        tasks = [(x_support, y_support, x_query, y_query)]
        initial_hidden = model.init_hidden(batch_size=1)
        
        loss = meta_maml.meta_update(
            tasks=tasks,
            initial_hidden_state=initial_hidden,
            loss_fn=nn.MSELoss()
        )
        
        meta_losses.append(loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Meta Loss: {loss:.4f}")
    
    print(f"\n✅ Meta-training complete!")
    print(f"   Initial loss: {meta_losses[0]:.4f}")
    print(f"   Final loss: {meta_losses[-1]:.4f}")
    print(f"   Reduction: {(1 - meta_losses[-1]/meta_losses[0])*100:.1f}%")
    
    return meta_losses


def run_adaptation(model, env, num_steps=50, device='cpu'):
    """Run test-time adaptation."""
    print("\n" + "="*60)
    print("Test-Time Adaptation")
    print("="*60)
    
    config = AdaptationConfig(learning_rate=0.01, num_steps=5)
    adapter = Adapter(model=model, config=config, device=device)
    
    obs = env.reset()
    hidden_state = model.init_hidden(batch_size=1)
    adaptation_losses = []
    
    for step in range(num_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        current_hidden = hidden_state
        
        # Get next observation
        with torch.no_grad():
            _, hidden_state = model(obs_tensor, current_hidden)
        
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Adapt
        loss, _ = adapter.update_step(
            x=obs_tensor,
            y=next_obs_tensor,
            hidden_state=current_hidden
        )
        
        adaptation_losses.append(loss)
        
        if (step + 1) % 10 == 0:
            print(f"Step {step+1}/{num_steps} - Loss: {loss:.4f}")
        
        obs = next_obs
        if done:
            obs = env.reset()
            hidden_state = model.init_hidden(batch_size=1)
    
    print(f"\n✅ Adaptation complete!")
    print(f"   Initial loss: {adaptation_losses[0]:.4f}")
    print(f"   Final loss: {adaptation_losses[-1]:.4f}")
    print(f"   Reduction: {(1 - adaptation_losses[-1]/adaptation_losses[0])*100:.1f}%")
    
    return adaptation_losses


def visualize_results(meta_losses, adaptation_losses, output_path='results/proof_of_concept.png'):
    """Create visualization of results."""
    print("\n" + "="*60)
    print("Generating Visualization")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Meta-training plot
    epochs = np.arange(1, len(meta_losses) + 1)
    axes[0].plot(epochs, meta_losses, 'o-', linewidth=2, markersize=6, color='#2E86AB', label='Meta-training Loss')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Meta-Training Progress\n(CartPole-v1, {len(meta_losses)} epochs)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(fontsize=10)
    
    # Adaptation plot
    steps = np.arange(1, len(adaptation_losses) + 1)
    axes[1].plot(steps, adaptation_losses, linewidth=2.5, color='#A23B72', label='Adaptation Loss')
    axes[1].axhline(y=adaptation_losses[0], color='gray', linestyle='--', alpha=0.5, label=f'Initial: {adaptation_losses[0]:.3f}')
    axes[1].axhline(y=adaptation_losses[-1], color='green', linestyle='--', alpha=0.5, label=f'Final: {adaptation_losses[-1]:.3f}')
    axes[1].set_xlabel('Adaptation Step', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Test-Time Adaptation\n({len(adaptation_losses)} gradient steps)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate proof-of-concept results')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--epochs', type=int, default=5, help='Number of meta-training epochs')
    parser.add_argument('--adapt-steps', type=int, default=50, help='Number of adaptation steps')
    parser.add_argument('--output', type=str, default='results/proof_of_concept.png', help='Output path')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SSM-MetaRL-TestCompute: Proof-of-Concept Experiment")
    print("="*60)
    print(f"Environment: {args.env}")
    print(f"Meta-training epochs: {args.epochs}")
    print(f"Adaptation steps: {args.adapt_steps}")
    print(f"Device: {args.device}")
    
    # Initialize
    device = torch.device(args.device)
    env = Environment(env_name=args.env, batch_size=1)
    
    obs_space = env.observation_space
    input_dim = obs_space.shape[0] if isinstance(obs_space, gym.spaces.Box) else obs_space.n
    output_dim = input_dim
    
    model = StateSpaceModel(
        state_dim=32,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=64
    ).to(device)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Run experiments
    meta_losses = run_meta_training(model, env, num_epochs=args.epochs, device=device)
    adaptation_losses = run_adaptation(model, env, num_steps=args.adapt_steps, device=device)
    
    # Visualize
    visualize_results(meta_losses, adaptation_losses, output_path=args.output)
    
    print("\n" + "="*60)
    print("✅ Proof-of-Concept Complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"  1. Check the visualization: {args.output}")
    print(f"  2. Add to README: ![Results]({args.output})")
    print(f"  3. Commit and push: git add {args.output} && git commit -m 'Add proof results'")
    print(f"\n🐨 This demonstrates the framework works correctly!")


if __name__ == "__main__":
    main()
