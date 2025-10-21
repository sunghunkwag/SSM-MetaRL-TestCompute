# -*- coding: utf-8 -*-
"""
Main training and adaptation script for SSM-MetaRL-TestCompute.
Demonstrates meta-learning with MetaMAML and test-time adaptation using env_runner.
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from core.ssm import StateSpaceModel  # Assuming SSM forward now returns (out, next_hidden)
from meta_rl.meta_maml import MetaMAML  # Assuming MAML handles stateful models
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from env_runner.environment import Environment  # Use the environment runner

def collect_data(env, policy_model, num_episodes=10, max_steps_per_episode=100, device='cpu'):
    """
    Collects simple trajectory data using the environment runner and a dummy policy.
    Returns data as time series (episodes, time_steps, features) for MAML.
    """
    all_obs, all_actions, all_rewards, all_next_obs, all_dones = [], [], [], [], []
    policy_model.eval()  # Ensure model is in eval mode for inference
    
    for _ in range(num_episodes):
        obs = env.reset()
        hidden_state = policy_model.init_hidden(batch_size=env.batch_size)  # Assuming batch_size=1 here
        done = False
        steps = 0
        
        while not done and steps < max_steps_per_episode:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dim
            
            with torch.no_grad():
                # Use model's output as logits for action probabilities (dummy policy)
                # Assuming model output dim matches action space size or needs mapping
                action_logits, next_hidden_state = policy_model(obs_tensor, hidden_state)
                
                # Simple discrete action sampling if action space is discrete
                if isinstance(env.action_space, gym.spaces.Discrete):
                    probs = torch.softmax(action_logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                else:  # Simple continuous action (e.g., mean of Gaussian)
                    action = action_logits.cpu().numpy().flatten()  # Needs clipping/scaling based on env.action_space
            
            next_obs, reward, done, info = env.step(action)  # Assuming batch_size=1
            all_obs.append(obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_obs.append(next_obs)
            all_dones.append(done)
            obs = next_obs
            hidden_state = next_hidden_state
            steps += 1
    
    # Return as time series data (no flattening) - shape will be handled by MAML
    return all_obs, all_actions, all_rewards, all_next_obs, all_dones

def train_meta(args, model, env, device):
    """
    Meta-training with MetaMAML.
    Now passes time series input (B, T, D) directly without flattening.
    """
    print("Starting MetaMAML training...")
    meta_learner = MetaMAML(
        model=model,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        device=device
    )
    
    for epoch in range(args.num_epochs):
        # Collect data for meta-task (simple example: same env, random episodes)
        obs_list, actions_list, rewards_list, next_obs_list, dones_list = collect_data(
            env, model, num_episodes=args.episodes_per_task, device=device
        )
        
        # Convert to time series tensors (B, T, D) without flattening
        # For simplicity: treat entire collected trajectory as one sequence
        obs_seq = torch.tensor(np.array(obs_list), dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, obs_dim)
        next_obs_seq = torch.tensor(np.array(next_obs_list), dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, obs_dim)
        rewards_seq = torch.tensor(np.array(rewards_list), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # (1, T, 1)
        
        # Create dummy target (for example, predict next observation or reward)
        # Here we use next_obs as target for simplicity
        x_support = obs_seq  # (B=1, T, D)
        y_support = next_obs_seq  # (B=1, T, D)
        x_query = obs_seq  # Could be different query set
        y_query = next_obs_seq
        
        # MetaMAML adapt and meta_update now receive time series input
        loss = meta_learner.meta_update(
            x_support=x_support,
            y_support=y_support,
            x_query=x_query,
            y_query=y_query
        )
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Meta Loss: {loss:.4f}")
    
    print("MetaMAML training completed.")

def test_time_adapt(args, model, env, device):
    """
    Test-time adaptation using Adapter.
    Now manages hidden_state through fwd_fn at each step.
    """
    print("Starting test-time adaptation...")
    
    # Create adapter config
    config = AdaptationConfig(
        learning_rate=args.adapt_lr,
        num_steps=args.num_adapt_steps
    )
    
    # Initialize hidden state
    hidden_state = model.init_hidden(batch_size=1)
    
    # Define forward function that manages hidden state
    def fwd_fn(x, h):
        """Forward function that takes input and hidden state, returns output and next hidden state."""
        output, next_h = model(x, h)
        return output, next_h
    
    # Create adapter
    adapter = Adapter(model=model, config=config, device=device)
    
    # Run adaptation loop with environment interaction
    obs = env.reset()
    
    for step in range(args.num_adapt_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Use fwd_fn to get output and update hidden state
        output, hidden_state = fwd_fn(obs_tensor, hidden_state)
        
        # Simple dummy action selection (could use output as policy)
        if isinstance(env.action_space, gym.spaces.Discrete):
            action = env.action_space.sample()
        else:
            action = env.action_space.sample()
        
        # Step environment
        next_obs, reward, done, info = env.step(action)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Perform adaptation update_step
        # For test-time adaptation, we typically adapt based on prediction error
        # Here we use a simple loss: predicting next observation
        loss_val, steps_taken = adapter.update_step(
            x=obs_tensor,
            y=next_obs_tensor,
            hidden_state=hidden_state  # Pass hidden state to adapter
        )
        
        obs = next_obs
        
        # Reset if episode ends
        if done:
            obs = env.reset()
            hidden_state = model.init_hidden(batch_size=1)
        
        if step % 10 == 0:
            print(f"Adaptation step {step}, Loss: {loss_val:.4f}, Steps taken: {steps_taken}")
    
    print("Adaptation completed.")
    env.close()

def main():
    parser = argparse.ArgumentParser(description="SSM-MetaRL Training and Adaptation with EnvRunner")
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Gymnasium environment name')
    parser.add_argument('--state_dim', type=int, default=32, help='SSM state dimension')
    # input_dim and output_dim will be determined by the environment
    parser.add_argument('--hidden_dim', type=int, default=64, help='SSM hidden layer dimension')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of meta-training epochs')
    parser.add_argument('--episodes_per_task', type=int, default=5, help='Episodes collected per meta-task')
    parser.add_argument('--batch_size', type=int, default=1, help='Environment batch size (currently only supports 1)')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner learning rate for MetaMAML')
    parser.add_argument('--outer_lr', type=float, default=0.001, help='Outer learning rate for MetaMAML')
    parser.add_argument('--adapt_lr', type=float, default=0.01, help='Learning rate for test-time adaptation')
    parser.add_argument('--num_adapt_steps', type=int, default=50, help='Number of adaptation steps during test')
    args = parser.parse_args()
    
    if args.batch_size != 1:
        print("Warning: This example currently assumes batch_size=1 for simplicity.")
        # Need to adjust hidden state init and data collection for batch_size > 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize environment
    env = Environment(env_name=args.env_name, batch_size=args.batch_size)
    obs_space = env.observation_space
    action_space = env.action_space
    
    # Determine model input and output dims based on env
    input_dim = obs_space.shape[0] if isinstance(obs_space, gym.spaces.Box) else obs_space.n
    output_dim = action_space.n if isinstance(action_space, gym.spaces.Discrete) else action_space.shape[0]
    
    args.input_dim = input_dim  # Store for use in test_adaptation
    args.output_dim = output_dim
    
    # Initialize model
    model = StateSpaceModel(
        state_dim=args.state_dim,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # Meta-Train with MetaMAML using environment data
    train_meta(args, model, env, device)
    
    # Test Time Adaptation using environment data
    test_time_adapt(args, model, env, device)

if __name__ == "__main__":
    import gymnasium as gym  # To access gym.spaces types
    main()
