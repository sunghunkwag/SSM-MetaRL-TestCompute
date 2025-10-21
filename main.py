# -*- coding: utf-8 -*-
"""
Main training and adaptation script for SSM-MetaRL-TestCompute.
Demonstrates meta-learning with MetaMAML and test-time adaptation using env_runner.
(This is the corrected version fixing MAML API calls and Adapter logic)
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import gymnasium as gym # Import gymnasium

from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from env_runner.environment import Environment

def collect_data(env, policy_model, num_episodes=10, max_steps_per_episode=100, device='cpu'):
    """
    Collects simple trajectory data.
    Returns data as a single long sequence for MAML.
    """
    all_obs, all_actions, all_rewards, all_next_obs, all_dones = [], [], [], [], []
    policy_model.eval()
    
    obs, _ = env.reset() # Gymnasium returns (obs, info)
    hidden_state = policy_model.init_hidden(batch_size=env.batch_size)
    
    total_steps = 0
    for ep in range(num_episodes):
        steps_in_ep = 0
        done = False
        
        while not done and steps_in_ep < max_steps_per_episode:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_logits, next_hidden_state = policy_model(obs_tensor, hidden_state)
                
                if isinstance(env.action_space, gym.spaces.Discrete):
                    probs = torch.softmax(action_logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                else:
                    action = action_logits.cpu().numpy().flatten()
            
            next_obs, reward, done, truncated, info = env.step(action) # Gymnasium returns 5 values
            done = done or truncated # Combine done and truncated
            
            all_obs.append(obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_obs.append(next_obs)
            all_dones.append(done)
            
            obs = next_obs
            hidden_state = next_hidden_state
            steps_in_ep += 1
            total_steps += 1
        
        # Reset at the end of an episode
        obs, _ = env.reset()
        hidden_state = policy_model.init_hidden(batch_size=env.batch_size)
            
    # Return as single sequence (Batch=1, Time=T, Dim=D)
    return {
        'observations': torch.tensor(np.array(all_obs), dtype=torch.float32).unsqueeze(0).to(device),
        'actions': torch.tensor(np.array(all_actions), dtype=torch.long).unsqueeze(0).to(device),
        'rewards': torch.tensor(np.array(all_rewards), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device),
        'next_observations': torch.tensor(np.array(all_next_obs), dtype=torch.float32).unsqueeze(0).to(device)
    }

def train_meta(args, model, env, device):
    """
    Meta-training with MetaMAML.
    FIXED:
    1. Passes tasks as a List[Tuple] (tasks) to meta_update.
    2. Passes initial_hidden_state to meta_update.
    3. Correctly splits data into support/query sets (no data leakage).
    """
    print("Starting MetaMAML training...")
    meta_learner = MetaMAML(
        model=model,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr
        # 'device' is not an arg for MetaMAML init, model is already on device
    )
    
    for epoch in range(args.num_epochs):
        data = collect_data(
            env, model, num_episodes=args.episodes_per_task, 
            max_steps_per_episode=100, device=device
        )
        
        # Data is (1, T, D)
        obs_seq = data['observations']
        # Use next_obs as target (example)
        next_obs_seq = data['next_observations'] 
        
        # Get total sequence length
        total_len = obs_seq.shape[1]
        if total_len < 2:
            print("Warning: Collected data is too short, skipping epoch.")
            continue
            
        split_idx = total_len // 2
        
        # --- FIX 3: No Data Leakage ---
        x_support = obs_seq[:, :split_idx]
        y_support = next_obs_seq[:, :split_idx]
        x_query = obs_seq[:, split_idx:]
        y_query = next_obs_seq[:, split_idx:]
        
        # --- FIX 1: Pass tasks as List[Tuple] ---
        # (support_x, support_y, query_x, query_y)
        tasks = [(x_support, y_support, x_query, y_query)]
        
        # --- FIX 2: Pass initial_hidden_state ---
        # Batch size is 1 (from unsqueeze in collect_data)
        initial_hidden = model.init_hidden(batch_size=1) 
        
        # Correctly call meta_update
        loss = meta_learner.meta_update(
            tasks, 
            initial_hidden_state=initial_hidden,
            loss_fn=nn.MSELoss() # Example loss
        )
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Meta Loss: {loss:.4f}")
    
    print("MetaMAML training completed.")

def test_time_adapt(args, model, env, device):
    """
    Test-time adaptation using Adapter.
    FIXED:
    4. Passes the *correct* hidden state (state_t) to update_step,
       not the future state (state_t+1).
    """
    print("Starting test-time adaptation...")
    
    # Create adapter config
    config = AdaptationConfig(
        learning_rate=args.adapt_lr,
        num_steps=5 # Internal steps per call (was args.num_adapt_steps, which is too large)
    )
    
    # Create adapter
    adapter = Adapter(model=model, config=config, device=device)
    
    # Initialize hidden state
    obs, _ = env.reset()
    hidden_state = model.init_hidden(batch_size=1) # This is state_t
    
    for step in range(args.num_adapt_steps): # Total adaptation steps
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device) # obs_t
        
        # --- FIX 4: Correct State Logic ---
        # 1. Store the *current* state (state_t)
        current_hidden_state_for_adapt = hidden_state 
        
        # 2. Get action and *next* state (state_t+1)
        with torch.no_grad():
            output, hidden_state = model(obs_tensor, current_hidden_state_for_adapt)
        
        if isinstance(env.action_space, gym.spaces.Discrete):
            action = env.action_space.sample() # Dummy action
        else:
            action = env.action_space.sample()
        
        # 3. Step environment
        next_obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device) # target_t+1
        
        # 4. Adapt using (obs_t, target_t+1, state_t)
        #    The adapter will internally run 'num_steps' times
        loss_val, steps_taken = adapter.update_step(
            x=obs_tensor,              # obs_t
            y=next_obs_tensor,         # target_t+1
            hidden_state=current_hidden_state_for_adapt # state_t
        )
        
        obs = next_obs
        
        if done:
            obs, _ = env.reset()
            hidden_state = model.init_hidden(batch_size=1) # Reset state
        
        if step % 10 == 0:
            print(f"Adaptation step {step}, Loss: {loss_val:.4f}, Steps taken: {steps_taken}")
    
    print("Adaptation completed.")
    env.close()

def main():
    parser = argparse.ArgumentParser(description="SSM-MetaRL Training and Adaptation with EnvRunner")
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Gymnasium environment name')
    parser.add_argument('--state_dim', type=int, default=32, help='SSM state dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='SSM hidden layer dimension')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of meta-training epochs')
    parser.add_argument('--episodes_per_task', type=int, default=5, help='Episodes collected per meta-task')
    parser.add_argument('--batch_size', type=int, default=1, help='Environment batch size (currently only supports 1)')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner learning rate for MetaMAML')
    parser.add_argument('--outer_lr', type=float, default=0.001, help='Outer learning rate for MetaMAML')
    parser.add_argument('--adapt_lr', type=float, default=0.01, help='Learning rate for test-time adaptation')
    parser.add_argument('--num_adapt_steps', type=int, default=50, help='Total number of adaptation steps during test')
    args = parser.parse_args()
    
    if args.batch_size != 1:
        print("Warning: This example currently assumes batch_size=1 for simplicity.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize environment
    env = Environment(env_name=args.env_name, batch_size=args.batch_size)
    obs_space = env.observation_space
    action_space = env.action_space
    
    input_dim = obs_space.shape[0] if isinstance(obs_space, gym.spaces.Box) else obs_space.n
    
    # --- FIX: Output dim was based on action space, but example uses 'next_obs' as target ---
    # The MAML/Adapter target is next_obs, so output_dim must match input_dim
    output_dim = input_dim 
    
    args.input_dim = input_dim
    args.output_dim = output_dim
    
    # Initialize model
    model = StateSpaceModel(
        state_dim=args.state_dim,
        input_dim=input_dim,
        output_dim=output_dim, # Must match target (next_obs_tensor)
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # Meta-Train with MetaMAML
    train_meta(args, model, env, device)
    
    # Test Time Adaptation
    test_time_adapt(args, model, env, device)

if __name__ == "__main__":
    main()
