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

from core.ssm import StateSpaceModel # Assuming SSM forward now returns (out, next_hidden)
from meta_rl.meta_maml import MetaMAML # Assuming MAML handles stateful models
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from env_runner.environment import Environment # Use the environment runner

def collect_data(env, policy_model, num_episodes=10, max_steps_per_episode=100, device='cpu'):
    """Collects simple trajectory data using the environment runner and a dummy policy."""
    all_obs, all_actions, all_rewards, all_next_obs, all_dones = [], [], [], [], []
    policy_model.eval() # Ensure model is in eval mode for inference

    for _ in range(num_episodes):
        obs = env.reset()
        hidden_state = policy_model.init_hidden(batch_size=env.batch_size) # Assuming batch_size=1 here
        done = False
        steps = 0
        while not done and steps < max_steps_per_episode:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dim

            with torch.no_grad():
                 # Use model's output as logits for action probabilities (dummy policy)
                 # Assuming model output dim matches action space size or needs mapping
                 action_logits, next_hidden_state = policy_model(obs_tensor, hidden_state)

                 # Simple discrete action sampling if action space is discrete
                 if isinstance(env.action_space, gym.spaces.Discrete):
                     probs = torch.softmax(action_logits, dim=-1)
                     action = torch.multinomial(probs, 1).item()
                 else: # Simple continuous action (e.g., mean of Gaussian)
                      action = action_logits.cpu().numpy().flatten() # Needs clipping/scaling based on env.action_space

            next_obs, reward, done, info = env.step(action) # Assuming batch_size=1

            all_obs.append(obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_obs.append(next_obs)
            all_dones.append(done)

            obs = next_obs
            hidden_state = next_hidden_state
            steps += 1

    # Convert lists to numpy arrays or tensors
    # This is a very basic data collection, real RL needs proper batching & processing
    return {
        'observations': torch.tensor(np.array(all_obs), dtype=torch.float32).to(device),
        'actions': torch.tensor(np.array(all_actions), dtype=torch.long).to(device), # Assuming discrete actions
        'rewards': torch.tensor(np.array(all_rewards), dtype=torch.float32).to(device),
        'next_observations': torch.tensor(np.array(all_next_obs), dtype=torch.float32).to(device),
        'dones': torch.tensor(np.array(all_dones), dtype=torch.float32).to(device)
    }


def train_meta(args, model, env, device):
    """Train model using MetaMAML with data collected from the environment."""
    meta_maml = MetaMAML(
        model=model,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr
    )
    print(f"Meta-Training with MetaMAML for {args.num_epochs} epochs using '{args.env_name}'...")

    for epoch in range(args.num_epochs):
        # Simulate task sampling by collecting slightly different data (e.g., reset env differently)
        # For simplicity, we just collect data once per epoch here
        data = collect_data(env, model, num_episodes=args.episodes_per_task, device=device)

        # Prepare dummy support/query splits (in real MAML, these come from different tasks)
        # We just split the collected data
        num_samples = data['observations'].shape[0]
        split_idx = num_samples // 2
        support_x = data['observations'][:split_idx]
        support_y = data['rewards'][:split_idx].unsqueeze(-1) # Example target: predict reward
        query_x = data['observations'][split_idx:]
        query_y = data['rewards'][split_idx:].unsqueeze(-1)

        # MAML expects (Batch, InputDim), need to handle sequences if necessary
        # Current MAML implementation assumes batch processing or sequential processing inside
        # Let's assume non-sequential for simplicity here
        batch_size_maml = support_x.shape[0] if support_x.ndim > 1 else 1

        initial_hidden = model.init_hidden(batch_size=batch_size_maml)

        tasks = [(support_x, support_y, query_x, query_y)] # Single task for simplicity

        # Perform meta-update
        # Loss function: predict reward from observation (example)
        meta_loss = meta_maml.meta_update(tasks, initial_hidden_state=initial_hidden, loss_fn=nn.MSELoss())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Meta Loss = {meta_loss:.4f}")

    print("Meta-training completed.")

def test_time_adapt(args, model, env, device):
    """Test model adaptation using Adapter with data from the environment."""
    config = AdaptationConfig(
        lr=args.adapt_lr,
        grad_clip_norm=1.0,
        max_steps_per_call=5
    )
    adapter = Adapter(model, config)
    loss_fn = nn.MSELoss() # Example: adapt to predict rewards better

    print(f"\nRunning Test-Time Adaptation for {args.num_adapt_steps} steps using '{args.env_name}'...")

    obs = env.reset()
    hidden_state = model.init_hidden(batch_size=env.batch_size) # Assuming batch_size=1

    for step in range(args.num_adapt_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dim
        # Dummy target for adaptation (e.g., observed reward)
        # In a real scenario, this target might be computed differently (e.g., TD error)
        # Here, just generate a dummy target for the adaptation step
        dummy_target_reward = torch.randn(1, args.output_dim).to(device) # Match model output dim

        # Prepare batch dict for adapter
        # Need 'x' for model input, 'targets' for loss
        batch_dict = {'x': obs_tensor, 'targets': dummy_target_reward}

        # Define wrapper functions for adapter's update_step
        def fwd_fn(batch):
            # Pass hidden state correctly, adapter needs to handle state if model is stateful
            # For simplicity, assume adapter internally calls model with state if needed,
            # OR pass state management function to adapter.
            # Current adapter doesn't handle state explicitly in update_step, needs modification
            # Let's assume for this example, we only adapt based on the output part.
            output, _ = adapter.target(batch['x'], hidden_state) # Call model with state
            return output # Return only output for loss calculation

        def loss_fn_wrapper(outputs, batch):
            return loss_fn(outputs, batch['targets'])

        # Perform adaptation step
        info = adapter.update_step(loss_fn_wrapper, batch_dict, fwd_fn=fwd_fn)

        # Step the environment to get next observation for the next loop iteration
        # Use the *adapted* model to choose action (though adaptation happens *after* action normally)
        with torch.no_grad():
             action_logits, next_hidden_state = model(obs_tensor, hidden_state) # Use current hidden state
             if isinstance(env.action_space, gym.spaces.Discrete):
                 probs = torch.softmax(action_logits, dim=-1)
                 action = torch.multinomial(probs, 1).item()
             else:
                 action = action_logits.cpu().numpy().flatten()

        next_obs, reward, done, info_env = env.step(action)

        if done:
            print(f"Adaptation step {step}: Episode finished. Resetting env.")
            obs = env.reset()
            hidden_state = model.init_hidden(batch_size=env.batch_size)
        else:
            obs = next_obs
            hidden_state = next_hidden_state # Use the state returned by the model

        if step % 5 == 0:
            loss_val = info.get('loss', float('nan'))
            steps_taken = info.get('steps', 0)
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
    args.input_dim = input_dim # Store for use in test_adaptation
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
    import gymnasium as gym # To access gym.spaces types
    main()
