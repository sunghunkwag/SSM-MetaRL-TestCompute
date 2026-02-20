# -*- coding: utf-8 -*-
"""Main training and adaptation script for SSM-MetaRL-TestCompute.

Supports multiple SSM architectures (Legacy, Mamba) and meta-learning
algorithms (MAML, RL², PEARL) with optional Wandb integration.

Usage:
    # Train with Mamba + MAML on CartPole
    python main.py --model_type mamba --env_name CartPole-v1

    # Train with Legacy SSM + MAML
    python main.py --model_type legacy --env_name CartPole-v1

    # Train with Mamba + RL² on CartPole
    python main.py --model_type mamba --meta_alg rl2 --env_name CartPole-v1

    # Full options
    python main.py --model_type mamba --meta_alg maml --env_name CartPole-v1 \\
                   --state_dim 16 --hidden_dim 64 --num_epochs 50 \\
                   --save_dir checkpoints/
"""
import argparse
import logging
import time
import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import gymnasium as gym

from core.ssm import StateSpaceModel  # Legacy
from core.ssm_mamba import MambaSSM   # New
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from env_runner.environment import Environment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional Wandb import
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def collect_data(env, policy_model, num_episodes=10, max_steps_per_episode=100, device='cpu'):
    """Collect trajectory data from environment using the policy model.

    Runs the policy in the environment for multiple episodes, collecting
    observations, actions, rewards, and next observations.

    Args:
        env: Environment instance with reset() and step() methods
        policy_model: SSM model used as policy (output interpreted as action logits)
        num_episodes: Number of episodes to collect
        max_steps_per_episode: Maximum steps per episode before truncation
        device: PyTorch device for tensor placement

    Returns:
        Dictionary with keys:
            - 'observations': Tensor (1, T, obs_dim)
            - 'actions': Tensor (1, T)
            - 'rewards': Tensor (1, T, 1)
            - 'next_observations': Tensor (1, T, obs_dim)
    """
    all_obs, all_actions, all_rewards, all_next_obs, all_dones = [], [], [], [], []
    policy_model.eval()

    obs = env.reset()
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
                    n_actions = env.action_space.n
                    probs = torch.softmax(action_logits[:, :n_actions], dim=-1)
                    action = torch.multinomial(probs, 1).item()
                else:
                    action = action_logits.cpu().numpy().flatten()

            next_obs, reward, done, info = env.step(action)

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
        obs = env.reset()
        hidden_state = policy_model.init_hidden(batch_size=env.batch_size)

    # Return as single sequence (Batch=1, Time=T, Dim=D)
    return {
        'observations': torch.tensor(np.array(all_obs), dtype=torch.float32).unsqueeze(0).to(device),
        'actions': torch.tensor(np.array(all_actions), dtype=torch.long).unsqueeze(0).to(device),
        'rewards': torch.tensor(np.array(all_rewards), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device),
        'next_observations': torch.tensor(np.array(all_next_obs), dtype=torch.float32).unsqueeze(0).to(device)
    }


def train_meta_maml(args, model, env, device, wandb_run=None):
    """Meta-training with MetaMAML.

    Performs MAML meta-learning: collects data, splits into support/query sets,
    runs inner-loop adaptation and outer-loop optimization.

    Args:
        args: Parsed command-line arguments
        model: SSM model (Legacy or Mamba)
        env: Environment instance
        device: PyTorch device
        wandb_run: Optional Wandb run for logging
    """
    logger.info("Starting MetaMAML training...")
    meta_learner = MetaMAML(
        model=model,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr
    )

    for epoch in range(args.num_epochs):
        epoch_start = time.time()

        data = collect_data(
            env, model, num_episodes=args.episodes_per_task,
            max_steps_per_episode=100, device=device
        )

        obs_seq = data['observations']
        next_obs_seq = data['next_observations']

        total_len = obs_seq.shape[1]
        if total_len < 2:
            logger.warning("Collected data is too short, skipping epoch.")
            continue

        split_idx = total_len // 2

        # Split into support and query sets (no data leakage)
        x_support = obs_seq[:, :split_idx]
        y_support = next_obs_seq[:, :split_idx]
        x_query = obs_seq[:, split_idx:]
        y_query = next_obs_seq[:, split_idx:]

        # Pass tasks as List[Tuple]
        tasks = [(x_support, y_support, x_query, y_query)]

        # Initial hidden state
        initial_hidden = model.init_hidden(batch_size=1)

        # Meta update
        loss = meta_learner.meta_update(
            tasks,
            initial_hidden_state=initial_hidden,
            loss_fn=nn.MSELoss()
        )

        epoch_time = time.time() - epoch_start

        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}/{args.num_epochs}, "
                f"Meta Loss: {loss:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )

        # Wandb logging
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch,
                'meta_loss': loss,
                'epoch_time': epoch_time,
            })

    logger.info("MetaMAML training completed.")


def train_meta_rl2(args, model, env, device, wandb_run=None):
    """Meta-training with RL² algorithm.

    RL² uses a recurrent policy that learns the learning algorithm itself.
    Adaptation happens through hidden state updates across episodes.

    Args:
        args: Parsed command-line arguments
        model: SSM model (Legacy or Mamba)
        env: Environment instance
        device: PyTorch device
        wandb_run: Optional Wandb run for logging
    """
    try:
        from meta_rl.rl2 import RL2Policy, RL2Trainer
    except ImportError:
        logger.error("RL² not available. Install required dependencies.")
        return

    obs_dim = args.input_dim
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[0]

    policy = RL2Policy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=args.hidden_dim,
        num_layers=2,
        device=device,
    )

    trainer = RL2Trainer(
        policy=policy,
        env_fn=lambda: Environment(env_name=args.env_name, batch_size=1),
        lr=args.outer_lr,
        episodes_per_task=args.episodes_per_task,
        device=device,
    )

    logger.info("Starting RL² training...")
    trainer.train(
        num_iterations=args.num_epochs,
        log_interval=10,
        wandb_run=wandb_run,
    )
    logger.info("RL² training completed.")


def train_meta_pearl(args, model, env, device, wandb_run=None):
    """Meta-training with PEARL algorithm.

    PEARL uses probabilistic context inference for task identification.
    A context encoder maps trajectories to latent context variables,
    and a SAC policy is conditioned on the inferred context.

    Args:
        args: Parsed command-line arguments
        model: SSM model (Legacy or Mamba)
        env: Environment instance
        device: PyTorch device
        wandb_run: Optional Wandb run for logging
    """
    try:
        from meta_rl.pearl import PEARLAgent
    except ImportError:
        logger.error("PEARL not available. Install required dependencies.")
        return

    obs_dim = args.input_dim
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[0]

    agent = PEARLAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=5,
        encoder_hidden_dim=128,
        policy_hidden_dim=args.hidden_dim,
        kl_weight=0.1,
        lr=args.outer_lr,
        device=device,
    )

    logger.info("Starting PEARL training...")
    agent.train_meta(
        env_fn=lambda: Environment(env_name=args.env_name, batch_size=1),
        num_iterations=args.num_epochs,
        episodes_per_task=args.episodes_per_task,
        log_interval=10,
        wandb_run=wandb_run,
    )
    logger.info("PEARL training completed.")


def test_time_adapt(args, model, env, device):
    """Test-time adaptation using Adapter.

    The Adapter uses hidden_state.detach() internally to prevent autograd
    computational graph errors during gradient updates.

    Args:
        args: Parsed command-line arguments
        model: SSM model (Legacy or Mamba)
        env: Environment instance
        device: PyTorch device
    """
    logger.info("Starting test-time adaptation...")

    config = AdaptationConfig(
        learning_rate=args.adapt_lr,
        num_steps=5
    )

    adapter = Adapter(model=model, config=config, device=device)

    obs = env.reset()
    hidden_state = model.init_hidden(batch_size=1)

    for step in range(args.num_adapt_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        current_hidden_state_for_adapt = hidden_state

        with torch.no_grad():
            output, hidden_state = model(obs_tensor, current_hidden_state_for_adapt)

        if isinstance(env.action_space, gym.spaces.Discrete):
            action = env.action_space.sample()
        else:
            action = env.action_space.sample()

        next_obs, reward, done, info = env.step(action)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)

        loss_val, steps_taken = adapter.update_step(
            x=obs_tensor,
            y=next_obs_tensor,
            hidden_state=current_hidden_state_for_adapt
        )

        obs = next_obs

        if done:
            obs = env.reset()
            hidden_state = model.init_hidden(batch_size=1)

        if step % 10 == 0:
            logger.info(
                f"Adaptation step {step}/{args.num_adapt_steps}, "
                f"Loss: {loss_val:.4f}, Steps taken: {steps_taken}"
            )

    logger.info("Adaptation completed.")
    env.close()


def main():
    """Main entry point for training and adaptation."""
    parser = argparse.ArgumentParser(
        description="SSM-MetaRL: Training and Adaptation with Multiple Architectures"
    )

    # Model architecture
    parser.add_argument(
        '--model_type', type=str, default='mamba',
        choices=['legacy', 'mamba', 's4'],
        help='SSM architecture type: legacy (MLP-based), mamba (structured SSM), s4 (future)'
    )

    # Meta-learning algorithm
    parser.add_argument(
        '--meta_alg', type=str, default='maml',
        choices=['maml', 'rl2', 'pearl'],
        help='Meta-learning algorithm: maml, rl2, pearl'
    )

    # Environment
    parser.add_argument(
        '--env_name', type=str, default='CartPole-v1',
        help='Gymnasium environment name'
    )

    # Model hyperparameters
    parser.add_argument('--state_dim', type=int, default=16,
                        help='SSM state dimension (d_state for Mamba)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden layer dimension (d_model for Mamba)')

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of meta-training epochs/iterations')
    parser.add_argument('--episodes_per_task', type=int, default=5,
                        help='Episodes collected per meta-task')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Environment batch size (currently only supports 1)')
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='Inner learning rate for MAML')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='Outer learning rate for meta-optimizer')
    parser.add_argument('--adapt_lr', type=float, default=0.01,
                        help='Learning rate for test-time adaptation')
    parser.add_argument('--num_adapt_steps', type=int, default=50,
                        help='Total number of adaptation steps during test')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')

    # Wandb
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='ssm-metarl-testcompute',
                        help='Wandb project name')

    args = parser.parse_args()

    if args.batch_size != 1:
        logger.warning("This example currently assumes batch_size=1 for simplicity.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize environment
    env = Environment(env_name=args.env_name, batch_size=args.batch_size)
    obs_space = env.observation_space
    action_space = env.action_space

    input_dim = obs_space.shape[0] if isinstance(obs_space, gym.spaces.Box) else obs_space.n
    output_dim = input_dim  # MAML/Adapter target is next_obs

    args.input_dim = input_dim
    args.output_dim = output_dim

    # Initialize model based on model_type
    if args.model_type == 'mamba':
        model = MambaSSM(
            state_dim=args.state_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=args.hidden_dim,
        ).to(device)
    elif args.model_type == 'legacy':
        model = StateSpaceModel(
            state_dim=args.state_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=args.hidden_dim
        ).to(device)
    elif args.model_type == 's4':
        logger.error("S4 model type is not yet implemented. Use 'mamba' or 'legacy'.")
        return
    else:
        logger.error(f"Unknown model type: {args.model_type}")
        return

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print configuration
    print("\n" + "=" * 50)
    print("SSM-MetaRL-TestCompute")
    print("=" * 50)
    print(f"Model:           {args.model_type}")
    print(f"Meta Algorithm:  {args.meta_alg}")
    print(f"Environment:     {args.env_name}")
    print(f"Device:          {device}")
    print(f"Input/Output:    {input_dim}/{output_dim}")
    print(f"State/Hidden:    {args.state_dim}/{args.hidden_dim}")
    print(f"Parameters:      {total_params:,} total, {trainable_params:,} trainable")

    if args.model_type == 'mamba':
        print(f"Complexity:      O(T·d) [Mamba Selective Scan]")
    elif args.model_type == 'legacy':
        print(f"Complexity:      O(T·d²) [MLP-based State Transition]")
    print("=" * 50 + "\n")

    # Initialize Wandb if requested
    wandb_run = None
    if args.use_wandb:
        if _WANDB_AVAILABLE:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"{args.model_type}_{args.meta_alg}_{args.env_name}",
                config=vars(args),
            )
            logger.info(f"Wandb initialized: {wandb_run.name}")
        else:
            logger.warning("Wandb requested but not installed. Skipping.")

    # Meta-training
    start_time = time.time()

    if args.meta_alg == 'maml':
        train_meta_maml(args, model, env, device, wandb_run)
    elif args.meta_alg == 'rl2':
        train_meta_rl2(args, model, env, device, wandb_run)
    elif args.meta_alg == 'pearl':
        train_meta_pearl(args, model, env, device, wandb_run)

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f}s")

    # Test-time adaptation (only for MAML with compatible models)
    if args.meta_alg == 'maml':
        test_time_adapt(args, model, env, device)

    # Save checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(
        args.save_dir,
        f"{args.model_type}_{args.meta_alg}_{args.env_name}.pt"
    )
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")

    # Log final metrics
    if wandb_run is not None:
        wandb_run.log({'total_train_time': train_time})
        wandb.save(save_path)
        wandb_run.finish()

    print("\n" + "=" * 50)
    print("Execution completed successfully")
    print(f"Total training time: {train_time:.2f}s")
    print(f"Checkpoint saved: {save_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
