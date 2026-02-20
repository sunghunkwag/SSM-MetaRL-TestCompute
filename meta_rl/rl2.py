"""RL² (Learning to Reinforcement Learn) Implementation

Reference: Duan et al., 2016 - "RL²: Fast Reinforcement Learning via Slow
Reinforcement Learning"

Algorithm Overview:
    RL² learns a recurrent policy that encodes the learning algorithm itself.
    There is no explicit inner loop adaptation; instead, adaptation happens
    implicitly through hidden state updates of a recurrent neural network
    across episodes within a task.

    The key insight is that an RNN trained across many tasks with REINFORCE
    can learn to perform Bayesian-optimal exploration and adaptation purely
    through its hidden state dynamics.

Architecture:
    Input:  (observation_t, action_{t-1}, reward_{t-1}, done_{t-1})
    ↓
    GRU/LSTM layers (hidden state persists across episodes within a task)
    ↓
    Output: action distribution (Categorical for discrete, Gaussian for continuous)

    The hidden state is ONLY reset between different tasks, not between
    episodes of the same task. This allows the RNN to accumulate information
    about the current task across multiple episodes.

Complexity:
    Time: O(T·H²) per timestep where H is hidden_size
    Space: O(H²) for GRU parameters + O(H) for hidden state

Example:
    >>> from meta_rl.rl2 import RL2Policy, RL2Trainer
    >>> policy = RL2Policy(obs_dim=4, action_dim=2, hidden_size=256)
    >>> trainer = RL2Trainer(policy, env_fn, lr=3e-4)
    >>> trainer.train(num_iterations=1000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Callable, Tuple, List, Dict, Any

logger = logging.getLogger(__name__)


class RL2Policy(nn.Module):
    """Recurrent policy for RL² meta-learning.

    The policy takes augmented observations (obs, prev_action, prev_reward,
    prev_done) and processes them through a GRU to produce action distributions.
    The hidden state persists across episodes within a task, allowing the
    policy to learn task-specific adaptation strategies.

    Args:
        obs_dim: Dimension of the observation space
        action_dim: Number of discrete actions or dimension of continuous actions
        hidden_size: Size of the GRU hidden state (default: 256)
        num_layers: Number of stacked GRU layers (default: 2)
        discrete: Whether the action space is discrete (default: True)
        device: Device for tensor placement (default: 'cpu')

    Attributes:
        augmented_dim: obs_dim + action_dim + 1 (reward) + 1 (done)
        gru: Multi-layer GRU for sequence processing
        action_head: Linear layer mapping hidden state to action logits/means
        value_head: Linear layer mapping hidden state to state value
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        discrete: bool = True,
        device: str = 'cpu',
    ):
        super(RL2Policy, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.discrete = discrete
        self.device = device

        # Augmented input: obs + one-hot action + reward + done
        self.augmented_dim = obs_dim + action_dim + 1 + 1

        # Input projection
        self.input_proj = nn.Linear(self.augmented_dim, hidden_size)

        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

        # Value head (for PPO/REINFORCE baseline)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.to(device)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize GRU hidden state to zeros.

        Args:
            batch_size: Number of parallel sequences

        Returns:
            Zero tensor of shape (num_layers, batch_size, hidden_size)
        """
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size,
            device=self.device,
        )

    def forward(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        prev_done: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the RL² policy.

        Constructs augmented input from (obs, prev_action, prev_reward, prev_done),
        processes through GRU, and outputs action logits and value estimate.

        Args:
            obs: Current observation, shape (B, obs_dim) or (B, T, obs_dim)
            prev_action: Previous action as one-hot, shape (B, action_dim) or (B, T, action_dim)
            prev_reward: Previous reward, shape (B, 1) or (B, T, 1)
            prev_done: Previous done flag, shape (B, 1) or (B, T, 1)
            hidden: GRU hidden state, shape (num_layers, B, hidden_size)

        Returns:
            Tuple of:
                - action_logits: Action distribution params (B, action_dim) or (B, T, action_dim)
                - value: State value estimate (B, 1) or (B, T, 1)
                - next_hidden: Updated hidden state (num_layers, B, hidden_size)
        """
        # Construct augmented input
        augmented = torch.cat([obs, prev_action, prev_reward, prev_done], dim=-1)

        # Determine if sequence or single step
        single_step = augmented.dim() == 2
        if single_step:
            augmented = augmented.unsqueeze(1)  # (B, 1, aug_dim)

        # Input projection
        x = F.relu(self.input_proj(augmented))  # (B, T, hidden_size)

        # GRU forward
        gru_out, next_hidden = self.gru(x, hidden)  # (B, T, hidden_size)

        # Action and value heads
        action_logits = self.action_head(gru_out)  # (B, T, action_dim)
        value = self.value_head(gru_out)  # (B, T, 1)

        if single_step:
            action_logits = action_logits.squeeze(1)
            value = value.squeeze(1)

        return action_logits, value, next_hidden

    def get_action(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        prev_done: torch.Tensor,
        hidden: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy.

        Args:
            obs: Current observation (1, obs_dim)
            prev_action: Previous action one-hot (1, action_dim)
            prev_reward: Previous reward (1, 1)
            prev_done: Previous done flag (1, 1)
            hidden: GRU hidden state
            deterministic: If True, take argmax instead of sampling

        Returns:
            Tuple of (action, log_prob, value, next_hidden)
        """
        action_logits, value, next_hidden = self.forward(
            obs, prev_action, prev_reward, prev_done, hidden
        )

        if self.discrete:
            probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob, value, next_hidden
        else:
            # Continuous: treat logits as mean, use fixed std
            mean = action_logits
            std = torch.ones_like(mean) * 0.5
            dist = torch.distributions.Normal(mean, std)
            if deterministic:
                action = mean
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action.cpu().numpy().flatten(), log_prob, value, next_hidden


class RL2Trainer:
    """Trainer for the RL² meta-learning algorithm.

    Implements the outer-loop optimization for RL² using REINFORCE with
    baseline (value function). Tasks are sampled, trajectories are collected
    with persistent hidden states, and the policy is updated.

    Args:
        policy: RL2Policy instance
        env_fn: Callable that creates a new environment instance
        lr: Learning rate for the outer-loop optimizer (default: 3e-4)
        gamma: Discount factor (default: 0.99)
        episodes_per_task: Number of episodes per task for hidden state
                          accumulation (default: 10)
        max_steps_per_episode: Maximum steps per episode (default: 200)
        num_tasks_per_batch: Number of tasks per meta-batch (default: 16)
        entropy_coef: Entropy bonus coefficient (default: 0.01)
        value_coef: Value loss coefficient (default: 0.5)
        device: Device for computation (default: 'cpu')

    Example:
        >>> policy = RL2Policy(obs_dim=4, action_dim=2)
        >>> trainer = RL2Trainer(policy, env_fn=lambda: gym.make('CartPole-v1'))
        >>> trainer.train(num_iterations=500)
    """

    def __init__(
        self,
        policy: RL2Policy,
        env_fn: Callable,
        lr: float = 3e-4,
        gamma: float = 0.99,
        episodes_per_task: int = 10,
        max_steps_per_episode: int = 200,
        num_tasks_per_batch: int = 16,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = 'cpu',
    ):
        self.policy = policy
        self.env_fn = env_fn
        self.lr = lr
        self.gamma = gamma
        self.episodes_per_task = episodes_per_task
        self.max_steps_per_episode = max_steps_per_episode
        self.num_tasks_per_batch = num_tasks_per_batch
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def collect_task_trajectory(self) -> Dict[str, List]:
        """Collect a full trajectory (multiple episodes) for one task.

        The hidden state persists across episodes within the task,
        allowing the policy to learn about the task through experience.

        Returns:
            Dictionary with trajectory data:
                - 'log_probs': List of log probabilities
                - 'values': List of value estimates
                - 'rewards': List of rewards
                - 'entropies': List of action entropies
        """
        import gymnasium as gym

        env = self.env_fn()
        self.policy.eval()

        log_probs = []
        values = []
        rewards = []
        entropies = []

        # Initialize hidden state (persists across episodes)
        hidden = self.policy.init_hidden(batch_size=1)

        # Initialize previous action/reward/done
        prev_action = torch.zeros(1, self.policy.action_dim, device=self.device)
        prev_reward = torch.zeros(1, 1, device=self.device)
        prev_done = torch.zeros(1, 1, device=self.device)

        for episode in range(self.episodes_per_task):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]  # Gymnasium returns (obs, info)
            done = False
            steps = 0

            while not done and steps < self.max_steps_per_episode:
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32
                ).unsqueeze(0).to(self.device)

                # Get action from policy
                action, log_prob, value, hidden = self.policy.get_action(
                    obs_tensor, prev_action, prev_reward, prev_done, hidden
                )

                # Step environment
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, info = step_result

                # Compute entropy for regularization
                with torch.no_grad():
                    action_logits, _, _ = self.policy.forward(
                        obs_tensor, prev_action, prev_reward, prev_done, hidden
                    )
                    probs = F.softmax(action_logits, dim=-1)
                    entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                entropies.append(entropy)

                # Update previous action/reward/done for next step
                prev_action = torch.zeros(1, self.policy.action_dim, device=self.device)
                if self.policy.discrete:
                    prev_action[0, action] = 1.0
                prev_reward = torch.tensor(
                    [[reward]], dtype=torch.float32, device=self.device
                )
                prev_done = torch.tensor(
                    [[float(done)]], dtype=torch.float32, device=self.device
                )

                obs = next_obs
                steps += 1

            # At episode boundary: update prev_done but do NOT reset hidden state
            prev_done = torch.ones(1, 1, device=self.device)

        env.close()
        return {
            'log_probs': log_probs,
            'values': values,
            'rewards': rewards,
            'entropies': entropies,
        }

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """Compute discounted returns from reward sequence.

        Args:
            rewards: List of scalar rewards

        Returns:
            Tensor of discounted returns, shape (T,)
        """
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def train(
        self,
        num_iterations: int = 1000,
        log_interval: int = 10,
        wandb_run: Optional[Any] = None,
    ) -> None:
        """Run RL² meta-training loop.

        For each iteration:
        1. Sample a batch of tasks
        2. Collect trajectories with persistent hidden states
        3. Compute policy gradient loss with entropy bonus
        4. Update policy parameters (outer loop)

        Args:
            num_iterations: Number of meta-training iterations
            log_interval: How often to log training metrics
            wandb_run: Optional Wandb run for metric logging
        """
        logger.info(
            f"RL² Training: {num_iterations} iterations, "
            f"{self.num_tasks_per_batch} tasks/batch, "
            f"{self.episodes_per_task} episodes/task"
        )

        for iteration in range(num_iterations):
            self.policy.train()
            self.optimizer.zero_grad()

            total_loss = 0.0
            total_reward = 0.0
            total_steps = 0

            for task_idx in range(self.num_tasks_per_batch):
                trajectory = self.collect_task_trajectory()

                if len(trajectory['rewards']) == 0:
                    continue

                # Compute discounted returns
                returns = self.compute_returns(trajectory['rewards'])

                # Normalize returns
                if returns.std() > 1e-8:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

                # Stack log_probs and values
                log_probs = torch.stack(trajectory['log_probs'])
                values = torch.stack(trajectory['values']).squeeze()
                entropies = torch.stack(trajectory['entropies'])

                # Advantage = returns - baseline (value)
                advantages = returns - values.detach()

                # Policy loss (REINFORCE with baseline)
                policy_loss = -(log_probs * advantages).mean()

                # Value loss
                value_loss = F.mse_loss(values, returns)

                # Entropy bonus (encourage exploration)
                entropy_loss = -entropies.mean()

                # Total loss
                task_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                total_loss += task_loss
                total_reward += sum(trajectory['rewards'])
                total_steps += len(trajectory['rewards'])

            if total_steps == 0:
                continue

            # Average over tasks and backpropagate
            avg_loss = total_loss / self.num_tasks_per_batch
            avg_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

            self.optimizer.step()

            avg_reward = total_reward / self.num_tasks_per_batch

            if iteration % log_interval == 0:
                logger.info(
                    f"RL² Iter {iteration}/{num_iterations}, "
                    f"Loss: {avg_loss.item():.4f}, "
                    f"Avg Reward: {avg_reward:.2f}"
                )

            if wandb_run is not None:
                wandb_run.log({
                    'rl2/iteration': iteration,
                    'rl2/loss': avg_loss.item(),
                    'rl2/avg_reward': avg_reward,
                    'rl2/total_steps': total_steps,
                })


if __name__ == "__main__":
    import gymnasium as gym
    logging.basicConfig(level=logging.INFO)

    print("=" * 40)
    print("Testing RL² Implementation")
    print("=" * 40)

    policy = RL2Policy(obs_dim=4, action_dim=2, hidden_size=64, num_layers=1)
    print(f"Policy created: {sum(p.numel() for p in policy.parameters()):,} parameters")

    trainer = RL2Trainer(
        policy=policy,
        env_fn=lambda: gym.make('CartPole-v1'),
        lr=3e-4,
        episodes_per_task=2,
        max_steps_per_episode=50,
        num_tasks_per_batch=2,
    )

    trainer.train(num_iterations=3, log_interval=1)
    print("\n✓ RL² test completed successfully!")
