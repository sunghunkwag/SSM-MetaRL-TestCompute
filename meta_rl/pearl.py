"""PEARL (Probabilistic Embeddings for Actor-critic RL) Implementation

Reference: Rakelly et al., 2019 - "Efficient Off-Policy Meta-Reinforcement
Learning via Probabilistic Context Variables"

Algorithm Overview:
    PEARL performs probabilistic context inference for task identification.
    A context encoder maps trajectory data (s, a, r, s') to a latent context
    variable z, which conditions the policy network. This enables rapid
    adaptation to new tasks by inferring their identity from a few transitions.

Architecture:
    Context Encoder: (s, a, r, s') -> MLP -> permutation-invariant aggregation -> (mu, sigma)
    Latent Context:  z ~ N(mu, sigma)
    Policy:          pi(a | s, z) via SAC-style actor
    Q-Function:      Q(s, a, z) via twin critics

    The context encoder uses mean pooling over a set of (s, a, r, s') tuples,
    making it permutation-invariant. The policy is conditioned on the sampled
    context z, allowing it to adapt its behavior based on the inferred task.

Key Design Choices:
    - Off-policy: Uses SAC for efficient sample reuse
    - Permutation-invariant encoder: Order of transitions doesn't matter
    - KL regularization: Prevents posterior collapse and encourages
      informative latent representations
    - Separate replay buffers per task for proper context encoding

Example:
    >>> from meta_rl.pearl import PEARLAgent
    >>> agent = PEARLAgent(obs_dim=4, action_dim=2, latent_dim=5)
    >>> agent.train_meta(env_fn, num_iterations=1000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Callable, Tuple, List, Dict, Any
from collections import deque

logger = logging.getLogger(__name__)


class ContextEncoder(nn.Module):
    """Permutation-invariant context encoder for PEARL.

    Encodes a set of (state, action, reward, next_state) transitions into
    a latent context variable z using mean pooling. The encoder outputs
    parameters of a Gaussian distribution (mu, log_sigma).

    Args:
        obs_dim: Observation space dimension
        action_dim: Action space dimension
        latent_dim: Dimension of the latent context variable z
        hidden_dim: Hidden layer dimension (default: 128)

    Architecture:
        Input: (s, a, r, s') concatenated -> (obs_dim + action_dim + 1 + obs_dim)
        MLP:   3-layer with ReLU activations
        Aggregation: Mean pooling over transitions
        Output: mu (latent_dim), log_sigma (latent_dim)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 5,
        hidden_dim: int = 128,
    ):
        super(ContextEncoder, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Input: (s, a, r, s') concatenated
        input_dim = obs_dim + action_dim + 1 + obs_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output heads for Gaussian parameters
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode transitions into latent context distribution parameters.

        Args:
            obs: Observations, shape (B, N, obs_dim) where N is number of transitions
            actions: Actions, shape (B, N, action_dim)
            rewards: Rewards, shape (B, N, 1)
            next_obs: Next observations, shape (B, N, obs_dim)

        Returns:
            Tuple of:
                - mu: Mean of latent distribution, shape (B, latent_dim)
                - log_sigma: Log std of latent distribution, shape (B, latent_dim)
        """
        # Concatenate transition components
        inputs = torch.cat([obs, actions, rewards, next_obs], dim=-1)  # (B, N, input_dim)

        # Encode each transition independently
        encoded = self.encoder(inputs)  # (B, N, hidden_dim)

        # Permutation-invariant aggregation: mean pooling
        aggregated = encoded.mean(dim=1)  # (B, hidden_dim)

        # Output distribution parameters
        mu = self.mu_head(aggregated)  # (B, latent_dim)
        log_sigma = self.log_sigma_head(aggregated)  # (B, latent_dim)

        # Clamp log_sigma for numerical stability
        log_sigma = torch.clamp(log_sigma, min=-20.0, max=2.0)

        return mu, log_sigma

    def sample_z(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample latent context using reparameterization trick.

        Args:
            mu: Mean of latent distribution (B, latent_dim)
            log_sigma: Log std of latent distribution (B, latent_dim)

        Returns:
            Tuple of:
                - z: Sampled latent context (B, latent_dim)
                - kl_div: KL divergence from prior N(0, I), scalar
        """
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps

        # KL divergence from standard normal prior
        kl_div = -0.5 * torch.sum(
            1 + 2 * log_sigma - mu.pow(2) - sigma.pow(2),
            dim=-1,
        ).mean()

        return z, kl_div


class ContextConditionedPolicy(nn.Module):
    """Policy network conditioned on latent context z.

    Maps (observation, context) pairs to action distributions.
    For discrete action spaces, outputs action logits.
    For continuous action spaces, outputs mean and log_std.

    Args:
        obs_dim: Observation space dimension
        action_dim: Action space dimension
        latent_dim: Latent context dimension
        hidden_dim: Hidden layer dimension (default: 256)
        discrete: Whether the action space is discrete
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        discrete: bool = True,
    ):
        super(ContextConditionedPolicy, self).__init__()

        self.discrete = discrete
        self.action_dim = action_dim

        input_dim = obs_dim + latent_dim

        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if discrete:
            self.action_head = nn.Linear(hidden_dim, action_dim)
        else:
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: compute action distribution given obs and context.

        Args:
            obs: Observation tensor (B, obs_dim)
            z: Latent context (B, latent_dim)

        Returns:
            Action logits (B, action_dim) for discrete, or
            (mean, log_std) tuple for continuous
        """
        x = torch.cat([obs, z], dim=-1)
        features = self.policy_net(x)

        if self.discrete:
            return self.action_head(features)
        else:
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, min=-20.0, max=2.0)
            return mean, log_std

    def get_action(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[Any, torch.Tensor]:
        """Sample action from the policy.

        Args:
            obs: Observation (1, obs_dim)
            z: Latent context (1, latent_dim)
            deterministic: If True, use argmax/mean instead of sampling

        Returns:
            Tuple of (action, log_prob)
        """
        if self.discrete:
            logits = self.forward(obs, z)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob
        else:
            mean, log_std = self.forward(obs, z)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            if deterministic:
                action = mean
            else:
                action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action.cpu().detach().numpy().flatten(), log_prob


class ContextConditionedCritic(nn.Module):
    """Q-value network conditioned on latent context z.

    Twin Q-networks for reducing overestimation bias (as in SAC/TD3).

    Args:
        obs_dim: Observation space dimension
        action_dim: Action space dimension (1 for discrete)
        latent_dim: Latent context dimension
        hidden_dim: Hidden layer dimension
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
    ):
        super(ContextConditionedCritic, self).__init__()

        input_dim = obs_dim + action_dim + latent_dim

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Q-values from both critics.

        Args:
            obs: Observations (B, obs_dim)
            action: Actions (B, action_dim)
            z: Latent context (B, latent_dim)

        Returns:
            Tuple of Q1 and Q2 values, each (B, 1)
        """
        x = torch.cat([obs, action, z], dim=-1)
        return self.q1(x), self.q2(x)


class TaskReplayBuffer:
    """Per-task replay buffer for PEARL.

    Stores transition data for a single task. Used for both context
    encoding and off-policy RL training.

    Args:
        capacity: Maximum number of transitions to store
    """

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        obs: np.ndarray,
        action: Any,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode ended
        """
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(
        self,
        batch_size: int,
        device: str = 'cpu',
    ) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            device: Device for tensor placement

        Returns:
            Dictionary with keys: obs, actions, rewards, next_obs, dones
        """
        indices = np.random.randint(0, len(self.buffer), size=min(batch_size, len(self.buffer)))
        batch = [self.buffer[i] for i in indices]

        obs, actions, rewards, next_obs, dones = zip(*batch)

        actions_arr = np.array(actions, dtype=np.float32)
        # Ensure actions always have a trailing dimension (N,) -> (N, 1)
        # This prevents shape mismatch in context encoder concatenation
        if actions_arr.ndim == 1:
            actions_arr = actions_arr.reshape(-1, 1)

        return {
            'obs': torch.tensor(np.array(obs), dtype=torch.float32, device=device),
            'actions': torch.tensor(actions_arr, dtype=torch.float32, device=device),
            'rewards': torch.tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(-1),
            'next_obs': torch.tensor(np.array(next_obs), dtype=torch.float32, device=device),
            'dones': torch.tensor(np.array(dones), dtype=torch.float32, device=device).unsqueeze(-1),
        }

    def sample_context(
        self,
        num_context: int,
        device: str = 'cpu',
    ) -> Dict[str, torch.Tensor]:
        """Sample transitions for context encoding.

        Returns data shaped for the context encoder: (1, N, dim).

        Args:
            num_context: Number of context transitions
            device: Device for tensor placement

        Returns:
            Dictionary with batched context data
        """
        data = self.sample(num_context, device)
        # Add batch dimension for encoder: (N, D) -> (1, N, D)
        return {k: v.unsqueeze(0) for k, v in data.items()}

    def __len__(self) -> int:
        return len(self.buffer)


class PEARLAgent:
    """PEARL meta-RL agent with probabilistic context inference.

    Combines a context encoder (for task inference), a context-conditioned
    SAC policy (for action selection), and per-task replay buffers
    (for off-policy training).

    Args:
        obs_dim: Observation space dimension
        action_dim: Action space dimension
        latent_dim: Dimension of latent context z (default: 5)
        encoder_hidden_dim: Context encoder hidden size (default: 128)
        policy_hidden_dim: Policy network hidden size (default: 256)
        kl_weight: Weight for KL divergence loss (default: 0.1)
        lr: Learning rate for all networks (default: 3e-4)
        gamma: Discount factor (default: 0.99)
        discrete: Whether the action space is discrete (default: True)
        device: Device for computation (default: 'cpu')
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 5,
        encoder_hidden_dim: int = 128,
        policy_hidden_dim: int = 256,
        kl_weight: float = 0.1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        discrete: bool = True,
        device: str = 'cpu',
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.gamma = gamma
        self.discrete = discrete
        self.device = device

        # Networks
        self.encoder = ContextEncoder(
            obs_dim=obs_dim,
            action_dim=action_dim if not discrete else 1,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden_dim,
        ).to(device)

        self.policy = ContextConditionedPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=policy_hidden_dim,
            discrete=discrete,
        ).to(device)

        self.critic = ContextConditionedCritic(
            obs_dim=obs_dim,
            action_dim=1 if discrete else action_dim,
            latent_dim=latent_dim,
            hidden_dim=policy_hidden_dim,
        ).to(device)

        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr
        )

        # Per-task replay buffers
        self.task_buffers: Dict[int, TaskReplayBuffer] = {}

    def _get_task_buffer(self, task_id: int) -> TaskReplayBuffer:
        """Get or create replay buffer for a task.

        Args:
            task_id: Task identifier

        Returns:
            TaskReplayBuffer for the specified task
        """
        if task_id not in self.task_buffers:
            self.task_buffers[task_id] = TaskReplayBuffer()
        return self.task_buffers[task_id]

    def collect_data(
        self,
        env,
        task_id: int,
        num_episodes: int = 1,
        max_steps: int = 200,
        z: Optional[torch.Tensor] = None,
    ) -> float:
        """Collect transition data for a specific task.

        Args:
            env: Environment instance
            task_id: Task identifier for replay buffer
            num_episodes: Number of episodes to collect
            max_steps: Maximum steps per episode
            z: Latent context for action selection (None uses prior)

        Returns:
            Average episode return
        """
        buffer = self._get_task_buffer(task_id)
        total_return = 0.0

        if z is None:
            z = torch.zeros(1, self.latent_dim, device=self.device)

        for ep in range(num_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            done = False
            ep_return = 0.0
            steps = 0

            while not done and steps < max_steps:
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32
                ).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action, _ = self.policy.get_action(obs_tensor, z)

                step_result = env.step(action)
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, info = step_result

                action_store = action if not self.discrete else float(action)
                buffer.add(obs, action_store, reward, next_obs, done)

                obs = next_obs
                ep_return += reward
                steps += 1

            total_return += ep_return

        return total_return / num_episodes

    def infer_context(self, task_id: int, num_context: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infer latent context from task buffer.

        Samples context transitions from the task's replay buffer,
        encodes them, and returns sampled z and KL divergence.

        Args:
            task_id: Task identifier
            num_context: Number of context transitions to use

        Returns:
            Tuple of (z, kl_div)
        """
        buffer = self._get_task_buffer(task_id)

        if len(buffer) < num_context:
            # Not enough data, use prior
            z = torch.zeros(1, self.latent_dim, device=self.device)
            kl_div = torch.tensor(0.0, device=self.device)
            return z, kl_div

        context = buffer.sample_context(num_context, self.device)

        mu, log_sigma = self.encoder(
            context['obs'],
            context['actions'],
            context['rewards'],
            context['next_obs'],
        )

        z, kl_div = self.encoder.sample_z(mu, log_sigma)
        return z, kl_div

    def update(
        self,
        task_ids: List[int],
        batch_size: int = 64,
        num_context: int = 16,
    ) -> Dict[str, float]:
        """Perform one meta-update step across tasks.

        For each task:
        1. Infer context z from context transitions
        2. Sample training batch from task buffer
        3. Compute policy and critic losses
        4. Add KL regularization loss

        Args:
            task_ids: List of task identifiers to update on
            batch_size: Training batch size per task
            num_context: Number of context transitions for inference

        Returns:
            Dictionary of loss metrics
        """
        total_policy_loss = 0.0
        total_critic_loss = 0.0
        total_kl_loss = 0.0
        valid_tasks = 0

        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        for task_id in task_ids:
            buffer = self._get_task_buffer(task_id)
            if len(buffer) < batch_size:
                continue

            valid_tasks += 1

            # Infer context
            z, kl_div = self.infer_context(task_id, num_context)

            # Sample training batch
            batch = buffer.sample(batch_size, self.device)

            # Expand z to batch size
            z_expanded = z.expand(batch['obs'].shape[0], -1)

            # Policy loss
            if self.discrete:
                logits = self.policy.forward(batch['obs'], z_expanded)
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)

                actions_1h = batch['actions'].long()
                q1, q2 = self.critic(
                    batch['obs'],
                    batch['actions'],
                    z_expanded,
                )
                q_min = torch.min(q1, q2)

                policy_loss = -(probs * (q_min - 0.2 * log_probs)).sum(dim=-1).mean()
            else:
                action, log_prob = self.policy.get_action(
                    batch['obs'], z_expanded
                )
                q1, q2 = self.critic(
                    batch['obs'],
                    torch.tensor(action, device=self.device).float(),
                    z_expanded,
                )
                q_min = torch.min(q1, q2)
                policy_loss = (0.2 * log_prob - q_min.squeeze()).mean()

            # Critic loss (simple TD target with no target network for brevity)
            with torch.no_grad():
                # Use current Q-values as target (simplified)
                target = batch['rewards'] + self.gamma * (1 - batch['dones']) * q_min.detach()

            q1_pred, q2_pred = self.critic(
                batch['obs'],
                batch['actions'],
                z_expanded,
            )
            critic_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

            # Total task loss
            task_loss = policy_loss + critic_loss + self.kl_weight * kl_div

            total_policy_loss += policy_loss.item()
            total_critic_loss += critic_loss.item()
            total_kl_loss += kl_div.item()
            task_loss.backward()

        if valid_tasks > 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

            self.encoder_optimizer.step()
            self.policy_optimizer.step()
            self.critic_optimizer.step()

        return {
            'policy_loss': total_policy_loss / max(valid_tasks, 1),
            'critic_loss': total_critic_loss / max(valid_tasks, 1),
            'kl_loss': total_kl_loss / max(valid_tasks, 1),
        }

    def train_meta(
        self,
        env_fn: Callable,
        num_iterations: int = 1000,
        episodes_per_task: int = 5,
        num_tasks_per_batch: int = 5,
        log_interval: int = 10,
        wandb_run: Optional[Any] = None,
    ) -> None:
        """Run PEARL meta-training loop.

        For each iteration:
        1. Sample a batch of tasks
        2. Collect exploration data with prior context
        3. Infer posterior context from collected data
        4. Collect exploitation data with posterior context
        5. Update all networks

        Args:
            env_fn: Callable returning an environment instance
            num_iterations: Number of meta-training iterations
            episodes_per_task: Episodes to collect per task
            num_tasks_per_batch: Number of tasks per meta-batch
            log_interval: How often to log training metrics
            wandb_run: Optional Wandb run for metric logging
        """
        logger.info(
            f"PEARL Training: {num_iterations} iterations, "
            f"{num_tasks_per_batch} tasks/batch"
        )

        for iteration in range(num_iterations):
            task_ids = list(range(num_tasks_per_batch))

            # Phase 1: Collect exploration data with prior
            for task_id in task_ids:
                env = env_fn()
                self.collect_data(
                    env, task_id,
                    num_episodes=episodes_per_task // 2 + 1,
                    z=None,
                )
                env.close()

            # Phase 2: Collect exploitation data with inferred context
            for task_id in task_ids:
                z, _ = self.infer_context(task_id)
                env = env_fn()
                avg_return = self.collect_data(
                    env, task_id,
                    num_episodes=episodes_per_task // 2 + 1,
                    z=z,
                )
                env.close()

            # Phase 3: Update networks
            losses = self.update(task_ids)

            if iteration % log_interval == 0:
                logger.info(
                    f"PEARL Iter {iteration}/{num_iterations}, "
                    f"Policy Loss: {losses['policy_loss']:.4f}, "
                    f"Critic Loss: {losses['critic_loss']:.4f}, "
                    f"KL Loss: {losses['kl_loss']:.4f}"
                )

            if wandb_run is not None:
                wandb_run.log({
                    'pearl/iteration': iteration,
                    'pearl/policy_loss': losses['policy_loss'],
                    'pearl/critic_loss': losses['critic_loss'],
                    'pearl/kl_loss': losses['kl_loss'],
                })

            # Rotate task buffers to prevent stale data
            if iteration % 50 == 0 and iteration > 0:
                self.task_buffers.clear()


if __name__ == "__main__":
    import gymnasium as gym
    logging.basicConfig(level=logging.INFO)

    print("=" * 40)
    print("Testing PEARL Implementation")
    print("=" * 40)

    agent = PEARLAgent(
        obs_dim=4, action_dim=2, latent_dim=5,
        encoder_hidden_dim=64, policy_hidden_dim=64,
    )

    total_params = sum(
        sum(p.numel() for p in net.parameters())
        for net in [agent.encoder, agent.policy, agent.critic]
    )
    print(f"Total parameters: {total_params:,}")

    agent.train_meta(
        env_fn=lambda: gym.make('CartPole-v1'),
        num_iterations=3,
        episodes_per_task=2,
        num_tasks_per_batch=2,
        log_interval=1,
    )

    print("\n✓ PEARL test completed successfully!")
