"""VariBAD (Variational Bayes-Adaptive Deep RL) Implementation

Reference: Zintgraf et al., 2019 - "VariBAD: A Very Good Method for
Bayes-Adaptive Deep RL via Meta-Learning"

Algorithm Overview:
    VariBAD combines variational inference with Bayes-adaptive MDPs to learn
    a belief state over tasks. An RNN-based encoder processes trajectory data
    to produce a posterior belief over the task distribution. The policy is
    then conditioned on this belief state, enabling principled exploration
    and adaptation.

    Key difference from PEARL: VariBAD uses an RNN to maintain a *sequential*
    belief state (updated at each timestep), while PEARL uses permutation-
    invariant encoding of collected transitions.

Architecture:
    Variational Encoder:
        Input:  (obs_t, action_{t-1}, reward_{t-1}) sequence
        RNN:    GRU layers processing the sequence
        Output: mu_t, log_sigma_t at each timestep (belief evolves over time)

    Policy:
        Input:  (obs_t, belief_t) where belief_t = (mu_t, log_sigma_t)
        MLP:    Policy network
        Output: Action distribution

    VAE Decoder (optional, for reward/transition prediction):
        Input:  (obs_t, action_t, z_t)
        Output: Predicted (reward, next_obs)

    Training:
        Loss = RL_loss(policy) + ELBO_loss(encoder, decoder) + KL_loss

Example:
    >>> from meta_rl.varibad import VariBADAgent
    >>> agent = VariBADAgent(obs_dim=4, action_dim=2, latent_dim=5)
    >>> agent.train_meta(env_fn, num_iterations=1000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Callable, Tuple, List, Dict, Any

logger = logging.getLogger(__name__)


class VariationalEncoder(nn.Module):
    """RNN-based variational encoder for VariBAD.

    Processes a sequence of (obs, prev_action, prev_reward) tuples through
    a GRU to produce a time-varying belief state. At each timestep t, the
    encoder outputs mu_t and log_sigma_t, representing the posterior belief
    about the current task given the history up to time t.

    Args:
        obs_dim: Observation space dimension
        action_dim: Action space dimension
        latent_dim: Dimension of the latent task variable z
        hidden_size: GRU hidden state dimension (default: 128)
        num_layers: Number of stacked GRU layers (default: 1)

    Architecture:
        Input projection: Linear(obs + action + reward -> hidden_size)
        GRU: hidden_size -> hidden_size
        Output heads: Linear(hidden_size -> latent_dim) for mu and log_sigma
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 5,
        hidden_size: int = 128,
        num_layers: int = 1,
    ):
        super(VariationalEncoder, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input: obs + action + reward
        input_dim = obs_dim + action_dim + 1

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
        )

        # GRU for sequential belief updating
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output heads for variational parameters
        self.mu_head = nn.Linear(hidden_size, latent_dim)
        self.log_sigma_head = nn.Linear(hidden_size, latent_dim)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize GRU hidden state.

        Args:
            batch_size: Number of parallel sequences

        Returns:
            Zero tensor (num_layers, batch_size, hidden_size)
        """
        device = next(self.parameters()).device
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )

    def forward(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: update belief given new observation.

        Can process single timestep or sequence.

        Args:
            obs: Observation (B, obs_dim) or (B, T, obs_dim)
            prev_action: Previous action (B, action_dim) or (B, T, action_dim)
            prev_reward: Previous reward (B, 1) or (B, T, 1)
            hidden: GRU hidden state (num_layers, B, hidden_size)

        Returns:
            Tuple of:
                - mu: Belief mean (B, latent_dim) or (B, T, latent_dim)
                - log_sigma: Belief log-std (B, latent_dim) or (B, T, latent_dim)
                - next_hidden: Updated hidden state
        """
        # Concatenate input components
        x = torch.cat([obs, prev_action, prev_reward], dim=-1)

        single_step = x.dim() == 2
        if single_step:
            x = x.unsqueeze(1)

        # Project input
        x = self.input_proj(x)  # (B, T, hidden_size)

        # GRU forward
        gru_out, next_hidden = self.gru(x, hidden)  # (B, T, hidden_size)

        # Compute variational parameters
        mu = self.mu_head(gru_out)
        log_sigma = self.log_sigma_head(gru_out)
        log_sigma = torch.clamp(log_sigma, min=-20.0, max=2.0)

        if single_step:
            mu = mu.squeeze(1)
            log_sigma = log_sigma.squeeze(1)

        return mu, log_sigma, next_hidden

    def sample_z(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample latent z using reparameterization trick.

        Args:
            mu: Belief mean (B, latent_dim)
            log_sigma: Belief log-std (B, latent_dim)

        Returns:
            Tuple of (z, kl_divergence)
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


class RewardDecoder(nn.Module):
    """Decoder for predicting rewards given (obs, action, z).

    Used as part of the VAE ELBO loss in VariBAD. Predicting rewards
    helps the encoder learn informative task representations.

    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        latent_dim: Latent task variable dimension
        hidden_dim: Hidden layer dimension
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
    ):
        super(RewardDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim + latent_dim, hidden_dim),
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
    ) -> torch.Tensor:
        """Predict reward given state, action, and task context.

        Args:
            obs: Observation (B, obs_dim)
            action: Action (B, action_dim)
            z: Latent task variable (B, latent_dim)

        Returns:
            Predicted reward (B, 1)
        """
        x = torch.cat([obs, action, z], dim=-1)
        return self.decoder(x)


class TransitionDecoder(nn.Module):
    """Decoder for predicting next state given (obs, action, z).

    Another component of the VAE ELBO loss. Predicting transitions
    encourages the latent z to capture task dynamics.

    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        latent_dim: Latent task variable dimension
        hidden_dim: Hidden layer dimension
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
    ):
        super(TransitionDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next observation given state, action, and context.

        Args:
            obs: Observation (B, obs_dim)
            action: Action (B, action_dim)
            z: Latent task variable (B, latent_dim)

        Returns:
            Predicted next observation (B, obs_dim)
        """
        x = torch.cat([obs, action, z], dim=-1)
        return self.decoder(x)


class VariBADPolicy(nn.Module):
    """Policy conditioned on belief state for VariBAD.

    Takes the current observation and belief state (mu, log_sigma or z)
    as input and outputs an action distribution.

    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        latent_dim: Latent belief dimension (uses 2*latent_dim for mu+log_sigma)
        hidden_dim: Hidden layer dimension
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
        super(VariBADPolicy, self).__init__()

        self.discrete = discrete
        self.action_dim = action_dim

        # Input: obs + belief (mu + log_sigma)
        input_dim = obs_dim + 2 * latent_dim

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

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        belief_mu: torch.Tensor,
        belief_log_sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: Observation (B, obs_dim)
            belief_mu: Belief mean (B, latent_dim)
            belief_log_sigma: Belief log-std (B, latent_dim)

        Returns:
            Tuple of (action_logits, value)
        """
        x = torch.cat([obs, belief_mu, belief_log_sigma], dim=-1)
        features = self.policy_net(x)

        if self.discrete:
            action_out = self.action_head(features)
        else:
            mean = self.mean_head(features)
            log_std = torch.clamp(self.log_std_head(features), -20.0, 2.0)
            action_out = (mean, log_std)

        value = self.value_head(features)
        return action_out, value

    def get_action(
        self,
        obs: torch.Tensor,
        belief_mu: torch.Tensor,
        belief_log_sigma: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        """Sample action from the policy.

        Args:
            obs: Observation (1, obs_dim)
            belief_mu: Belief mean (1, latent_dim)
            belief_log_sigma: Belief log-std (1, latent_dim)
            deterministic: Whether to use deterministic action

        Returns:
            Tuple of (action, log_prob, value)
        """
        action_out, value = self.forward(obs, belief_mu, belief_log_sigma)

        if self.discrete:
            probs = F.softmax(action_out, dim=-1)
            dist = torch.distributions.Categorical(probs)
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob, value
        else:
            mean, log_std = action_out
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            if deterministic:
                action = mean
            else:
                action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action.cpu().detach().numpy().flatten(), log_prob, value


class VariBADAgent:
    """VariBAD meta-RL agent.

    Combines variational task inference, reward/transition decoders,
    and a belief-conditioned policy for Bayes-adaptive meta-RL.

    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        latent_dim: Latent task variable dimension (default: 5)
        encoder_hidden: Encoder hidden size (default: 128)
        policy_hidden: Policy hidden size (default: 256)
        kl_weight: KL divergence loss weight (default: 0.1)
        reconstruction_weight: Reward/transition decoder loss weight (default: 1.0)
        lr: Learning rate (default: 3e-4)
        gamma: Discount factor (default: 0.99)
        discrete: Whether action space is discrete (default: True)
        device: Computation device (default: 'cpu')
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 5,
        encoder_hidden: int = 128,
        policy_hidden: int = 256,
        kl_weight: float = 0.1,
        reconstruction_weight: float = 1.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        discrete: bool = True,
        device: str = 'cpu',
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.reconstruction_weight = reconstruction_weight
        self.gamma = gamma
        self.discrete = discrete
        self.device = device

        # Networks
        action_input_dim = 1 if discrete else action_dim

        self.encoder = VariationalEncoder(
            obs_dim=obs_dim,
            action_dim=action_input_dim,
            latent_dim=latent_dim,
            hidden_size=encoder_hidden,
        ).to(device)

        self.policy = VariBADPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=policy_hidden,
            discrete=discrete,
        ).to(device)

        self.reward_decoder = RewardDecoder(
            obs_dim=obs_dim,
            action_dim=action_input_dim,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden,
        ).to(device)

        self.transition_decoder = TransitionDecoder(
            obs_dim=obs_dim,
            action_dim=action_input_dim,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden,
        ).to(device)

        # Single optimizer for all components (joint training)
        all_params = (
            list(self.encoder.parameters())
            + list(self.policy.parameters())
            + list(self.reward_decoder.parameters())
            + list(self.transition_decoder.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=lr)

    def collect_trajectory(
        self,
        env,
        max_steps: int = 200,
    ) -> Dict[str, List]:
        """Collect a trajectory with online belief updating.

        At each timestep, the encoder updates the belief, the policy
        selects actions based on the current belief, and all data is
        stored for later training.

        Args:
            env: Environment instance
            max_steps: Maximum steps per episode

        Returns:
            Dictionary of trajectory data
        """
        import gymnasium as gym

        self.encoder.eval()
        self.policy.eval()

        hidden = self.encoder.init_hidden(batch_size=1)
        prev_action = torch.zeros(1, 1 if self.discrete else self.action_dim, device=self.device)
        prev_reward = torch.zeros(1, 1, device=self.device)

        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        trajectory = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'next_obs': [],
            'dones': [],
            'log_probs': [],
            'values': [],
            'belief_mus': [],
            'belief_log_sigmas': [],
            'kl_divs': [],
        }

        done = False
        steps = 0

        while not done and steps < max_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Update belief
            with torch.no_grad():
                mu, log_sigma, hidden = self.encoder(
                    obs_tensor, prev_action, prev_reward, hidden
                )

            # Get action from belief-conditioned policy
            action, log_prob, value = self.policy.get_action(
                obs_tensor, mu, log_sigma
            )

            # Step environment
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result

            # Store trajectory data
            trajectory['obs'].append(obs)
            trajectory['actions'].append(action if not self.discrete else float(action))
            trajectory['rewards'].append(reward)
            trajectory['next_obs'].append(next_obs)
            trajectory['dones'].append(done)
            trajectory['log_probs'].append(log_prob)
            trajectory['values'].append(value)
            trajectory['belief_mus'].append(mu.detach())
            trajectory['belief_log_sigmas'].append(log_sigma.detach())

            # Update prev_action and prev_reward
            if self.discrete:
                prev_action = torch.tensor([[float(action)]], device=self.device)
            else:
                prev_action = torch.tensor([action], dtype=torch.float32, device=self.device)
            prev_reward = torch.tensor([[reward]], dtype=torch.float32, device=self.device)

            obs = next_obs
            steps += 1

        return trajectory

    def compute_vae_loss(
        self,
        trajectory: Dict[str, List],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute VAE ELBO loss for the encoder and decoders.

        The ELBO consists of:
        1. Reconstruction loss: reward prediction + transition prediction
        2. KL divergence: regularization towards prior N(0, I)

        Args:
            trajectory: Collected trajectory data

        Returns:
            Tuple of (reconstruction_loss, kl_loss)
        """
        if len(trajectory['obs']) < 2:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        obs = torch.tensor(np.array(trajectory['obs']), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(trajectory['actions']), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array(trajectory['rewards']), dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(np.array(trajectory['next_obs']), dtype=torch.float32, device=self.device)

        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)

        # Re-encode with gradients
        hidden = self.encoder.init_hidden(batch_size=1)
        prev_action = torch.zeros(1, 1 if self.discrete else self.action_dim, device=self.device)
        prev_reward = torch.zeros(1, 1, device=self.device)

        total_recon_loss = 0.0
        total_kl_loss = 0.0
        T = obs.shape[0]

        for t in range(T):
            obs_t = obs[t:t+1]
            mu_t, log_sigma_t, hidden = self.encoder(
                obs_t, prev_action, prev_reward, hidden
            )

            z_t, kl_t = self.encoder.sample_z(mu_t, log_sigma_t)

            # Reward prediction loss
            pred_reward = self.reward_decoder(obs_t, actions[t:t+1], z_t)
            reward_loss = F.mse_loss(pred_reward, rewards[t:t+1])

            # Transition prediction loss
            pred_next_obs = self.transition_decoder(obs_t, actions[t:t+1], z_t)
            transition_loss = F.mse_loss(pred_next_obs, next_obs[t:t+1])

            total_recon_loss = total_recon_loss + reward_loss + transition_loss
            total_kl_loss = total_kl_loss + kl_t

            # Update previous action/reward
            prev_action = actions[t:t+1]
            prev_reward = rewards[t:t+1]

        return total_recon_loss / T, total_kl_loss / T

    def compute_rl_loss(
        self,
        trajectory: Dict[str, List],
    ) -> torch.Tensor:
        """Compute RL policy gradient loss with value baseline.

        Args:
            trajectory: Collected trajectory data

        Returns:
            Policy gradient loss scalar
        """
        if len(trajectory['rewards']) == 0:
            return torch.tensor(0.0, device=self.device)

        rewards = trajectory['rewards']
        log_probs = torch.stack(trajectory['log_probs'])
        values = torch.stack(trajectory['values']).squeeze()

        # Compute discounted returns
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize returns
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Advantage
        advantages = returns - values.detach()

        # Policy loss
        policy_loss = -(log_probs * advantages).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        return policy_loss + 0.5 * value_loss

    def train_meta(
        self,
        env_fn: Callable,
        num_iterations: int = 1000,
        episodes_per_task: int = 5,
        num_tasks_per_batch: int = 4,
        log_interval: int = 10,
        wandb_run: Optional[Any] = None,
    ) -> None:
        """Run VariBAD meta-training loop.

        For each iteration:
        1. Sample tasks and collect trajectories with online belief updating
        2. Compute joint loss: RL loss + VAE ELBO (reconstruction + KL)
        3. Update all networks jointly

        Args:
            env_fn: Callable returning an environment instance
            num_iterations: Number of meta-training iterations
            episodes_per_task: Episodes per task (for multi-episode adaptation)
            num_tasks_per_batch: Tasks per meta-batch
            log_interval: Logging frequency
            wandb_run: Optional Wandb run for metric logging
        """
        logger.info(
            f"VariBAD Training: {num_iterations} iterations, "
            f"{num_tasks_per_batch} tasks/batch"
        )

        for iteration in range(num_iterations):
            self.encoder.train()
            self.policy.train()
            self.reward_decoder.train()
            self.transition_decoder.train()

            self.optimizer.zero_grad()

            total_loss = 0.0
            total_rl_loss = 0.0
            total_recon_loss = 0.0
            total_kl_loss = 0.0
            total_reward = 0.0

            for task in range(num_tasks_per_batch):
                env = env_fn()

                for episode in range(episodes_per_task):
                    trajectory = self.collect_trajectory(env, max_steps=200)

                    # RL loss
                    rl_loss = self.compute_rl_loss(trajectory)

                    # VAE loss
                    recon_loss, kl_loss = self.compute_vae_loss(trajectory)

                    # Combined loss
                    task_loss = (
                        rl_loss
                        + self.reconstruction_weight * recon_loss
                        + self.kl_weight * kl_loss
                    )

                    total_loss += task_loss
                    total_rl_loss += rl_loss.item()
                    total_recon_loss += recon_loss.item()
                    total_kl_loss += kl_loss.item()
                    total_reward += sum(trajectory['rewards'])

                env.close()

            # Average and backpropagate
            n = num_tasks_per_batch * episodes_per_task
            avg_loss = total_loss / n
            avg_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

            self.optimizer.step()

            if iteration % log_interval == 0:
                logger.info(
                    f"VariBAD Iter {iteration}/{num_iterations}, "
                    f"RL Loss: {total_rl_loss/n:.4f}, "
                    f"Recon Loss: {total_recon_loss/n:.4f}, "
                    f"KL Loss: {total_kl_loss/n:.4f}, "
                    f"Avg Reward: {total_reward/n:.2f}"
                )

            if wandb_run is not None:
                wandb_run.log({
                    'varibad/iteration': iteration,
                    'varibad/rl_loss': total_rl_loss / n,
                    'varibad/recon_loss': total_recon_loss / n,
                    'varibad/kl_loss': total_kl_loss / n,
                    'varibad/avg_reward': total_reward / n,
                })


if __name__ == "__main__":
    import gymnasium as gym
    logging.basicConfig(level=logging.INFO)

    print("=" * 40)
    print("Testing VariBAD Implementation")
    print("=" * 40)

    agent = VariBADAgent(
        obs_dim=4, action_dim=2, latent_dim=5,
        encoder_hidden=64, policy_hidden=64,
    )

    total_params = sum(
        sum(p.numel() for p in net.parameters())
        for net in [agent.encoder, agent.policy, agent.reward_decoder, agent.transition_decoder]
    )
    print(f"Total parameters: {total_params:,}")

    agent.train_meta(
        env_fn=lambda: gym.make('CartPole-v1'),
        num_iterations=3,
        episodes_per_task=1,
        num_tasks_per_batch=2,
        log_interval=1,
    )

    print("\n✓ VariBAD test completed successfully!")
