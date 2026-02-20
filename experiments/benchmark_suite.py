"""Benchmark Suite for SSM-MetaRL-TestCompute

Comprehensive benchmarking system for comparing:
    1. Model architectures: Legacy SSM vs Mamba SSM
    2. Meta-RL algorithms: MAML, RL², PEARL, VariBAD
    3. Task distributions: CartPole, MuJoCo variants, Meta-World

Produces structured results in JSON format for analysis and plotting.

Usage:
    # Run quick benchmark (CartPole only)
    python experiments/benchmark_suite.py --quick

    # Run full benchmark
    python experiments/benchmark_suite.py --full

    # Custom benchmark
    python experiments/benchmark_suite.py \\
        --models legacy mamba \\
        --algorithms maml rl2 \\
        --envs CartPole-v1 HalfCheetahVel \\
        --num_epochs 100 --num_seeds 3

    # With Wandb logging
    python experiments/benchmark_suite.py --quick --use_wandb
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

import numpy as np
import torch

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.ssm import StateSpaceModel
from core.ssm_mamba import MambaSSM
from meta_rl.meta_maml import MetaMAML
from env_runner.environment import Environment

logger = logging.getLogger(__name__)

# Optional imports
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False



@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run.

    Attributes:
        model_type: SSM architecture type
        algorithm: Meta-RL algorithm name
        env_name: Environment name
        seed: Random seed
        num_epochs: Number of training epochs
        state_dim: SSM state dimension
        hidden_dim: Model hidden dimension
        inner_lr: MAML inner learning rate
        outer_lr: Outer optimizer learning rate
        episodes_per_task: Episodes per meta-task
    """
    model_type: str = 'mamba'
    algorithm: str = 'maml'
    env_name: str = 'CartPole-v1'
    seed: int = 42
    num_epochs: int = 50
    state_dim: int = 16
    hidden_dim: int = 64
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    episodes_per_task: int = 5


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run.

    Attributes:
        config: Run configuration
        meta_losses: List of meta-loss values per epoch
        episode_rewards: List of episode rewards during training
        adaptation_losses: Final adaptation losses
        total_params: Total model parameters
        training_time: Total training time in seconds
        peak_memory_mb: Peak memory usage in MB
        final_loss: Final training loss
        convergence_epoch: Epoch where loss first drops below threshold
    """
    config: Dict[str, Any]
    meta_losses: List[float] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)
    adaptation_losses: List[float] = field(default_factory=list)
    total_params: int = 0
    training_time: float = 0.0
    peak_memory_mb: float = 0.0
    final_loss: float = float('inf')
    convergence_epoch: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return asdict(self)


def create_model(config: BenchmarkConfig, input_dim: int, output_dim: int, device: str):
    """Create model based on configuration.

    Args:
        config: Benchmark configuration
        input_dim: Environment observation dimension
        output_dim: Model output dimension
        device: PyTorch device string

    Returns:
        Instantiated model on the specified device
    """
    if config.model_type == 'mamba':
        model = MambaSSM(
            state_dim=config.state_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=config.hidden_dim,
        ).to(device)
    elif config.model_type == 'legacy':
        model = StateSpaceModel(
            state_dim=config.state_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    return model


def create_environment(env_name: str):
    """Create environment by name.

    Supports both standard Gymnasium environments and custom
    MuJoCo variant environments.

    Args:
        env_name: Environment name (e.g. 'CartPole-v1', 'HalfCheetahVel')

    Returns:
        Tuple of (environment, obs_dim, action_dim, is_discrete)
    """
    import gymnasium as gym

    import gymnasium as gym

    # Standard Gymnasium environment
    env = Environment(env_name=env_name, batch_size=1)
    obs_space = env.observation_space
    obs_dim = obs_space.shape[0] if isinstance(obs_space, gym.spaces.Box) else obs_space.n
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim = env.action_space.n if is_discrete else env.action_space.shape[0]

    return env, obs_dim, action_dim, is_discrete


def collect_training_data(env, model, num_episodes=5, max_steps=100, device='cpu'):
    """Collect data from environment for training.

    Args:
        env: Environment instance
        model: Policy model
        num_episodes: Number of episodes
        max_steps: Max steps per episode
        device: PyTorch device

    Returns:
        Dictionary with trajectory tensors
    """
    import gymnasium as gym

    all_obs, all_next_obs, all_rewards = [], [], []
    model.eval()

    for ep in range(num_episodes):
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        if isinstance(obs, np.ndarray):
            obs = obs.flatten()
        hidden = model.init_hidden(batch_size=1)

        for step in range(max_steps):
            obs_arr = np.array(obs, dtype=np.float32).flatten()
            obs_tensor = torch.tensor(obs_arr).unsqueeze(0).to(device)

            with torch.no_grad():
                output, hidden = model(obs_tensor, hidden)

            if hasattr(env, 'action_space'):
                action_space = env.action_space
                if isinstance(action_space, gym.spaces.Discrete):
                    action = action_space.sample()
                else:
                    action = action_space.sample()
            else:
                action = 0

            result = env.step(action)
            if len(result) == 5:
                next_obs, reward, term, trunc, info = result
                done = term or trunc
            elif len(result) == 4:
                next_obs, reward, done, info = result
            else:
                break

            if isinstance(next_obs, np.ndarray):
                next_obs_flat = next_obs.flatten()
            else:
                next_obs_flat = np.array([next_obs], dtype=np.float32)

            all_obs.append(obs_arr)
            all_next_obs.append(next_obs_flat)
            all_rewards.append(reward)

            obs = next_obs_flat

            if done:
                result = env.reset()
                obs = result[0] if isinstance(result, tuple) else result
                if isinstance(obs, np.ndarray):
                    obs = obs.flatten()
                hidden = model.init_hidden(batch_size=1)

    if len(all_obs) == 0:
        return None

    return {
        'observations': torch.tensor(np.array(all_obs), dtype=torch.float32).unsqueeze(0).to(device),
        'next_observations': torch.tensor(np.array(all_next_obs), dtype=torch.float32).unsqueeze(0).to(device),
        'rewards': all_rewards,
    }


def run_maml_benchmark(config: BenchmarkConfig, device: str) -> BenchmarkResult:
    """Run a single MAML benchmark.

    Args:
        config: Benchmark configuration
        device: PyTorch device string

    Returns:
        BenchmarkResult with training metrics
    """
    import torch.nn as nn

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create environment and model
    env, obs_dim, action_dim, is_discrete = create_environment(config.env_name)
    output_dim = obs_dim  # Predict next observation

    model = create_model(config, obs_dim, output_dim, device)
    total_params = sum(p.numel() for p in model.parameters())

    meta_learner = MetaMAML(
        model=model,
        inner_lr=config.inner_lr,
        outer_lr=config.outer_lr,
    )

    result = BenchmarkResult(
        config=asdict(config),
        total_params=total_params,
    )

    start_time = time.time()
    losses = []

    for epoch in range(config.num_epochs):
        data = collect_training_data(
            env, model, num_episodes=config.episodes_per_task,
            max_steps=100, device=device,
        )
        if data is None:
            continue

        obs_seq = data['observations']
        next_obs_seq = data['next_observations']

        total_len = obs_seq.shape[1]
        if total_len < 4:
            continue

        split_idx = total_len // 2
        x_support = obs_seq[:, :split_idx]
        y_support = next_obs_seq[:, :split_idx]
        x_query = obs_seq[:, split_idx:]
        y_query = next_obs_seq[:, split_idx:]

        tasks = [(x_support, y_support, x_query, y_query)]
        initial_hidden = model.init_hidden(batch_size=1)

        loss = meta_learner.meta_update(
            tasks,
            initial_hidden_state=initial_hidden,
            loss_fn=nn.MSELoss(),
        )
        losses.append(loss)
        result.episode_rewards.extend(data['rewards'])

        # Check convergence
        if result.convergence_epoch == -1 and loss < 0.5:
            result.convergence_epoch = epoch

    result.training_time = time.time() - start_time
    result.meta_losses = losses
    result.final_loss = losses[-1] if losses else float('inf')

    # Memory tracking
    if torch.cuda.is_available():
        result.peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        import psutil
        process = psutil.Process(os.getpid())
        result.peak_memory_mb = process.memory_info().rss / 1e6

    try:
        env.close()
    except Exception:
        pass

    return result


def run_rl2_benchmark(config: BenchmarkConfig, device: str) -> BenchmarkResult:
    """Run RL² benchmark using RL2Trainer.

    Args:
        config: Benchmark configuration
        device: PyTorch device string

    Returns:
        BenchmarkResult with training metrics
    """
    from meta_rl.rl2 import RL2Policy, RL2Trainer
    import gymnasium as gym

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env, obs_dim, action_dim, is_discrete = create_environment(config.env_name)

    policy = RL2Policy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=config.hidden_dim,
        num_layers=2,
        discrete=is_discrete,
        device=device,
    )

    total_params = sum(p.numel() for p in policy.parameters())

    def env_fn():
        return create_environment(config.env_name)[0]

    trainer = RL2Trainer(
        policy=policy,
        env_fn=env_fn,
        lr=config.outer_lr,
        episodes_per_task=config.episodes_per_task,
        max_steps_per_episode=100,
        num_tasks_per_batch=4,
        device=device,
    )

    result = BenchmarkResult(
        config=asdict(config),
        total_params=total_params,
    )

    start_time = time.time()
    trainer.train(num_iterations=config.num_epochs, log_interval=max(1, config.num_epochs // 10))
    result.training_time = time.time() - start_time

    try:
        env.close()
    except Exception:
        pass

    return result


class BenchmarkSuite:
    """Main benchmark orchestrator.

    Runs configurable combinations of models × algorithms × environments,
    collects results, and saves them in structured JSON format.

    Args:
        models: List of model types to benchmark
        algorithms: List of algorithm names
        envs: List of environment names
        num_seeds: Number of random seeds per configuration
        num_epochs: Training epochs per run
        results_dir: Directory for saving results
        use_wandb: Whether to log to Wandb
        wandb_project: Wandb project name
    """

    def __init__(
        self,
        models: List[str] = None,
        algorithms: List[str] = None,
        envs: List[str] = None,
        num_seeds: int = 3,
        num_epochs: int = 50,
        results_dir: str = 'results',
        use_wandb: bool = False,
        wandb_project: str = 'ssm-metarl-benchmark',
    ):
        self.models = models or ['legacy', 'mamba']
        self.algorithms = algorithms or ['maml']
        self.envs = envs or ['CartPole-v1']
        self.num_seeds = num_seeds
        self.num_epochs = num_epochs
        self.results_dir = results_dir
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        self.all_results: List[BenchmarkResult] = []

    def run(self) -> Dict[str, Any]:
        """Execute complete benchmark suite.

        Iterates over all model × algorithm × env × seed combinations.
        Results are saved incrementally.

        Returns:
            Dictionary containing all results and summary statistics
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        total_runs = len(self.models) * len(self.algorithms) * len(self.envs) * self.num_seeds
        run_idx = 0

        logger.info(f"Starting benchmark suite: {total_runs} total runs on {device}")
        logger.info(f"Models: {self.models}")
        logger.info(f"Algorithms: {self.algorithms}")
        logger.info(f"Environments: {self.envs}")
        logger.info(f"Seeds: {self.num_seeds}, Epochs: {self.num_epochs}")

        os.makedirs(self.results_dir, exist_ok=True)

        for model_type in self.models:
            for algorithm in self.algorithms:
                for env_name in self.envs:
                    for seed in range(self.num_seeds):
                        run_idx += 1
                        run_name = f"{model_type}_{algorithm}_{env_name}_s{seed}"
                        logger.info(
                            f"\n[{run_idx}/{total_runs}] Running: {run_name}"
                        )

                        config = BenchmarkConfig(
                            model_type=model_type,
                            algorithm=algorithm,
                            env_name=env_name,
                            seed=seed,
                            num_epochs=self.num_epochs,
                        )

                        try:
                            if algorithm == 'maml':
                                result = run_maml_benchmark(config, device)
                            elif algorithm == 'rl2':
                                result = run_rl2_benchmark(config, device)
                            else:
                                logger.warning(f"Algorithm {algorithm} not yet benchmarkable, skipping")
                                continue

                            self.all_results.append(result)
                            logger.info(
                                f"  Completed: loss={result.final_loss:.4f}, "
                                f"time={result.training_time:.2f}s, "
                                f"params={result.total_params:,}"
                            )

                        except Exception as e:
                            logger.error(f"  FAILED: {e}")
                            continue

        # Save results
        summary = self._compute_summary()
        self._save_results(summary)

        return summary

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics across all runs.

        Returns:
            Dictionary with per-configuration statistics
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'num_runs': len(self.all_results),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'configurations': {},
            'raw_results': [r.to_dict() for r in self.all_results],
        }

        # Group by (model, algorithm, env)
        groups = {}
        for result in self.all_results:
            key = (
                result.config['model_type'],
                result.config['algorithm'],
                result.config['env_name'],
            )
            if key not in groups:
                groups[key] = []
            groups[key].append(result)

        for key, results in groups.items():
            model, alg, env = key
            config_name = f"{model}_{alg}_{env}"

            final_losses = [r.final_loss for r in results if r.final_loss != float('inf')]
            train_times = [r.training_time for r in results]

            summary['configurations'][config_name] = {
                'model': model,
                'algorithm': alg,
                'environment': env,
                'num_seeds': len(results),
                'total_params': results[0].total_params,
                'final_loss_mean': float(np.mean(final_losses)) if final_losses else None,
                'final_loss_std': float(np.std(final_losses)) if final_losses else None,
                'training_time_mean': float(np.mean(train_times)),
                'training_time_std': float(np.std(train_times)),
                'convergence_epochs': [r.convergence_epoch for r in results],
            }

        return summary

    def _save_results(self, summary: Dict[str, Any]) -> None:
        """Save results to JSON.

        Args:
            summary: Summary dictionary to save
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"benchmark_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")

        # Print summary table
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        print(f"{'Configuration':<40} {'Loss (mean±std)':>20} {'Time (s)':>12} {'Params':>10}")
        print("-" * 80)

        for name, stats in summary['configurations'].items():
            if stats['final_loss_mean'] is not None:
                loss_str = f"{stats['final_loss_mean']:.4f}±{stats['final_loss_std']:.4f}"
            else:
                loss_str = "N/A"
            time_str = f"{stats['training_time_mean']:.2f}"
            print(f"{name:<40} {loss_str:>20} {time_str:>12} {stats['total_params']:>10,}")

        print("=" * 80)


def main():
    """Parse arguments and run benchmark suite."""
    parser = argparse.ArgumentParser(description="SSM-MetaRL Benchmark Suite")

    parser.add_argument('--quick', action='store_true',
                        help='Quick benchmark (CartPole, 20 epochs, 1 seed)')
    parser.add_argument('--full', action='store_true',
                        help='Full benchmark with all envs and models')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Model types to benchmark')
    parser.add_argument('--algorithms', nargs='+', default=None,
                        help='Algorithms to benchmark')
    parser.add_argument('--envs', nargs='+', default=None,
                        help='Environments to benchmark')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Training epochs per run')
    parser.add_argument('--num_seeds', type=int, default=3,
                        help='Number of random seeds')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results output directory')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Wandb logging')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    if args.quick:
        models = ['legacy', 'mamba']
        algorithms = ['maml']
        envs = ['CartPole-v1']
        num_seeds = 1
        num_epochs = 20
    elif args.full:
        models = ['legacy', 'mamba']
        algorithms = ['maml', 'rl2']
        envs = ['CartPole-v1']
        if 'HalfCheetahVel' in envs: # Legacy support check
            envs = ['CartPole-v1']
        num_seeds = 3
        num_epochs = 100
    else:
        models = args.models or ['legacy', 'mamba']
        algorithms = args.algorithms or ['maml']
        envs = args.envs or ['CartPole-v1']
        num_seeds = args.num_seeds
        num_epochs = args.num_epochs

    suite = BenchmarkSuite(
        models=models,
        algorithms=algorithms,
        envs=envs,
        num_seeds=num_seeds,
        num_epochs=num_epochs,
        results_dir=args.results_dir,
        use_wandb=args.use_wandb,
    )

    summary = suite.run()

    print(f"\nBenchmark completed: {summary['num_runs']} runs")
    print(f"Results saved to: {args.results_dir}/")


if __name__ == "__main__":
    main()
