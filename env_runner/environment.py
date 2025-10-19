"""Environment Runner Module for SSM-MetaRL

This module implements an RL environment runner for meta-RL tasks, following
Meta-World and RLlib interfaces. It supports batched multi-task training with
integration hooks for SSM-based policies.

References:
- Meta-World: https://github.com/Farama-Foundation/Metaworld
- RLlib: https://docs.ray.io/en/latest/rllib/index.html
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import gym
from gym import spaces


class Environment:
    """Base Environment class for Meta-RL tasks.
    
    This class provides a unified interface for managing RL environments
    in meta-learning settings, supporting batched operations and multi-task
    training scenarios.
    
    Attributes:
        env_name (str): Name of the environment
        task_id (int): Current task identifier
        observation_space (gym.Space): Observation space definition
        action_space (gym.Space): Action space definition
        batch_size (int): Number of parallel environments
    """
    
    def __init__(self, 
                 env_name: str,
                 batch_size: int = 1,
                 max_episode_steps: int = 200,
                 seed: Optional[int] = None):
        """Initialize the Environment.
        
        Args:
            env_name: Name of the environment to create
            batch_size: Number of parallel environment instances
            max_episode_steps: Maximum steps per episode
            seed: Random seed for reproducibility
        """
        self.env_name = env_name
        self.batch_size = batch_size
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        
        # Initialize environment-specific attributes
        self._current_step = 0
        self.task_id = 0
        self._done = False
        
        # Set up spaces (to be overridden by specific environments)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        if seed is not None:
            self.set_seed(seed)
    
    def reset(self, task_id: Optional[int] = None) -> np.ndarray:
        """Reset the environment to initial state.
        
        Args:
            task_id: Optional task identifier for multi-task settings
            
        Returns:
            Initial observation as numpy array
        """
        self._current_step = 0
        self._done = False
        
        if task_id is not None:
            self.task_id = task_id
        
        # Return initial observation (placeholder)
        observation = self.observation_space.sample()
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self._current_step += 1
        
        # Placeholder step logic
        observation = self.observation_space.sample()
        reward = 0.0
        done = self._current_step >= self.max_episode_steps
        info = {
            'task_id': self.task_id,
            'step': self._current_step
        }
        
        self._done = done
        return observation, reward, done, info
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility.
        
        Args:
            seed: Random seed value
        """
        self.seed = seed
        np.random.seed(seed)
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass


class BatchedEnvironment:
    """Batched environment wrapper for parallel execution.
    
    This class manages multiple environment instances for efficient
    batched operations during training.
    
    Attributes:
        envs (List[Environment]): List of environment instances
        batch_size (int): Number of parallel environments
    """
    
    def __init__(self, 
                 env_name: str,
                 batch_size: int,
                 max_episode_steps: int = 200,
                 seed: Optional[int] = None):
        """Initialize batched environments.
        
        Args:
            env_name: Name of the environment
            batch_size: Number of parallel environments
            max_episode_steps: Maximum steps per episode
            seed: Base random seed
        """
        self.env_name = env_name
        self.batch_size = batch_size
        
        # Create multiple environment instances
        self.envs = [
            Environment(env_name, batch_size=1, max_episode_steps=max_episode_steps,
                       seed=seed + i if seed is not None else None)
            for i in range(batch_size)
        ]
        
        # Use first environment's spaces as reference
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self, task_ids: Optional[List[int]] = None) -> np.ndarray:
        """Reset all environments in the batch.
        
        Args:
            task_ids: Optional list of task IDs for each environment
            
        Returns:
            Batched observations as numpy array of shape (batch_size, obs_dim)
        """
        if task_ids is None:
            task_ids = [None] * self.batch_size
        
        observations = []
        for env, task_id in zip(self.envs, task_ids):
            obs = env.reset(task_id=task_id)
            observations.append(obs)
        
        return np.stack(observations, axis=0)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Execute batched step across all environments.
        
        Args:
            actions: Batched actions of shape (batch_size, action_dim)
            
        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        observations, rewards, dones, infos = [], [], [], []
        
        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return (
            np.stack(observations, axis=0),
            np.array(rewards),
            np.array(dones),
            infos
        )
    
    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()


class SSMPolicyIntegration:
    """Integration hooks for SSM-based policies.
    
    This class provides utility functions for integrating SSM (State Space Model)
    based policies with the environment runner.
    """
    
    @staticmethod
    def prepare_observation_sequence(observations: np.ndarray,
                                    sequence_length: int) -> np.ndarray:
        """Prepare observation sequence for SSM input.
        
        Args:
            observations: Raw observations
            sequence_length: Length of sequence for SSM
            
        Returns:
            Formatted observation sequence
        """
        # Placeholder for sequence preparation logic
        return observations
    
    @staticmethod
    def extract_hidden_state(policy_output: Dict[str, Any]) -> np.ndarray:
        """Extract hidden state from SSM policy output.
        
        Args:
            policy_output: Output dictionary from SSM policy
            
        Returns:
            Extracted hidden state
        """
        return policy_output.get('hidden_state', np.array([]))
    
    @staticmethod
    def compute_meta_features(trajectory: List[Tuple]) -> np.ndarray:
        """Compute meta-learning features from trajectory.
        
        Args:
            trajectory: List of (obs, action, reward, next_obs) tuples
            
        Returns:
            Computed meta-features
        """
        # Placeholder for meta-feature computation
        return np.array([])


class MultiTaskEnvironment(Environment):
    """Multi-task environment supporting task switching.
    
    This class extends the base Environment to support multiple tasks,
    following the Meta-World benchmark structure.
    
    Attributes:
        tasks (List[int]): Available task identifiers
        current_task (int): Currently active task
    """
    
    def __init__(self,
                 env_name: str,
                 tasks: List[int],
                 batch_size: int = 1,
                 max_episode_steps: int = 200,
                 seed: Optional[int] = None):
        """Initialize multi-task environment.
        
        Args:
            env_name: Base environment name
            tasks: List of task identifiers
            batch_size: Number of parallel environments
            max_episode_steps: Maximum steps per episode
            seed: Random seed
        """
        super().__init__(env_name, batch_size, max_episode_steps, seed)
        self.tasks = tasks
        self.current_task = tasks[0] if tasks else 0
    
    def sample_task(self) -> int:
        """Sample a random task from available tasks.
        
        Returns:
            Sampled task identifier
        """
        return np.random.choice(self.tasks)
    
    def set_task(self, task_id: int) -> None:
        """Set the active task.
        
        Args:
            task_id: Task identifier to activate
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not in available tasks: {self.tasks}")
        self.current_task = task_id
        self.task_id = task_id
    
    def reset(self, task_id: Optional[int] = None) -> np.ndarray:
        """Reset environment with optional task specification.
        
        Args:
            task_id: Optional task to set before reset
            
        Returns:
            Initial observation
        """
        if task_id is not None:
            self.set_task(task_id)
        else:
            # Sample random task if not specified
            self.set_task(self.sample_task())
        
        return super().reset(task_id=self.current_task)


def create_environment(env_name: str,
                      batch_size: int = 1,
                      multi_task: bool = False,
                      tasks: Optional[List[int]] = None,
                      **kwargs) -> Environment:
    """Factory function to create environment instances.
    
    Args:
        env_name: Name of environment to create
        batch_size: Number of parallel environments
        multi_task: Whether to create multi-task environment
        tasks: List of task IDs for multi-task setting
        **kwargs: Additional environment parameters
        
    Returns:
        Environment instance (batched if batch_size > 1)
    """
    if multi_task:
        if tasks is None:
            tasks = list(range(10))  # Default to 10 tasks
        base_env = MultiTaskEnvironment(env_name, tasks=tasks, **kwargs)
    else:
        base_env = Environment(env_name, **kwargs)
    
    if batch_size > 1:
        return BatchedEnvironment(env_name, batch_size=batch_size, **kwargs)
    
    return base_env
