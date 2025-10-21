"""Environment Runner Module for SSM-MetaRL

This module implements an RL environment runner using Gymnasium
environments for meta-RL tasks. It supports both single and batched
environments with proper reset/step functionality.

References:
- Gymnasium: https://github.com/Farama-Foundation/Gymnasium
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import gymnasium as gym
import warnings


class Environment:
    """Environment wrapper for Meta-RL tasks using Gymnasium.
    
    This class provides a unified interface for managing RL environments
    with actual gym.make() calls and proper reset/step functionality.
    
    Args:
        env_name: Name of the gym environment (e.g., 'CartPole-v1', 'Pendulum-v1')
        batch_size: Number of parallel environments (if > 1, creates multiple env instances)
        max_episode_steps: Maximum steps per episode (None uses env default)
        seed: Random seed for reproducibility
    
    Attributes:
        env_name: Name of the environment
        batch_size: Number of parallel environments
        envs: List of gym environment instances
        observation_space: Observation space of the environment
        action_space: Action space of the environment
    """
    
    def __init__(self,
                 env_name: str = 'CartPole-v1',
                 batch_size: int = 1,
                 max_episode_steps: Optional[int] = None,
                 seed: Optional[int] = None):
        
        self.env_name = env_name
        self.batch_size = batch_size
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        
        # Create gym environments
        self.envs = []
        for i in range(batch_size):
            try:
                # Use Gymnasium directly
                env = gym.make(env_name)
                
                # Set max episode steps if specified
                if max_episode_steps is not None:
                    env = gym.wrappers.TimeLimit(env, max_episode_steps)
                
                # Set seed
                if seed is not None:
                    env.reset(seed=seed + i)
                
                self.envs.append(env)
            except Exception as e:
                # If environment creation fails, create a simple placeholder
                warnings.warn(f"Failed to create {env_name}: {e}. Using placeholder.")
                env = self._create_placeholder_env()
                self.envs.append(env)
        
        # Store observation and action spaces from first environment
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        # Track episode statistics
        self._episode_returns = [0.0] * batch_size
        self._episode_lengths = [0] * batch_size
    
    def _create_placeholder_env(self):
        """Create a simple placeholder environment for testing."""
        class PlaceholderEnv:
            def __init__(self):
                self.observation_space = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float32
                )
                self.action_space = gym.spaces.Discrete(2)
                self.state = None
            
            def reset(self, seed=None):
                if seed is not None:
                    np.random.seed(seed)
                self.state = self.observation_space.sample()
                return self.state, {} # Gymnasium returns (obs, info)
            
            def step(self, action):
                next_state = self.observation_space.sample()
                reward = np.random.randn()
                done = np.random.rand() < 0.05  # 5% chance of episode end
                truncated = False
                info = {}
                self.state = next_state
                return next_state, reward, done, truncated, info
            
            def seed(self, seed):
                np.random.seed(seed)
            
            def render(self):
                pass
            
            def close(self):
                pass
        
        return PlaceholderEnv()
    
    def reset(self, task_id: Optional[int] = None) -> Union[np.ndarray, List[np.ndarray]]:
        """Reset all environments and return initial observations.
        
        Args:
            task_id: Optional task identifier (for multi-task settings)
            
        Returns:
            Initial observation(s). If batch_size=1, returns single observation.
            Otherwise returns list of observations.
        """
        observations = []
        
        for i, env in enumerate(self.envs):
            # Gymnasium returns (obs, info)
            obs, info = env.reset()
            
            observations.append(np.array(obs, dtype=np.float32))
            
            # Reset episode statistics
            self._episode_returns[i] = 0.0
            self._episode_lengths[i] = 0
        
        if self.batch_size == 1:
            return observations[0]
        return observations
    
    def step(self, actions: Union[int, float, np.ndarray, List]) -> Tuple[
        Union[np.ndarray, List[np.ndarray]],
        Union[float, List[float]],
        Union[bool, List[bool]],
        Union[Dict, List[Dict]]
    ]:
        """Execute actions in all environments.
        
        Args:
            actions: Action(s) to take. Can be single action (if batch_size=1)
                    or list of actions (if batch_size > 1)
        
        Returns:
            Tuple of (observations, rewards, dones, infos)
            If batch_size=1, returns single values. Otherwise returns lists.
        """
        # Convert single action to list for uniform processing
        if self.batch_size == 1 and not isinstance(actions, list):
            actions = [actions]
        
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            # Execute step
            # Gymnasium returns (obs, reward, done, truncated, info)
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated  # Combine done and truncated
            
            observations.append(np.array(obs, dtype=np.float32))
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(info)
            
            # Update episode statistics
            self._episode_returns[i] += reward
            self._episode_lengths[i] += 1
            
            # Add episode statistics to info when episode ends
            if done:
                info['episode_return'] = self._episode_returns[i]
                info['episode_length'] = self._episode_lengths[i]
        
        # Return single values if batch_size=1, otherwise return lists
        if self.batch_size == 1:
            return observations[0], rewards[0], dones[0], infos[0]
        return observations, rewards, dones, infos
    
    def sample_action(self) -> Union[int, float, np.ndarray, List]:
        """Sample random action(s) from action space.
        
        Returns:
            Random action(s). If batch_size=1, returns single action.
            Otherwise returns list of actions.
        """
        if self.batch_size == 1:
            return self.action_space.sample()
        return [env.action_space.sample() for env in self.envs]
    
    def render(self, mode: str = 'human') -> None:
        """Render the first environment.
        
        Args:
            mode: Render mode (e.g., 'human', 'rgb_array')
        """
        if self.envs:
            try:
                # Gymnasium render mode is often set at make()
                return self.envs[0].render()
            except:
                pass # Handle different render setups
    
    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            try:
                env.close()
            except:
                pass
    
    def get_episode_statistics(self) -> Dict[str, List[float]]:
        """Get current episode statistics.
        
        Returns:
            Dictionary with 'returns' and 'lengths' lists
        """
        return {
            'returns': self._episode_returns.copy(),
            'lengths': self._episode_lengths.copy()
        }
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()


class MetaEnvironment(Environment):
    """Meta-learning environment wrapper with task sampling.
    
    Extends Environment with task distribution support for meta-RL.
    
    Args:
        env_name: Base environment name
        num_tasks: Number of different tasks in the distribution
        batch_size: Number of parallel environments
        **kwargs: Additional arguments passed to Environment
    """
    
    def __init__(self,
                 env_name: str = 'CartPole-v1',
                 num_tasks: int = 5,
                 batch_size: int = 1,
                 **kwargs):
        super().__init__(env_name=env_name, batch_size=batch_size, **kwargs)
        
        self.num_tasks = num_tasks
        self.current_task_id = 0
    
    def sample_task(self) -> int:
        """Sample a random task ID.
        
        Returns:
            Task ID (integer between 0 and num_tasks-1)
        """
        self.current_task_id = np.random.randint(0, self.num_tasks)
        return self.current_task_id
    
    def reset(self, task_id: Optional[int] = None):
        """Reset with optional task ID.
        
        Args:
            task_id: Optional task ID to reset to. If None, uses current task.
        
        Returns:
            Initial observation(s)
        """
        if task_id is not None:
            self.current_task_id = task_id
        
        return super().reset(task_id=self.current_task_id)


if __name__ == "__main__":
    # Quick test
    print("Testing Environment module...")
    
    # Test basic environment
    print("\n1. Testing single environment:")
    env = Environment(env_name='CartPole-v1', batch_size=1)
    print(f"   Environment: {env.env_name}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Test reset
    obs = env.reset()
    print(f"   Reset observation shape: {obs.shape}")
    
    # Test step
    action = env.sample_action()
    next_obs, reward, done, info = env.step(action)
    print(f"   Step: action={action}, reward={reward:.2f}, done={done}")
    
    env.close()
    
    # Test batched environment
    print("\n2. Testing batched environment:")
    batch_env = Environment(env_name='CartPole-v1', batch_size=3)
    obs_list = batch_env.reset()
    print(f"   Batch size: {len(obs_list)}")
    print(f"   Each observation shape: {obs_list[0].shape}")
    
    actions = batch_env.sample_action()
    obs_list, rewards, dones, infos = batch_env.step(actions)
    print(f"   Batch step: {len(rewards)} rewards")
    
    batch_env.close()
    
    # Test meta environment
    print("\n3. Testing MetaEnvironment:")
    meta_env = MetaEnvironment(env_name='CartPole-v1', num_tasks=5, batch_size=1)
    task_id = meta_env.sample_task()
    print(f"   Sampled task ID: {task_id}")
    obs = meta_env.reset(task_id=task_id)
    print(f"   Reset observation shape: {obs.shape}")
    meta_env.close()
    
    print("\nEnvironment module test completed successfully!")
