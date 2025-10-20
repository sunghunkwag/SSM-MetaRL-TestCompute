# SSM-MetaRL-TestCompute
A minimal Python package for State Space Models (SSM) with Meta Reinforcement Learning and Test-Time Adaptation.

## Installation

### Install from GitHub
You can install the package directly from this GitHub repository:
```bash
pip install git+https://github.com/sunghunkwag/SSM-MetaRL-TestCompute.git
```

### Install for Development
For development, clone the repository and install in editable mode:
```bash
git clone https://github.com/sunghunkwag/SSM-MetaRL-TestCompute.git
cd SSM-MetaRL-TestCompute
pip install -e .
```

### Build and Install from Wheel
To build the package as a wheel:
```bash
pip install build
python -m build
pip install dist/ssm_metarl-0.1.0-py3-none-any.whl
```

## Usage

### Quick Start: Running the Main Script

The main entrypoint is `main.py`, which demonstrates the full workflow:

```bash
python main.py
```

This script:
1. Creates a gym environment (default: 'CartPole-v1')
2. Initializes an SSM with state/action dimensions matching the environment
3. Trains the SSM policy using MetaMAML across multiple tasks
4. Performs test-time adaptation on new tasks
5. Evaluates final performance

### Import the Package

```python
import torch
from core.ssm import SSM
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter

# Initialize State Space Model
# SSM requires state_dim, action_dim, and hidden_dim
state_dim = 4  # e.g., CartPole observation space
action_dim = 2  # e.g., CartPole action space
hidden_dim = 64

model = SSM(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

# Forward pass through SSM
observation = torch.randn(1, state_dim)
action_logits = model(observation)  # Returns action logits

# Sample action from categorical distribution
action_dist = torch.distributions.Categorical(logits=action_logits)
action = action_dist.sample()
```

### Meta-Learning with MetaMAML

```python
import torch
import torch.nn as nn
from meta_rl.meta_maml import MetaMAML
from core.ssm import SSM

# Initialize SSM model
model = SSM(state_dim=4, action_dim=2, hidden_dim=64)

# Initialize MetaMAML learner
meta_lr = 0.001
inner_lr = 0.01
inner_steps = 5

meta_learner = MetaMAML(
    model=model,
    meta_lr=meta_lr,
    inner_lr=inner_lr,
    inner_steps=inner_steps
)

# Prepare task batch
# Each task contains support and query sets
# tasks = [(support_states, support_actions, support_rewards, 
#           query_states, query_actions, query_rewards), ...]

# Run one meta-training step
meta_loss = meta_learner.outer_step(tasks)
print(f"Meta-training loss: {meta_loss.item()}")
```

### Test-Time Adaptation

```python
import gym
from adaptation.test_time_adaptation import Adapter
from core.ssm import SSM

# Create environment
env = gym.make('CartPole-v1')

# Initialize pre-trained SSM model
model = SSM(state_dim=4, action_dim=2, hidden_dim=64)
# Load pre-trained weights if available
# model.load_state_dict(torch.load('model.pth'))

# Initialize adapter
adapt_lr = 0.01
adapt_steps = 50

adapter = Adapter(learning_rate=adapt_lr)

# Adapt the policy to the current environment
adapted_model = adapter.adapt(
    policy=model,
    env=env,
    steps=adapt_steps
)

# Use adapted model for evaluation
obs = env.reset()
for _ in range(100):
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        action_logits = adapted_model(obs_tensor)
    action = torch.argmax(action_logits, dim=-1).item()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

### Complete Workflow Example

Here's a complete example matching `main.py`:

```python
import gym
import torch
from core.ssm import SSM
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter

def main():
    # Environment setup
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Model initialization
    model = SSM(state_dim=state_dim, action_dim=action_dim, hidden_dim=64)
    
    # Meta-learning phase
    print("Starting meta-training...")
    meta_learner = MetaMAML(
        model=model,
        meta_lr=0.001,
        inner_lr=0.01,
        inner_steps=5
    )
    
    # Generate tasks and train
    num_meta_iterations = 100
    for iteration in range(num_meta_iterations):
        tasks = generate_task_batch(env, num_tasks=8)
        meta_loss = meta_learner.outer_step(tasks)
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Meta Loss: {meta_loss.item():.4f}")
    
    # Test-time adaptation phase
    print("\nStarting test-time adaptation...")
    adapter = Adapter(learning_rate=0.01)
    adapted_model = adapter.adapt(
        policy=model,
        env=env,
        steps=50
    )
    
    # Evaluation phase
    print("\nEvaluating adapted model...")
    total_reward = 0
    num_episodes = 10
    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0
        done = False
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action_logits = adapted_model(obs_tensor)
            action = torch.argmax(action_logits, dim=-1).item()
            obs, reward, done, info = env.step(action)
            ep_reward += reward
        total_reward += ep_reward
    
    avg_reward = total_reward / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    env.close()

if __name__ == '__main__':
    main()
```

## Structure

The package is organized into four main components:

- **core/**: Core SSM implementation
  - `ssm.py`: State Space Model (nn.Module with Linear layers)
- **meta_rl/**: Meta-learning algorithms
  - `meta_maml.py`: MetaMAML implementation with inner/outer loop optimization
- **adaptation/**: Test-time adaptation
  - `test_time_adaptation.py`: Adapter class for online policy adaptation
- **env_runner/**: Environment utilities
  - `environment.py`: Gym environment wrapper and utilities

## API Reference

### SSM(state_dim, action_dim, hidden_dim)

**Parameters:**
- `state_dim` (int): Dimension of state/observation space
- `action_dim` (int): Dimension of action space
- `hidden_dim` (int): Hidden layer dimension (default: 64)

**Methods:**
- `forward(state)`: Returns action logits for given state tensor

### MetaMAML(model, meta_lr, inner_lr, inner_steps)

**Parameters:**
- `model` (nn.Module): The policy model to meta-train
- `meta_lr` (float): Meta-level learning rate
- `inner_lr` (float): Task-level adaptation learning rate
- `inner_steps` (int): Number of gradient steps per task adaptation

**Methods:**
- `outer_step(tasks)`: Performs one meta-training iteration across task batch
  - `tasks`: List of (support_data, query_data) tuples
  - Returns meta_loss tensor

### Adapter(learning_rate)

**Parameters:**
- `learning_rate` (float): Learning rate for test-time adaptation

**Methods:**
- `adapt(policy, env, steps)`: Adapts policy to environment
  - `policy` (nn.Module): The policy to adapt
  - `env`: Gym environment
  - `steps` (int): Number of adaptation steps
  - Returns adapted policy model

## License

MIT License
