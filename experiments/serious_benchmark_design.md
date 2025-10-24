# Serious Benchmark Suite Design for SSM-MetaRL

## Problem Statement

Current benchmarks (CartPole-v1, Pendulum-v1) are **toy problems** that don't prove the framework's value:
- **Low dimensional**: 4-8 state dimensions
- **Simple dynamics**: Linear/trivial physics
- **No SOTA comparisons**: Can't prove "why this combination is good"
- **No scaling validation**: Unclear if it works on real problems

## Benchmark Suite Design

### 1. High-Dimensional Continuous Control Tasks

#### MuJoCo Environments (via Gymnasium)
- **HalfCheetah-v4**: 17-dim state, 6-dim action
- **Ant-v4**: 27-dim state, 8-dim action  
- **Humanoid-v4**: 376-dim state, 17-dim action
- **Walker2d-v4**: 17-dim state, 6-dim action

#### DMControl Suite (via dm_control)
- **Quadruped-walk**: High-dimensional quadruped locomotion
- **Humanoid-stand**: Complex humanoid balance
- **Manipulator-bring_ball**: Robotic manipulation

**Why these?**
- Industry-standard benchmarks
- High-dimensional state/action spaces
- Complex, non-linear dynamics
- Used in SOTA meta-RL papers (PEARL, MAML, RL²)

---

### 2. Meta-Learning Task Distributions

Instead of single tasks, create **task distributions** for meta-learning:

#### Goal-Velocity Tasks
- **HalfCheetah-Vel**: Different target velocities [0.5, 1.0, 1.5, 2.0, 2.5]
- **Ant-Vel**: Different target velocities [0.3, 0.6, 0.9, 1.2, 1.5]

#### Goal-Direction Tasks
- **Ant-Dir**: Navigate to different goal directions [0°, 45°, 90°, 135°, 180°]
- **Walker-Dir**: Walk in different directions

#### Dynamics Variation Tasks
- **HalfCheetah-Gravity**: Different gravity values [0.5g, 0.75g, 1.0g, 1.25g, 1.5g]
- **Ant-Mass**: Different body mass multipliers [0.5x, 0.75x, 1.0x, 1.25x, 1.5x]

**Why task distributions?**
- Tests true meta-learning capability
- Matches PEARL/MAML evaluation protocol
- Shows adaptation speed across tasks

---

### 3. SOTA Baseline Comparisons

Compare SSM-MetaRL against established methods:

#### Meta-Learning Baselines
1. **MAML (Finn et al., 2017)**
   - Original model-agnostic meta-learning
   - Use MLP policy network
   - Same inner/outer learning rates

2. **PEARL (Rakelly et al., 2019)**
   - Off-policy meta-RL with context encoding
   - Uses SAC as base algorithm
   - Probabilistic task embeddings

3. **RL² (Duan et al., 2016)**
   - Recurrent policy with LSTM
   - Meta-learns through temporal context
   - Direct competitor to SSM approach

#### Sequence Model Baselines
4. **LSTM-MAML**
   - MAML with LSTM policy
   - Tests if SSM is better than LSTM

5. **Transformer-MAML**
   - MAML with Transformer policy
   - Tests SSM vs attention mechanism

6. **Vanilla MLP-MAML**
   - MAML with feedforward network
   - Baseline without sequence modeling

**Why these baselines?**
- PEARL/RL²: SOTA meta-RL methods
- LSTM/Transformer: Tests SSM advantage
- MLP: Shows value of sequence modeling

---

### 4. Evaluation Metrics

#### Primary Metrics
1. **Sample Efficiency**
   - Episodes to reach 80% of expert performance
   - Compare across all methods

2. **Adaptation Speed**
   - Performance after K gradient steps (K=1,3,5,10)
   - Shows fast adaptation capability

3. **Final Performance**
   - Average return after full training
   - Asymptotic performance comparison

4. **Generalization**
   - Performance on held-out tasks
   - Test on unseen task variations

#### Secondary Metrics
5. **Computational Efficiency**
   - Wall-clock time per episode
   - Memory usage
   - FLOPs per forward pass

6. **Stability**
   - Standard deviation across seeds
   - Convergence rate
   - Training stability

---

### 5. Experimental Protocol

#### Meta-Training Phase
```python
For each task distribution:
    1. Sample N tasks from distribution
    2. For each task:
        - Collect support set (K episodes)
        - Perform inner loop adaptation
        - Collect query set (K episodes)
        - Compute meta-loss
    3. Update meta-parameters
    4. Log metrics every M steps
```

#### Meta-Testing Phase
```python
For each held-out task:
    1. Start with meta-learned parameters
    2. Collect adaptation data (K episodes)
    3. Perform K gradient steps
    4. Evaluate on test episodes
    5. Measure: return, success rate, adaptation curve
```

#### Hyperparameters
- **Meta-batch size**: 20 tasks
- **Inner steps**: 5
- **Outer steps**: 1000
- **Support episodes**: 10
- **Query episodes**: 10
- **Seeds**: 5 random seeds per experiment

---

### 6. Implementation Plan

#### Phase 1: Environment Setup
```python
# Install dependencies
pip install gymnasium[mujoco] dm_control

# Create task distribution wrappers
class VelocityTask:
    def __init__(self, env_name, target_velocity):
        self.env = gym.make(env_name)
        self.target_velocity = target_velocity
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Modify reward based on velocity target
        velocity = info['x_velocity']
        reward = -abs(velocity - self.target_velocity)
        return obs, reward, done, info
```

#### Phase 2: Baseline Implementations
```python
# LSTM-MAML baseline
class LSTMPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden

# Use with MetaMAML
lstm_policy = LSTMPolicy(state_dim, action_dim)
lstm_maml = MetaMAML(lstm_policy, inner_lr=0.01, outer_lr=0.001)
```

#### Phase 3: Evaluation Framework
```python
class BenchmarkSuite:
    def __init__(self, methods, tasks, metrics):
        self.methods = methods
        self.tasks = tasks
        self.metrics = metrics
    
    def run(self):
        results = {}
        for method in self.methods:
            for task in self.tasks:
                # Meta-train
                meta_train(method, task)
                # Meta-test
                results[method][task] = meta_test(method, task)
        return results
    
    def plot_results(self, results):
        # Generate comparison plots
        plot_sample_efficiency(results)
        plot_adaptation_curves(results)
        plot_final_performance(results)
```

---

### 7. Expected Outcomes

#### Hypothesis 1: SSM > LSTM
**Claim**: SSM-MetaRL should outperform LSTM-MAML due to:
- Linear-time complexity vs quadratic
- Better long-range dependencies
- More efficient gradient flow

**Test**: Compare on HalfCheetah-Vel with long episodes (1000 steps)

#### Hypothesis 2: SSM Competitive with PEARL
**Claim**: SSM-MetaRL should match or exceed PEARL on:
- Sample efficiency (fewer episodes needed)
- Adaptation speed (faster convergence)

**Test**: Compare on Ant-Dir task distribution

#### Hypothesis 3: Scales to High Dimensions
**Claim**: SSM-MetaRL should maintain performance on Humanoid-v4 (376-dim state)

**Test**: Compare all methods on Humanoid-stand task

---

### 8. Deliverables

1. **Benchmark Code**
   - `experiments/serious_benchmark.py`
   - Task distribution implementations
   - Baseline method implementations

2. **Results**
   - Performance tables (CSV)
   - Comparison plots (PNG)
   - Statistical significance tests

3. **Documentation**
   - Benchmark README
   - Reproduction instructions
   - Hyperparameter configurations

4. **Paper-Ready Figures**
   - Learning curves
   - Adaptation curves
   - Bar charts with error bars
   - Ablation studies

---

## Timeline

- **Phase 1** (2 hours): Environment setup + task distributions
- **Phase 2** (3 hours): Baseline implementations
- **Phase 3** (2 hours): Evaluation framework
- **Phase 4** (4 hours): Run experiments (can parallelize)
- **Phase 5** (1 hour): Generate plots and documentation

**Total**: ~12 hours of compute time (can run overnight)

---

## Success Criteria

✅ **Minimum**: SSM-MetaRL outperforms vanilla MLP-MAML on at least 3/5 tasks
✅ **Target**: SSM-MetaRL competitive with LSTM-MAML (within 10% performance)
✅ **Stretch**: SSM-MetaRL outperforms LSTM-MAML on sample efficiency or adaptation speed

This would provide **strong evidence** that the SSM+MetaRL+Adaptation combination is valuable beyond toy problems.

