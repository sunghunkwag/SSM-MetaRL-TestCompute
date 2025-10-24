# Serious Benchmark Suite for SSM-MetaRL

This directory contains a comprehensive benchmark suite that goes **beyond toy problems** to prove the effectiveness of the SSM-MetaRL framework on high-dimensional, complex tasks.

## 🎯 Motivation

The original benchmarks (CartPole-v1, Pendulum-v1) are **toy problems**:
- Low dimensional (4-8 state dimensions)
- Simple dynamics (linear/trivial physics)
- No SOTA comparisons
- No scaling validation

This benchmark suite addresses these limitations with:
- ✅ **High-dimensional tasks**: Up to 376-dim state spaces (Humanoid)
- ✅ **Complex dynamics**: MuJoCo physics simulation
- ✅ **SOTA baselines**: LSTM, GRU, Transformer, MLP comparisons
- ✅ **Meta-learning protocols**: Task distributions for true meta-RL evaluation
- ✅ **Proper metrics**: Sample efficiency, adaptation speed, generalization

---

## 📁 Files

### Core Modules

- **`task_distributions.py`**: High-dimensional task distribution implementations
  - HalfCheetah-Vel (17-dim state, 6-dim action)
  - Ant-Vel, Ant-Dir (27-dim state, 8-dim action)
  - Walker2d-Vel (17-dim state, 6-dim action)
  - Gravity/Mass variation tasks

- **`baselines.py`**: SOTA baseline method implementations
  - MLP Policy (no sequence modeling)
  - LSTM Policy (quadratic complexity)
  - GRU Policy (lighter RNN)
  - Transformer Policy (attention-based)

- **`serious_benchmark.py`**: Main benchmark runner
  - Meta-training loop
  - Meta-testing with adaptation
  - Metrics collection
  - Results saving

- **`visualize_results.py`**: Visualization tools
  - Training curves
  - Test performance comparison
  - Adaptation curves
  - Summary tables

### Documentation

- **`serious_benchmark_design.md`**: Detailed design document
  - Problem statement
  - Benchmark architecture
  - Evaluation metrics
  - Expected outcomes

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install 'gymnasium[mujoco]' matplotlib
```

### 2. Run a Single Benchmark

```bash
# Test SSM on HalfCheetah-Vel
python experiments/serious_benchmark.py --task halfcheetah-vel --method ssm --epochs 50

# Test LSTM baseline
python experiments/serious_benchmark.py --task ant-vel --method lstm --epochs 50

# Run all methods
python experiments/serious_benchmark.py --task walker2d-vel --method all --epochs 100
```

### 3. Visualize Results

```bash
python experiments/visualize_results.py --results-dir results --output-dir figures
```

---

## 📊 Available Task Distributions

| Task Distribution | Base Environment | State Dim | Action Dim | Variation |
|-------------------|------------------|-----------|------------|-----------|
| `halfcheetah-vel` | HalfCheetah-v4 | 17 | 6 | Target velocities [0.5, 1.0, 1.5, 2.0, 2.5] |
| `ant-vel` | Ant-v4 | 27 | 8 | Target velocities [0.3, 0.6, 0.9, 1.2, 1.5] |
| `ant-dir` | Ant-v4 | 27 | 8 | Target directions [0°, 45°, 90°, 135°, 180°] |
| `walker2d-vel` | Walker2d-v4 | 17 | 6 | Target velocities [0.5, 1.0, 1.5, 2.0, 2.5] |
| `halfcheetah-gravity` | HalfCheetah-v4 | 17 | 6 | Gravity multipliers [0.5, 0.75, 1.0, 1.25, 1.5] |
| `ant-mass` | Ant-v4 | 27 | 8 | Mass multipliers [0.5, 0.75, 1.0, 1.25, 1.5] |

---

## 🔬 Baseline Methods

| Method | Description | Parameters | Complexity |
|--------|-------------|------------|------------|
| **SSM** | State Space Model | ~53K | O(n) - Linear |
| **LSTM** | Long Short-Term Memory | ~76K | O(n²) - Quadratic |
| **GRU** | Gated Recurrent Unit | ~57K | O(n²) - Quadratic |
| **Transformer** | Self-Attention | ~400K | O(n²) - Quadratic |
| **MLP** | Feedforward Network | ~20K | O(1) - No sequence |

*Parameters shown for HalfCheetah-v4 (17-dim state, 6-dim action, 128 hidden dim)*

---

## 📈 Evaluation Metrics

### Primary Metrics

1. **Sample Efficiency**
   - Episodes required to reach 80% of expert performance
   - Lower is better

2. **Adaptation Speed**
   - Performance after K gradient steps (K=1,3,5,10)
   - Shows fast adaptation capability

3. **Final Performance**
   - Average return after full meta-training
   - Asymptotic performance comparison

4. **Generalization**
   - Performance on held-out tasks
   - Tests meta-learning effectiveness

### Secondary Metrics

5. **Computational Efficiency**
   - Wall-clock time per epoch
   - Memory usage
   - Model parameter count

6. **Stability**
   - Standard deviation across random seeds
   - Convergence rate

---

## 🎯 Expected Outcomes

### Hypothesis 1: SSM > LSTM
**Claim**: SSM-MetaRL should outperform LSTM-MAML due to:
- Linear-time complexity vs quadratic
- Better long-range dependencies
- More efficient gradient flow

**Test**: Compare on HalfCheetah-Vel with long episodes (1000 steps)

### Hypothesis 2: SSM Scales to High Dimensions
**Claim**: SSM-MetaRL should maintain performance on high-dimensional tasks

**Test**: Compare all methods on Ant-v4 (27-dim) and Humanoid-v4 (376-dim)

### Hypothesis 3: Fast Adaptation
**Claim**: Meta-learned SSM should adapt faster than baselines

**Test**: Measure performance after 1, 3, 5, 10 gradient steps on held-out tasks

---

## 📝 Example Usage

### Run Complete Benchmark Suite

```bash
#!/bin/bash
# Run all methods on all tasks

TASKS="halfcheetah-vel ant-vel ant-dir walker2d-vel"
METHODS="ssm lstm gru mlp"
EPOCHS=100

for task in $TASKS; do
    for method in $METHODS; do
        echo "Running $method on $task..."
        python experiments/serious_benchmark.py \
            --task $task \
            --method $method \
            --epochs $EPOCHS \
            --hidden-dim 128 \
            --device cpu \
            --output-dir results/$task
    done
done

# Generate visualizations
python experiments/visualize_results.py \
    --results-dir results \
    --output-dir figures
```

### Custom Task Distribution

```python
from experiments.task_distributions import TaskDistribution, VelocityTask
import gymnasium as gym

class CustomVelDistribution(TaskDistribution):
    def __init__(self):
        velocities = [1.0, 2.0, 3.0]  # Custom velocities
        super().__init__('HalfCheetah-v4', velocities)
    
    def create_task(self, task_id: int):
        env = gym.make(self.env_name)
        return VelocityTask(env, self.task_params[task_id])

# Use in benchmark
dist = CustomVelDistribution()
task = dist.sample_task(0)
```

---

## 🔍 Results Interpretation

### Training Loss
- **Decreasing**: Model is learning to predict next states
- **Plateau**: May need more epochs or higher learning rate
- **Increasing**: Learning rate too high or instability

### Test Reward
- **Positive**: Model successfully controls the agent
- **Negative**: Model struggles with the task
- **Compare across methods**: Shows relative effectiveness

### Adaptation Curve
- **Steep drop**: Fast adaptation (good meta-learning)
- **Gradual drop**: Slow adaptation
- **No drop**: Poor meta-learning or task mismatch

---

## 🐛 Troubleshooting

### MuJoCo Installation Issues

```bash
# Install MuJoCo dependencies
pip install 'gymnasium[mujoco]'

# If you get "mujoco not found" error:
pip install mujoco
```

### Out of Memory

```bash
# Reduce batch size or hidden dimension
python experiments/serious_benchmark.py \
    --task halfcheetah-vel \
    --method ssm \
    --hidden-dim 64  # Smaller hidden dim
```

### Slow Training

```bash
# Use GPU if available
python experiments/serious_benchmark.py \
    --task ant-vel \
    --method ssm \
    --device cuda
```

---

## 📚 References

1. **MAML**: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", ICML 2017
2. **PEARL**: Rakelly et al., "Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables", ICML 2019
3. **RL²**: Duan et al., "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning", arXiv 2016
4. **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", arXiv 2023
5. **MuJoCo**: Todorov et al., "MuJoCo: A physics engine for model-based control", IROS 2012

---

## 🤝 Contributing

To add a new task distribution:

1. Create a new class in `task_distributions.py`
2. Inherit from `TaskDistribution`
3. Implement `create_task(task_id)`
4. Register in `TASK_DISTRIBUTIONS` dict

To add a new baseline:

1. Create a new policy class in `baselines.py`
2. Implement `forward()` and optionally `init_hidden()`
3. Register in `BASELINE_POLICIES` dict

---

## 📄 License

MIT License - See main repository LICENSE file

---

## 🙏 Acknowledgments

This benchmark suite builds upon:
- Gymnasium/MuJoCo for physics simulation
- MAML framework for meta-learning
- State Space Models (S4/Mamba) for sequence modeling

**Note**: This is a research benchmark. Results may vary based on hyperparameters, random seeds, and hardware.

