# SSM-MetaRL-TestCompute

> **Research Note**: This is a benchmarking utility used to compare standard recurrent architectures (RNN, LSTM, GRU) against neural state models in a Meta-RL setting. It is **not** a novel state space model implementation.

## Abstract

This experimentation framework allows for the rapid comparison of different state parameterizations when combined with Model-Agnostic Meta-Learning (MAML).

The primary negative result from this work is that **unstructured neural state models (simple RNNs with residual connections) do not offer significant advantages over LSTMs or GRUs in low-dimensional Meta-RL tasks**, despite having similar parameter counts.

---

## Architecture Comparison

We compare standard recurrent baselines against a custom "Neural State Model" (an explicit residual RNN).

| Component | Our Neural State Model | Structured SSM (S4/Mamba) | Traditional RNN |
|-----------|------------------------|---------------------------|-----------------|
| Forward pass | O(T·d²) | O(T·d) or O(T log T) | O(T·d²) |
| Complexity | Quadratic | Linear/Log-Linear | Quadratic |
| Parallelizable | No | Yes | No |

**Conclusion**: The custom "State Space Model" implemented here is mathematically equivalent to a residual RNN and suffers from the same O(T·d²) bottleneck. It does not possess the efficient scaling properties of modern structured SSMs.

---

## Experimental Observations

Benchmarks on CartPole and Pendulum (Meta-RL adaptation):

| Model | Loss Reduction (adaptation) | Parameters | Observation |
|-------|-----------------------------|------------|-------------|
| LSTM | ~95% | ~76K | Stable baseline |
| GRU | ~94% | ~57K | Efficient baseline |
| **This Model** | ~92% | ~53K | **No significant advantage** |

## Usage
This code is provided for reference on implementing MAML with stateful PyTorch modules.

```bash
# Run the benchmark comparison
python experiments/quick_benchmark.py
```

## License
MIT License