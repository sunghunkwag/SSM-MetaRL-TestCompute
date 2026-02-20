"""Meta Reinforcement Learning module.

Exports:
    MetaMAML: Model-Agnostic Meta-Learning
    RL2Policy, RL2Trainer: RL² (Learning to Reinforcement Learn)
    PEARLAgent: Probabilistic Embeddings for Actor-critic RL
    VariBADAgent: Variational Bayes-Adaptive Deep RL
"""

from .meta_maml import MetaMAML
from .rl2 import RL2Policy, RL2Trainer
from .pearl import PEARLAgent
from .varibad import VariBADAgent

__all__ = ['MetaMAML', 'RL2Policy', 'RL2Trainer', 'PEARLAgent', 'VariBADAgent']

