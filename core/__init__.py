"""Core module for State Space Models.

Exports:
    SSM: Legacy MLP-based neural state model (O(T·d²) complexity)
    StateSpaceModel: Alias for SSM (backward compatibility)
    MambaSSM: Mamba-based structured SSM (O(T·d) complexity)
"""
from .ssm import SSM, StateSpaceModel
from .ssm_mamba import MambaSSM

__all__ = ['SSM', 'StateSpaceModel', 'MambaSSM']
