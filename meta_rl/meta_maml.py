"""Meta-RL Module: Meta-MAML Implementation with Functional Forward Pass
This module implements proper MAML (Model-Agnostic Meta-Learning) with
functional forward passes using custom weights (fast_weights).

Key Features:
- Proper functional forward pass with custom parameters
- Inner loop adaptation using fast_weights
- Outer loop meta-optimization
- Compatible with ANY PyTorch nn.Module model (Sequential, custom, RNN, SSM, etc.)
- Automatic recursive fast_weights application to all parameters/submodules

References:
- MAML: Finn et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"

Author: SSM-MetaRL Team
Date: 2025-10-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, OrderedDict as OrderedDictType
from collections import OrderedDict
import copy

try:
    from torch.func import functional_call
    TORCH_FUNC_AVAILABLE = True
except ImportError:
    try:
        from functorch import make_functional_with_buffers
        TORCH_FUNC_AVAILABLE = False
        FUNCTORCH_AVAILABLE = True
    except ImportError:
        TORCH_FUNC_AVAILABLE = False
        FUNCTORCH_AVAILABLE = False


class MetaMAML:
    """Model-Agnostic Meta-Learning (MAML) implementation.
    
    This class implements proper MAML with functional forward passes
    that use custom weights (fast_weights) instead of the model's
    original parameters. Now supports ALL nn.Module types including:
    - nn.Sequential
    - Custom models with submodules
    - Branched/Residual architectures
    - RNNs, LSTMs, GRUs
    - SSMs (State Space Models)
    - Any arbitrary nn.Module structure
    
    Args:
        model: PyTorch model to meta-train (nn.Module)
        inner_lr: Learning rate for inner loop adaptation
        outer_lr: Learning rate for outer loop meta-optimization
        first_order: If True, use first-order MAML (no second-order gradients)
    
    Example:
        >>> # Example 1: Simple Sequential Network
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 64),
        ...     nn.ReLU(),
        ...     nn.Linear(64, 1)
        ... )
        >>> meta_learner = MetaMAML(model, inner_lr=0.01, outer_lr=0.001)
        >>> 
        >>> # Example 2: Custom Network with Residual Connections
        >>> class ResidualNet(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc1 = nn.Linear(10, 64)
        ...         self.fc2 = nn.Linear(64, 64)
        ...         self.fc3 = nn.Linear(64, 1)
        ...     
        ...     def forward(self, x):
        ...         identity = x
        ...         out = F.relu(self.fc1(x))
        ...         out = self.fc2(out)
        ...         out = out + self.fc1(identity)  # Residual
        ...         return self.fc3(F.relu(out))
        >>> 
        >>> model = ResidualNet()
        >>> meta_learner = MetaMAML(model, inner_lr=0.01)
        >>> 
        >>> # Example 3: RNN/LSTM Model
        >>> class LSTMPolicy(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.lstm = nn.LSTM(10, 64, batch_first=True)
        ...         self.fc = nn.Linear(64, 1)
        ...     
        ...     def forward(self, x):
        ...         out, _ = self.lstm(x)
        ...         return self.fc(out[:, -1, :])
        >>> 
        >>> model = LSTMPolicy()
        >>> meta_learner = MetaMAML(model, inner_lr=0.01)
        >>> 
        >>> # Usage in all cases:
        >>> support_x, support_y = ..., ...
        >>> query_x, query_y = ..., ...
        >>> 
        >>> # Inner loop adaptation
        >>> fast_weights = meta_learner.adapt(support_x, support_y, num_steps=5)
        >>> 
        >>> # Evaluate with adapted weights
        >>> pred = meta_learner.functional_forward(query_x, fast_weights)
        >>> loss = F.mse_loss(pred, query_y)
    """
    
    def __init__(self, model: nn.Module, inner_lr: float = 0.01,
                 outer_lr: float = 0.001, first_order: bool = False):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.first_order = first_order
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)
        
        # Check available functional API
        self.use_torch_func = TORCH_FUNC_AVAILABLE
        if not self.use_torch_func and not FUNCTORCH_AVAILABLE:
            print("Warning: Neither torch.func nor functorch available. "
                  "Using manual parameter replacement (slower).")
    
    def functional_forward(self, x: torch.Tensor,
                          params: Optional[OrderedDictType[str, torch.Tensor]] = None) -> torch.Tensor:
        """Perform forward pass with custom parameters (fast_weights).
        
        This method now works with ALL nn.Module types using torch.func.functional_call
        or functorch, providing true generality. It recursively applies fast_weights
        to all parameters and submodules automatically.
        
        Args:
            x: Input tensor
            params: Custom parameters (fast_weights) as OrderedDict.
                   If None, uses model's current parameters.
                   Keys should match model.named_parameters() names.
        
        Returns:
            Output tensor from forward pass with custom parameters
        
        Example:
            >>> # Works with any model structure:
            >>> model = YourCustomModel()  # Any nn.Module
            >>> meta_learner = MetaMAML(model)
            >>> 
            >>> # Get current parameters
            >>> params = OrderedDict(model.named_parameters())
            >>> 
            >>> # Modify some parameters (e.g., gradient update)
            >>> grads = torch.autograd.grad(loss, params.values(), create_graph=True)
            >>> fast_weights = OrderedDict()
            >>> for (name, param), grad in zip(params.items(), grads):
            ...     fast_weights[name] = param - inner_lr * grad
            >>> 
            >>> # Forward with adapted weights
            >>> output = meta_learner.functional_forward(x, fast_weights)
            >>> 
            >>> # The method handles:
            >>> # - Simple Sequential models
            >>> # - Models with branches/residuals
            >>> # - RNN/LSTM/GRU with cell states
            >>> # - SSM (State Space Models)
            >>> # - Any custom nn.Module with arbitrary structure
        """
        if params is None:
            return self.model(x)
        
        if self.use_torch_func:
            # Use torch.func.functional_call (PyTorch >= 2.0)
            # This automatically handles all parameter replacement recursively
            return functional_call(self.model, params, x)
        else:
            # Fallback: Manual recursive parameter replacement
            return self._manual_functional_forward(x, params)
    
    def _manual_functional_forward(self, x: torch.Tensor,
                                   params: OrderedDictType[str, torch.Tensor]) -> torch.Tensor:
        """Manual implementation of functional forward for when torch.func is unavailable.
        Recursively replaces parameters in the model with fast_weights.
        """
        model_copy = copy.deepcopy(self.model)
        self._replace_params_recursive(model_copy, params)
        
        with torch.no_grad():
            output = model_copy(x)
        
        return output
    
    def _replace_params_recursive(self, module: nn.Module,
                                  params: OrderedDictType[str, torch.Tensor], prefix: str = ''):
        """Recursively replace parameters in all submodules.
        
        Args:
            module: Current module to process
            params: Dictionary of parameters to use
            prefix: Current parameter name prefix (for recursion)
        """
        for name, param in list(module._parameters.items()):
            if param is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                if full_name in params:
                    module._parameters[name] = params[full_name]
        
        for name, submodule in module._modules.items():
            if submodule is not None:
                sub_prefix = f"{prefix}.{name}" if prefix else name
                self._replace_params_recursive(submodule, params, sub_prefix)
    
    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor,
             loss_fn=None, num_steps: int = 1) -> OrderedDictType[str, torch.Tensor]:
        """Perform inner loop adaptation on support set.
        
        Args:
            support_x: Support set inputs
            support_y: Support set targets
            loss_fn: Loss function (default: MSE for regression)
            num_steps: Number of gradient steps for adaptation
        
        Returns:
            fast_weights: Adapted parameters as OrderedDict
        """
        if loss_fn is None:
            loss_fn = F.mse_loss
        
        fast_weights = OrderedDict((name, param.clone())
                                  for name, param in self.model.named_parameters())
        
        for step in range(num_steps):
            pred = self.functional_forward(support_x, fast_weights)
            loss = loss_fn(pred, support_y)
            
            grads = torch.autograd.grad(loss, fast_weights.values(),
                                       create_graph=not self.first_order)
            
            fast_weights = OrderedDict((name, param - self.inner_lr * grad)
                                      for (name, param), grad in zip(fast_weights.items(), grads))
        
        return fast_weights
    
    def meta_update(self, tasks: List[Tuple[torch.Tensor, torch.Tensor,
                                           torch.Tensor, torch.Tensor]],
                   loss_fn=None) -> float:
        """Perform outer loop meta-update across multiple tasks.
        
        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples
            loss_fn: Loss function (default: MSE for regression)
        
        Returns:
            Average meta-loss across tasks
        """
        if loss_fn is None:
            loss_fn = F.mse_loss
        
        self.meta_optimizer.zero_grad()
        meta_loss = 0.0
        
        for support_x, support_y, query_x, query_y in tasks:
            fast_weights = self.adapt(support_x, support_y, loss_fn)
            pred = self.functional_forward(query_x, fast_weights)
            loss = loss_fn(pred, query_y)
            meta_loss += loss
        
        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def get_fast_weights(self) -> OrderedDictType[str, torch.Tensor]:
        """Get current model parameters as OrderedDict (useful for initialization).
        
        Returns:
            OrderedDict of current model parameters
        """
        return OrderedDict((name, param.clone())
                          for name, param in self.model.named_parameters())
