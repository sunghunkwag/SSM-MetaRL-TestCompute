"""Meta-RL Module: Meta-MAML Implementation with Functional Forward Pass
... (docstring comments) ...
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, OrderedDict as OrderedDictType, Union
from collections import OrderedDict

# Ensure torch.func is available
try:
    from torch.func import functional_call
except ImportError:
    print("Warning: torch.func not found. MAML requires PyTorch >= 2.0.")
    functional_call = None

class MetaMAML:
    """Model-Agnostic Meta-Learning (MAML) implementation.
    ... (Args documentation) ...

    Now handles models with explicit state passing (e.g., RNNs, SSMs where
    forward returns (output, next_state)). Assumes the state is the second
    argument to forward and the second element in the return tuple.
    """

    def __init__(self, model: nn.Module, inner_lr: float = 0.01,
                 outer_lr: float = 0.001, first_order: bool = False):
        if functional_call is None:
            raise ImportError("MetaMAML requires torch.func (PyTorch >= 2.0).")
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.first_order = first_order
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)

        # Check if model requires hidden state
        import inspect
        sig = inspect.signature(self.model.forward)
        self._stateful = 'hidden_state' in sig.parameters

    def functional_forward(self, x: torch.Tensor,
                           hidden_state: Optional[torch.Tensor],
                           params: Optional[OrderedDictType[str, torch.Tensor]] = None
                           ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Perform forward pass with custom parameters (fast_weights).
        Handles both stateless and stateful models.

        Args:
            x: Input tensor
            hidden_state: Current hidden state (or None if stateless)
            params: Custom parameters (fast_weights) as OrderedDict.
                    If None, uses model's current parameters.

        Returns:
            - If stateless: Output tensor
            - If stateful: Tuple (Output tensor, Next hidden state tensor)
        """
        model_to_call = self.model
        args = (x,)
        if self._stateful:
            if hidden_state is None:
                raise ValueError("hidden_state must be provided for stateful models.")
            args = (x, hidden_state)

        if params is None:
            # Use model directly if no custom params
            return model_to_call(*args)
        else:
            # Use functional_call with custom params
            return functional_call(model_to_call, params, args)

    def adapt_task(self, support_x: torch.Tensor, support_y: torch.Tensor,
                   initial_hidden_state: Optional[torch.Tensor] = None,
                   loss_fn=None, num_steps: int = 1
                   ) -> OrderedDictType[str, torch.Tensor]:
        """Perform inner loop adaptation on support set. Handles state propagation.

        Args:
            support_x: Support set inputs (Batch, InputDim) or (Batch, Time, InputDim)
            support_y: Support set targets (Batch, OutputDim) or (Batch, Time, OutputDim)
            initial_hidden_state: Initial hidden state for stateful models. Required if model is stateful.
            loss_fn: Loss function (default: MSE for regression)
            num_steps: Number of gradient steps for adaptation

        Returns:
            fast_weights: Adapted parameters as OrderedDict
        """
        if self._stateful and initial_hidden_state is None:
             raise ValueError("initial_hidden_state must be provided for stateful models.")

        if loss_fn is None:
            loss_fn = F.mse_loss

        fast_weights = OrderedDict((name, param.clone())
                                   for name, param in self.model.named_parameters())

        # Determine if data has a time dimension
        time_dim_present = support_x.ndim == 3

        for step in range(num_steps):
            hidden_state = initial_hidden_state
            step_loss = 0.0

            if time_dim_present:
                # Process sequence step by step if time dimension exists
                T = support_x.shape[1]
                outputs = []
                for t in range(T):
                    x_t = support_x[:, t, :]
                    output_t, hidden_state = self.functional_forward(x_t, hidden_state, fast_weights)
                    outputs.append(output_t)
                # Stack outputs along time dimension: (Batch, Time, OutputDim)
                pred = torch.stack(outputs, dim=1)
                # Compute loss over the sequence
                step_loss = loss_fn(pred, support_y)

            else:
                 # Process as a single step if no time dimension
                 if self._stateful:
                     pred, _ = self.functional_forward(support_x, hidden_state, fast_weights)
                 else:
                     pred = self.functional_forward(support_x, None, fast_weights)
                 step_loss = loss_fn(pred, support_y)


            grads = torch.autograd.grad(step_loss, fast_weights.values(),
                                        create_graph=not self.first_order)

            fast_weights = OrderedDict((name, param - self.inner_lr * grad)
                                       for (name, param), grad in zip(fast_weights.items(), grads))

        return fast_weights

    def meta_update(self, tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                   initial_hidden_state: Optional[torch.Tensor] = None, # Assume same init state for all tasks for simplicity
                   loss_fn=None) -> float:
        """Perform outer loop meta-update across multiple tasks. Handles state.

        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples
            initial_hidden_state: Initial hidden state for stateful models.
            loss_fn: Loss function (default: MSE for regression)

        Returns:
            Average meta-loss across tasks
        """
        if self._stateful and initial_hidden_state is None:
             raise ValueError("initial_hidden_state must be provided for stateful models.")

        if loss_fn is None:
            loss_fn = F.mse_loss

        self.meta_optimizer.zero_grad()
        meta_loss = 0.0

        for support_x, support_y, query_x, query_y in tasks:
            # Adapt on support set
            fast_weights = self.adapt_task(support_x, support_y, initial_hidden_state, loss_fn)

            # Evaluate on query set using adapted weights
            hidden_state = initial_hidden_state
            query_loss = 0.0
            time_dim_present = query_x.ndim == 3

            if time_dim_present:
                 T = query_x.shape[1]
                 outputs = []
                 for t in range(T):
                     x_t = query_x[:, t, :]
                     output_t, hidden_state = self.functional_forward(x_t, hidden_state, fast_weights)
                     outputs.append(output_t)
                 pred = torch.stack(outputs, dim=1)
                 query_loss = loss_fn(pred, query_y)
            else:
                 if self._stateful:
                     pred, _ = self.functional_forward(query_x, hidden_state, fast_weights)
                 else:
                     pred = self.functional_forward(query_x, None, fast_weights)
                 query_loss = loss_fn(pred, query_y)

            meta_loss += query_loss

        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def get_fast_weights(self) -> OrderedDictType[str, torch.Tensor]:
        """Get current model parameters as OrderedDict (useful for initialization)."""
        return OrderedDict((name, param.clone())
                           for name, param in self.model.named_parameters())
