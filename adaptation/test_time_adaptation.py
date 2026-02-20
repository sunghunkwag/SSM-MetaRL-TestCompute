"""
Test-Time Adaptation Module for SSM-MetaRL-TestCompute
(This is the corrected version with the new API and English comments,
matching the calls in main.py and test_adaptation.py)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

# AdaptationConfig (Required by main.py and tests/test_adaptation.py)
@dataclass
class AdaptationConfig:
    """Configuration for the Adapter."""
    learning_rate: float = 0.01
    num_steps: int = 5
    # (Other fields from main.py's argparse can be added here if needed)
    grad_clip_norm: Optional[float] = 1.0
    trust_region_eps: Optional[float] = None
    ema_decay: Optional[float] = None
    entropy_weight: Optional[float] = None
    max_steps_per_call: int = 5 # This is the internal step count


class Adapter:
    """
    Performs test-time adaptation.
    This simplified version matches the API expected by:
    - main.py
    - tests/test_adaptation.py
    - experiments/quick_benchmark.py
    """

    def __init__(self,
                 model: nn.Module,
                 config: AdaptationConfig,
                 device: str = 'cpu',
                 params_to_adapt: Optional[list] = None):
        
        if torch is None:
            raise RuntimeError("PyTorch is required for test-time adaptation")
        
        self.model = model
        self.config = config
        self.device = device
        
        # If specific params are provided, use them. 
        # Otherwise, use all model parameters.
        params = params_to_adapt if params_to_adapt is not None else self.model.parameters()
        
        self.optimizer = torch.optim.Adam(
            params, 
            lr=self.config.learning_rate
        )
        self.loss_fn = nn.MSELoss()

    def update_step(self,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    hidden_state: torch.Tensor
                    ) -> Tuple[float, int]:
        """
        Performs adaptation update steps.
        """
        self.model.train()
        current_loss = 0.0
        
        for step in range(self.config.num_steps):
            self.optimizer.zero_grad()
            output, next_hidden_state = self.model(x, hidden_state)
            loss = self.loss_fn(output, y)
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip_norm is not None:
                # Get parameters currently being optimized
                params_to_clip = []
                for group in self.optimizer.param_groups:
                    params_to_clip.extend(group['params'])
                torch.nn.utils.clip_grad_norm_(params_to_clip, self.config.grad_clip_norm)
            
            self.optimizer.step()
            hidden_state = next_hidden_state.detach() if next_hidden_state is not None else None
            current_loss = loss.item()

        return current_loss, self.config.num_steps

        # main.py 
        # expects a (loss, steps) tuple as a return value.
        return current_loss, self.config.num_steps
