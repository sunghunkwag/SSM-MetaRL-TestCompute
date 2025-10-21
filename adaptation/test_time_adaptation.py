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
                 device: str = 'cpu'):
        
        if torch is None:
            raise RuntimeError("PyTorch is required for test-time adaptation")
        
        self.model = model
        self.config = config
        self.device = device
        
        # To match the API calls in main.py and tests/test_adaptation.py,
        # we embed the optimizer and loss function.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        # The loss function (e.g., MSE) is used to compare output and target (y)
        self.loss_fn = nn.MSELoss()

    def update_step(self,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    hidden_state: torch.Tensor
                    ) -> Tuple[float, int]:
        """
        Performs adaptation update steps based on the API call from main.py.
        This function NOW correctly updates the hidden state internally
        across 'num_steps' iterations.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            y: Target tensor (batch_size, output_dim) (e.g., next_observation)
            hidden_state: Current hidden state (batch_size, state_dim)
            
        Returns:
            Tuple[float, int]:
                - loss (float): The loss value from the final adaptation step.
                - steps (int): The number of steps taken.
        """
        
        # Set the model to training mode for adaptation
        self.model.train()
        
        current_loss = 0.0
        
        # Loop for the number of steps defined in the config.
        # This fixes the "duplicate learning" problem by unrolling the state.
        for step in range(self.config.num_steps):
            
            # 1. Zero gradients
            self.optimizer.zero_grad()
            
            # 2. Forward pass (Call the SSM model)
            #    Use the current hidden_state to get the next state.
            output, next_hidden_state = self.model(x, hidden_state)
            
            # 3. Calculate loss (comparing model output to target 'y')
            loss = self.loss_fn(output, y)
            
            # 4. Backpropagation
            loss.backward()
            
            # 5. Optimizer step
            self.optimizer.step()
            
            # 6. ★CRITICAL★: Update the hidden_state for the next iteration.
            #    .detach() is used to stop gradients from flowing back in time.
            hidden_state = next_hidden_state.detach()
            
            current_loss = loss.item()

        # main.py 
        # expects a (loss, steps) tuple as a return value.
        return current_loss, self.config.num_steps
