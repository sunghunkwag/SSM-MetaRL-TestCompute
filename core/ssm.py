"""Core SSM Module: PyTorch State Space Model Implementation

This module provides a PyTorch-based State Space Model (SSM) class
for meta-reinforcement learning applications. The SSM includes:
- Linear state transition dynamics
- Neural network-based processing
- torch.nn.Module integration
- Model persistence (torch.save/torch.load)
- GPU support

Author: SSM-MetaRL Team
Date: 2025-10-20
"""

import torch
import torch.nn as nn
import os
from typing import Tuple, Optional, Dict, Any


class SSM(nn.Module):
    """PyTorch State Space Model with neural network components.
    
    This is a PyTorch nn.Module implementation of a State Space Model
    suitable for meta-reinforcement learning tasks.
    
    Args:
        state_dim: Dimension of the hidden state
        hidden_dim: Dimension of intermediate hidden layers
        output_dim: Dimension of the output
        device: Device to run the model on ('cpu' or 'cuda')
    """
    
    def __init__(self, 
                 state_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 32,
                 device: str = 'cpu'):
        super(SSM, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        
        # State transition network (A matrix)
        self.state_transition = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Input projection network (B matrix)
        self.input_projection = nn.Linear(state_dim, state_dim)
        
        # Output network (C matrix)
        self.output_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Direct feedthrough (D matrix) - optional
        self.feedthrough = nn.Linear(state_dim, output_dim)
        
        # Initialize hidden state
        self.hidden_state = None
        
        # Move model to device
        self.to(device)
    
    def reset(self, batch_size: int = 1) -> torch.Tensor:
        """Reset the hidden state of the SSM.
        
        Args:
            batch_size: Batch size for the hidden state
            
        Returns:
            Initial hidden state tensor
        """
        self.hidden_state = torch.zeros(batch_size, self.state_dim, device=self.device)
        return self.hidden_state
    
    def forward(self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the SSM.
        
        Args:
            x: Input tensor of shape (batch_size, state_dim)
            hidden_state: Optional hidden state. If None, uses self.hidden_state
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if hidden_state is None:
            if self.hidden_state is None:
                self.reset(x.shape[0])
            hidden_state = self.hidden_state
        
        # State transition: h_t = A @ h_{t-1} + B @ u_t
        state_update = self.state_transition(hidden_state)
        input_update = self.input_projection(x)
        new_hidden = state_update + input_update
        
        # Update hidden state
        self.hidden_state = new_hidden
        
        # Output: y_t = C @ h_t + D @ u_t
        output = self.output_network(new_hidden)
        feedthrough_output = self.feedthrough(x)
        final_output = output + feedthrough_output
        
        return final_output
    
    def save(self, path: str) -> None:
        """Save model parameters using torch.save.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save state dict
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'device': self.device
            }
        }, path)
    
    @staticmethod
    def load(path: str, device: Optional[str] = None) -> 'SSM':
        """Load model parameters using torch.load.
        
        Args:
            path: Path to load the model from
            device: Device to load the model on. If None, uses saved device
            
        Returns:
            Loaded SSM model
        """
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        
        # Override device if specified
        if device is not None:
            config['device'] = device
        
        # Create model
        model = SSM(**config)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(config['device'])
        
        return model
    
    def get_hidden_state(self) -> Optional[torch.Tensor]:
        """Get current hidden state.
        
        Returns:
            Current hidden state tensor or None
        """
        return self.hidden_state
    
    def set_hidden_state(self, hidden_state: torch.Tensor) -> None:
        """Set hidden state.
        
        Args:
            hidden_state: Hidden state tensor to set
        """
        self.hidden_state = hidden_state


# Alias for backward compatibility
StateSpaceModel = SSM


if __name__ == "__main__":
    # Quick test
    print("Testing SSM module...")
    
    ssm = SSM(state_dim=64, hidden_dim=128, output_dim=32)
    print(f"Created SSM: state_dim=64, hidden_dim=128, output_dim=32")
    
    # Reset
    state = ssm.reset(batch_size=2)
    print(f"Reset state shape: {state.shape}")
    
    # Forward pass
    x = torch.randn(2, 64)
    output = ssm(x)
    print(f"Output shape: {output.shape}")
    
    # Save and load test
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    ssm.save(temp_path)
    print(f"Saved model to {temp_path}")
    
    loaded_ssm = SSM.load(temp_path)
    print(f"Loaded model successfully")
    
    os.remove(temp_path)
    print("\nSSM module test completed successfully!")
