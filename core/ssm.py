"""Core SSM Module: PyTorch State Space Model Implementation
... (docstring comments) ...
"""
import torch
import torch.nn as nn
import os
from typing import Tuple, Optional, Dict, Any

class SSM(nn.Module):
    """PyTorch State Space Model with neural network components.
    ... (Args documentation) ...
    """

    def __init__(self,
                 state_dim: int,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 device: str = 'cpu'):
        super(SSM, self).__init__()

        self.state_dim = state_dim
        self.input_dim = input_dim
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
        self.input_projection = nn.Linear(input_dim, state_dim)

        # Output network (C matrix)
        self.output_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Direct feedthrough (D matrix) - optional
        self.feedthrough = nn.Linear(input_dim, output_dim)

        # Move model to device
        self.to(device)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize the hidden state."""
        return torch.zeros(batch_size, self.state_dim, device=self.device)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the SSM. Returns output and next hidden state.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            hidden_state: Current hidden state tensor of shape (batch_size, state_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output tensor of shape (batch_size, output_dim)
                - Next hidden state tensor of shape (batch_size, state_dim)
        """
        # State transition: h_t = A @ h_{t-1} + B @ u_t
        state_update = self.state_transition(hidden_state)
        input_update = self.input_projection(x)
        next_hidden_state = state_update + input_update

        # Output: y_t = C @ h_t + D @ u_t (using the *next* state)
        output = self.output_network(next_hidden_state)
        feedthrough_output = self.feedthrough(x)
        final_output = output + feedthrough_output

        return final_output, next_hidden_state

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
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'device': self.device
            }
        }, path)

    @staticmethod
    def load(path: str, device: Optional[str] = None) -> 'SSM':
        """Load model parameters using torch.load.
        ... (Args documentation) ...
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

# Alias for backward compatibility
StateSpaceModel = SSM

if __name__ == "__main__":
    # Quick test
    print("Testing SSM module...")

    ssm = SSM(state_dim=64, input_dim=32, output_dim=16, hidden_dim=128)
    print(f"Created SSM: state_dim=64, input_dim=32, output_dim=16, hidden_dim=128")

    # Initialize hidden state
    batch_size = 4
    hidden = ssm.init_hidden(batch_size)
    print(f"Initial hidden state shape: {hidden.shape}") # Expected: [4, 64]

    # Forward pass
    x = torch.randn(batch_size, 32) # input_dim = 32
    output, next_hidden = ssm(x, hidden)
    print(f"Input shape: {x.shape}")        # Expected: [4, 32]
    print(f"Output shape: {output.shape}")    # Expected: [4, 16]
    print(f"Next hidden shape: {next_hidden.shape}") # Expected: [4, 64]

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
