"""Core Neural State Model: PyTorch Implementation with Explicit State Tracking

⚠️ IMPORTANT: This is NOT a structured State Space Model (SSM) like S4/Mamba/LRU.

This module implements a neural network with explicit state representation,
inspired by State Space Model concepts but using standard MLP components.

Architecture:
    h_t = MLP(h_{t-1}) + Linear(x_t)  # State transition with residual
    y_t = MLP(h_t) + Linear(x_t)       # Output with feedthrough

What this IS:
    - Neural network with explicit state tracking
    - MLP-based state transitions with residual connections
    - Compatible with meta-learning algorithms (MAML)
    - Recurrent processing (not parallelizable)

What this is NOT:
    - Structured SSM (no HiPPO, diagonal, or low-rank parameterization)
    - Continuous-time dynamics (no discretization)
    - FFT-based convolution mode (no parallel processing)
    - Sub-quadratic complexity (actual: O(d²) per timestep)

Complexity:
    - Forward pass: O(d²) per timestep (due to MLP layers)
    - Similar to GRU/LSTM, not faster
    - No convolution mode for parallelization

Use this if:
    - You need explicit state representation for RL
    - You want compatibility with standard meta-learning
    - You prioritize simplicity over efficiency

Consider alternatives if:
    - You need true sub-quadratic complexity
    - You want FFT-based parallel processing
    - You require structured SSM guarantees

Example:
    >>> import torch
    >>> from core.ssm import SSM
    >>> 
    >>> model = SSM(state_dim=64, input_dim=32, output_dim=16)
    >>> x = torch.randn(4, 32)  # batch_size=4
    >>> h = model.init_hidden(4)
    >>> output, next_h = model(x, h)
    >>> print(output.shape, next_h.shape)
    torch.Size([4, 16]) torch.Size([4, 64])
"""
import torch
import torch.nn as nn
import os
from typing import Tuple, Optional, Dict, Any

class SSM(nn.Module):
    """Neural State Model with MLP-based transitions (NOT structured SSM).
    
    ⚠️ WARNING: Despite the name, this is NOT a structured State Space Model.
    This is a neural network with explicit state, using MLP for transitions.
    
    Architecture:
        State transition: h_t = MLP(h_{t-1}) + Linear(x_t)
        Output: y_t = MLP(h_t) + Linear(x_t)
    
    The "SSM" naming is kept for backward compatibility, but this should be
    understood as "Stateful Sequential Model" not "State Space Model".
    
    Args:
        state_dim (int): Dimension of the internal hidden state
        input_dim (int): Dimension of input features
        output_dim (int): Dimension of output features
        hidden_dim (int): Hidden layer size in MLP networks (default: 128)
        device (str): Device to run on ('cpu' or 'cuda')
    
    Attributes:
        state_transition: MLP network for state updates (A matrix analog)
        input_projection: Linear layer for input (B matrix analog)
        output_network: MLP network for output (C matrix analog)
        feedthrough: Direct input-to-output connection (D matrix analog)
    
    Methods:
        forward(x, hidden_state): Process one timestep
        init_hidden(batch_size): Initialize hidden state
        save(path): Save model checkpoint
        load(path): Load model checkpoint
    
    Complexity:
        Time: O(d²) per timestep (due to Linear layers in MLPs)
        Space: O(d²) for parameters
        Not parallelizable (recurrent structure)
    
    Example:
        >>> model = SSM(state_dim=128, input_dim=64, output_dim=32)
        >>> h = model.init_hidden(batch_size=4)
        >>> x = torch.randn(4, 64)
        >>> y, next_h = model(x, h)
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

        # State transition network (A matrix analog)
        # Uses MLP instead of structured matrix
        self.state_transition = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Input projection network (B matrix analog)
        self.input_projection = nn.Linear(input_dim, state_dim)

        # Output network (C matrix analog)
        self.output_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Direct feedthrough (D matrix analog)
        self.feedthrough = nn.Linear(input_dim, output_dim)

        # Move model to device
        self.to(device)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize the hidden state to zeros.
        
        Args:
            batch_size: Number of sequences in batch
        
        Returns:
            Zero tensor of shape (batch_size, state_dim)
        """
        return torch.zeros(batch_size, self.state_dim, device=self.device)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: process one timestep with explicit state.
        
        Architecture:
            h_t = MLP(h_{t-1}) + Linear(x_t)  # State update with residual
            y_t = MLP(h_t) + Linear(x_t)       # Output with feedthrough

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            hidden_state: Current hidden state (batch_size, state_dim)

        Returns:
            Tuple of:
                - output: Output tensor (batch_size, output_dim)
                - next_hidden_state: Updated state (batch_size, state_dim)
        
        Complexity:
            O(d²) due to Linear layers in MLPs, where d ~ hidden_dim
        """
        if x.dim() == 3:  # (batch, seq_len, input_dim)
            outputs = []
            for t in range(x.shape[1]):
                out, hidden_state = self.forward(x[:, t, :], hidden_state)
                outputs.append(out)
            return torch.stack(outputs, dim=1), hidden_state

        # Single-step forward:
        # State transition: h_t = MLP(h_{t-1}) + Linear(x_t)
        state_update = self.state_transition(hidden_state)
        input_update = self.input_projection(x)
        next_hidden_state = state_update + input_update

        # Output: y_t = MLP(h_t) + Linear(x_t)
        output = self.output_network(next_hidden_state)
        feedthrough_output = self.feedthrough(x)
        final_output = output + feedthrough_output

        return final_output, next_hidden_state

    def save(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save the checkpoint
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Save state dict and config
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
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            device: Override device (default: use saved device)
        
        Returns:
            Loaded SSM model
        """
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']

        # Override device if specified
        if device is not None:
            config['device'] = device

        # Create and load model
        model = SSM(**config)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(config['device'])

        return model

# Alias for backward compatibility
# NOTE: This is NOT a true "State Space Model", but a neural network
# with explicit state tracking. The name is kept for compatibility.
StateSpaceModel = SSM

if __name__ == "__main__":
    # Quick test
    print("Testing Neural State Model (SSM)...")
    print("Note: This is NOT a structured SSM, but an MLP-based state model.\n")

    ssm = SSM(state_dim=64, input_dim=32, output_dim=16, hidden_dim=128)
    print(f"Created model: state_dim=64, input_dim=32, output_dim=16, hidden_dim=128")

    # Initialize hidden state
    batch_size = 4
    hidden = ssm.init_hidden(batch_size)
    print(f"Initial hidden state shape: {hidden.shape}")  # Expected: [4, 64]

    # Forward pass
    x = torch.randn(batch_size, 32)  # input_dim = 32
    output, next_hidden = ssm(x, hidden)
    print(f"Input shape: {x.shape}")          # Expected: [4, 32]
    print(f"Output shape: {output.shape}")     # Expected: [4, 16]
    print(f"Next hidden shape: {next_hidden.shape}")  # Expected: [4, 64]

    # Save and load test
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name

    ssm.save(temp_path)
    print(f"\nSaved model to {temp_path}")

    loaded_ssm = SSM.load(temp_path)
    print(f"Loaded model successfully")

    os.remove(temp_path)
    print("\n✓ Neural State Model test completed successfully!")
