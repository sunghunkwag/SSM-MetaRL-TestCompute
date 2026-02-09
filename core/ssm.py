"""Structured State Space Model (S4D) with explicit recurrence.

This module implements a diagonal SSM (S4D) with bilinear (Tustin) discretization
in pure PyTorch, compatible with higher-order gradients for MAML.
"""
from __future__ import annotations

import os
from typing import Tuple, Optional

import torch
import torch.nn as nn


class SSM(nn.Module):
    """Structured State Space Model (S4D) with diagonal dynamics.

    State equation (continuous-time):
        h'(t) = A h(t) + B x(t)
    Output equation:
        y(t) = C h(t) + D x(t)

    Discretization (bilinear/Tustin) with learnable step size Δ:
        Ā = (I - Δ/2 A)^{-1} (I + Δ/2 A)
        B̄ = (I - Δ/2 A)^{-1} (Δ B)

    Forward uses explicit recurrence per timestep for RL compatibility.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        # Diagonal continuous-time dynamics A = diag(a), complex-valued.
        # Initialize with stable real parts (negative) and small imaginary parts.
        a_real = -0.5 * torch.ones(state_dim)
        a_imag = 0.1 * torch.randn(state_dim)
        self.a_real = nn.Parameter(a_real)
        self.a_imag = nn.Parameter(a_imag)

        # Input and output projections in complex space.
        self.B_real = nn.Parameter(0.1 * torch.randn(state_dim, input_dim))
        self.B_imag = nn.Parameter(0.1 * torch.randn(state_dim, input_dim))
        self.C_real = nn.Parameter(0.1 * torch.randn(output_dim, state_dim))
        self.C_imag = nn.Parameter(0.1 * torch.randn(output_dim, state_dim))

        # Real feedthrough term D.
        self.D = nn.Linear(input_dim, output_dim)

        # Learnable step size Δ (positive via softplus).
        self.log_dt = nn.Parameter(torch.zeros(state_dim))

        self.to(device)

    def _complex_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = torch.complex(self.a_real, self.a_imag)
        b = torch.complex(self.B_real, self.B_imag)
        c = torch.complex(self.C_real, self.C_imag)
        return a, b, c

    def _discretize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute discrete-time (Ā, B̄) with bilinear/Tustin method.

        This keeps the computation in PyTorch for autograd compatibility,
        ensuring higher-order gradients can flow through Δ and A.
        """
        a, b, _ = self._complex_params()
        dt = torch.nn.functional.softplus(self.log_dt)

        # Diagonal A -> elementwise discretization.
        denom = 1.0 - 0.5 * dt * a
        a_bar = (1.0 + 0.5 * dt * a) / denom
        b_bar = (dt[:, None] * b) / denom[:, None]
        return a_bar, b_bar

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize the hidden state to zeros (complex).

        Args:
            batch_size: Number of sequences in batch

        Returns:
            Zero tensor of shape (batch_size, state_dim) with complex dtype.
        """
        return torch.zeros(
            batch_size,
            self.state_dim,
            device=self.device,
            dtype=torch.complex64,
        )

    def forward(
        self, x: torch.Tensor, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single timestep.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            hidden_state: Current hidden state (batch_size, state_dim), complex

        Returns:
            output: Real-valued output tensor (batch_size, output_dim)
            next_hidden_state: Updated complex state (batch_size, state_dim)
        """
        if not torch.is_complex(hidden_state):
            hidden_state = torch.complex(hidden_state, torch.zeros_like(hidden_state))

        a_bar, b_bar = self._discretize()
        _, _, c = self._complex_params()

        # Explicit recurrence for RL inference.
        x_complex = torch.complex(x, torch.zeros_like(x))
        next_hidden_state = hidden_state * a_bar + x_complex @ b_bar.T

        # Output: take real projection to ensure real-valued outputs.
        y_complex = next_hidden_state @ c.T
        y_real = y_complex.real + self.D(x)

        return y_real, next_hidden_state

    def save(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save the checkpoint
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "state_dim": self.state_dim,
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "output_dim": self.output_dim,
                    "device": self.device,
                },
            },
            path,
        )

    @staticmethod
    def load(path: str, device: Optional[str] = None) -> "SSM":
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
            device: Override device (default: use saved device)

        Returns:
            Loaded SSM model
        """
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]

        if device is not None:
            config["device"] = device

        model = SSM(**config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(config["device"])
        return model


StateSpaceModel = SSM

if __name__ == "__main__":
    print("Testing Structured State Space Model (S4D)...")

    ssm = SSM(state_dim=64, input_dim=32, output_dim=16, hidden_dim=128)
    batch_size = 4
    hidden = ssm.init_hidden(batch_size)
    x = torch.randn(batch_size, 32)
    output, next_hidden = ssm(x, hidden)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Next hidden shape: {next_hidden.shape}")
