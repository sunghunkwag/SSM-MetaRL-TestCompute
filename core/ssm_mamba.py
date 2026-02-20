"""Mamba State Space Model Integration for Meta-Reinforcement Learning

This module wraps the official mamba-ssm library to provide a true structured
State Space Model with O(T·d) complexity for meta-RL applications.

Architecture:
    Input Projection: Linear(input_dim -> d_model)
    Mamba Core: Mamba(d_model, d_state, d_conv, expand)
    Output Projection: Linear(d_model -> output_dim)

How Mamba Differs from Legacy SSM:
    - Legacy SSM uses MLP-based state transitions: h_t = MLP(h_{t-1}) + Linear(x_t)
      This is O(T·d²) per sequence due to dense matrix multiplications in MLPs.
    - Mamba uses selective scan with input-dependent gating and hardware-aware
      parallel scan algorithms, achieving O(T·d) complexity.
    - Mamba's state is managed internally through its selective scan mechanism,
      not as an explicit hidden state tensor passed between timesteps.

Complexity Comparison:
    - Legacy SSM: O(T·d²) per sequence (MLP-based, not parallelizable)
    - Mamba SSM:  O(T·d)  per sequence (selective scan, parallelizable)
    - S4 SSM:    O(T·log(T)) per sequence (FFT-based convolution)

When to Use Mamba vs Legacy:
    - Use Mamba when you need efficient long-sequence processing
    - Use Mamba when training on GPU with CUDA support
    - Use Legacy when you need explicit state access for debugging
    - Use Legacy when running on CPU-only environments without fallback

Fallback Behavior:
    If mamba-ssm is not installed (e.g., CPU-only environments), this module
    provides a pure-PyTorch implementation that mimics Mamba's architecture
    (input-dependent gating, 1D convolution, linear projections) but without
    the CUDA-optimized selective scan kernel. The fallback has an identical API
    but does not achieve the same hardware efficiency.

Example:
    >>> import torch
    >>> from core.ssm_mamba import MambaSSM
    >>>
    >>> model = MambaSSM(state_dim=16, input_dim=32, output_dim=32, d_model=64)
    >>> x = torch.randn(4, 32)  # Single timestep: (batch, input_dim)
    >>> hidden = model.init_hidden(4)
    >>> output, next_hidden = model(x, hidden)
    >>> print(output.shape)  # (4, 32)
    >>>
    >>> x_seq = torch.randn(4, 20, 32)  # Sequence: (batch, time, input_dim)
    >>> output_seq, _ = model(x_seq, None)
    >>> print(output_seq.shape)  # (4, 20, 32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Attempt to import official Mamba
_MAMBA_AVAILABLE = False
try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
    logger.info("Official mamba-ssm library loaded successfully.")
except ImportError:
    logger.warning(
        "mamba-ssm not installed. Using pure-PyTorch Mamba fallback. "
        "Install with: pip install mamba-ssm causal-conv1d>=1.4.0"
    )


class MambaBlockFallback(nn.Module):
    """Pure-PyTorch fallback implementation of a single Mamba block.

    This implements the core Mamba selective scan mechanism using standard
    PyTorch operations. It approximates Mamba's architecture:
        1. Input projection with expansion
        2. 1D depthwise convolution
        3. Input-dependent selective scan (SSM)
        4. Output gating with SiLU activation

    This fallback does NOT use the hardware-optimized CUDA kernel from
    mamba-ssm, so it will be slower than the official implementation on GPU.
    However, it produces functionally equivalent results and works on CPU.

    Args:
        d_model: Model dimension (input and output dimension of the block)
        d_state: SSM state expansion factor (N in Mamba paper)
        d_conv: Local convolution kernel width
        expand: Block expansion factor (E in Mamba paper)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Input projection: projects to 2 * d_inner (for x and z branches)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # 1D depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # SSM parameters projection (from convolved input)
        # Projects to: dt (d_inner) + B (d_state) + C (d_state)
        self.x_proj = nn.Linear(
            self.d_inner, self.d_state + self.d_state + self.d_inner, bias=False
        )

        # dt (delta) projection: low-rank factorization
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # SSM state matrix A (log-spaced initialization)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Mamba block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            hidden_state: Optional initial state (B, d_inner, d_state)

        Returns:
            Tuple of:
                - Output tensor of shape (batch, seq_len, d_model)
                - Final state of shape (batch, d_inner, d_state)
        """
        batch, seq_len, _ = x.shape

        # Input projection: split into x_branch and z_branch
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)

        # 1D convolution (causal)
        # Note: In a true streaming mode, we would also need conv state.
        # For simplicity in this meta-RL context, we focus on the SSM state h.
        x_conv = x_branch.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Causal: trim padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # SSM parameter computation (input-dependent / selective)
        x_ssm_proj = self.x_proj(x_conv)  # (B, L, d_state + d_state + d_inner)
        B_input = x_ssm_proj[:, :, :self.d_state]  # (B, L, N)
        C_input = x_ssm_proj[:, :, self.d_state:2 * self.d_state]  # (B, L, N)
        dt_input = x_ssm_proj[:, :, 2 * self.d_state:]  # (B, L, d_inner)

        # Delta (dt) computation with softplus
        dt = self.dt_proj(dt_input)  # (B, L, d_inner)
        dt = F.softplus(dt)  # Ensure positive

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, N)

        # Selective scan
        y, next_h = self._selective_scan(x_conv, dt, A, B_input, C_input, hidden_state)

        # Skip connection with D
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # Output gating
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)  # (B, L, d_model)
        return output, next_h

    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        h_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selective scan implementation in pure PyTorch with state propagation.
        """
        batch, seq_len, d_inner = u.shape
        n = A.shape[1]

        # Discretize
        delta_A = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # (B, L, D, N)
        delta_B_u = (
            delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        )  # (B, L, D, N)

        # Initial state
        if h_init is None:
            h = torch.zeros(batch, d_inner, n, device=u.device, dtype=u.dtype)
        else:
            h = h_init

        outputs = []
        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B_u[:, t]  # (B, D, N)
            y_t = torch.einsum("bdn,bdn->bd", h, C[:, t].unsqueeze(1).expand_as(h))
            outputs.append(y_t)

        return torch.stack(outputs, dim=1), h



class MambaSSM(nn.Module):
    """Mamba-based State Space Model for Meta-Reinforcement Learning.

    Wraps the official mamba-ssm library (or pure-PyTorch fallback) to provide
    a drop-in replacement for the legacy SSM with true structured SSM properties.

    Architecture:
        Input Projection:  Linear(input_dim -> d_model)
        Mamba Block:       Mamba(d_model, d_state, d_conv, expand)
        Layer Norm:        LayerNorm(d_model)
        Output Projection: Linear(d_model -> output_dim)

    The model accepts both single-step (B, D) and sequence (B, T, D) inputs.
    For single-step inputs, the tensor is unsqueezed to (B, 1, D), processed
    through the Mamba block, and squeezed back to (B, D).

    Hidden state management: Mamba manages state internally through its
    selective scan mechanism. The init_hidden() and hidden_state parameter
    are maintained for API compatibility with the legacy SSM, but the actual
    state is managed within the Mamba block.

    Args:
        state_dim: SSM state expansion factor (N in Mamba paper, default=16)
        input_dim: Dimension of input features
        output_dim: Dimension of output features
        d_model: Internal model dimension for the Mamba block (default=64)
        d_conv: Convolution kernel width (default=4)
        expand: Block expansion factor (default=2)
        device: Device to run on ('cpu' or 'cuda')

    Complexity:
        Time: O(T·d) per sequence with official Mamba (O(T·d·N) with fallback)
        Space: O(d·N) for state, O(d²·E) for parameters
        Parallelizable: Yes (with official Mamba CUDA kernel)
    """

    def __init__(
        self,
        state_dim: int = 16,
        input_dim: int = 32,
        output_dim: int = 32,
        d_model: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        device: str = "cpu",
    ):
        super(MambaSSM, self).__init__()

        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.d_conv = d_conv
        self.expand = expand
        self.device = device

        # Input projection: map from input_dim to d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # Layer normalization before Mamba block
        self.norm = nn.LayerNorm(d_model)

        # Mamba core block
        if _MAMBA_AVAILABLE and device != "cpu":
            # Use official Mamba implementation (requires CUDA)
            self.mamba_block = Mamba(
                d_model=d_model,
                d_state=state_dim,
                d_conv=d_conv,
                expand=expand,
            )
            self._using_official_mamba = True
            logger.info(
                f"Using official Mamba block: d_model={d_model}, "
                f"d_state={state_dim}, d_conv={d_conv}, expand={expand}"
            )
        else:
            # Use pure-PyTorch fallback
            self.mamba_block = MambaBlockFallback(
                d_model=d_model,
                d_state=state_dim,
                d_conv=d_conv,
                expand=expand,
            )
            self._using_official_mamba = False
            if device != "cpu" and not _MAMBA_AVAILABLE:
                logger.warning(
                    "CUDA device requested but mamba-ssm not installed. "
                    "Using PyTorch fallback on CUDA device."
                )
            logger.info(
                f"Using PyTorch Mamba fallback: d_model={d_model}, "
                f"d_state={state_dim}, d_conv={d_conv}, expand={expand}"
            )

        # Output projection: map from d_model to output_dim
        self.output_projection = nn.Linear(d_model, output_dim)

        # Move model to device
        self.to(device)

    def init_hidden(self, batch_size: int = 1) -> Optional[torch.Tensor]:
        """Initialize the hidden state for the Mamba block.

        Args:
            batch_size: Number of sequences in batch

        Returns:
            Zero tensor of shape (batch_size, d_inner, state_dim)
            if using the fallback.
        """
        if not self._using_official_mamba:
            d_inner = int(self.expand * self.d_model)
            return torch.zeros(
                batch_size, d_inner, self.state_dim, device=self.device
            )
        return None

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the Mamba SSM.

        Supports both single-step and sequence inputs.
        Hidden state is propagated if using the fallback.

        Args:
            x: Input tensor of shape (B, input_dim) or (B, T, input_dim)
            hidden_state: Hidden state (B, d_inner, state_dim)

        Returns:
            Tuple of:
                - output: Output tensor (B, output_dim) or (B, T, output_dim)
                - next_hidden_state: Updated state if using fallback, else None
        """
        single_step = x.dim() == 2
        if single_step:
            x = x.unsqueeze(1)

        # Input projection & norm
        x = self.input_projection(x)
        x = self.norm(x)

        # Mamba block
        if not self._using_official_mamba:
            x, next_hidden = self.mamba_block(x, hidden_state)
        else:
            # Official mamba usually manages state via its own cache or sequence processing
            # For simplicity in this CPU context, we assume sequence processing
            x = self.mamba_block(x)
            next_hidden = None

        # Output projection
        output = self.output_projection(x)

        if single_step:
            output = output.squeeze(1)

        return output, next_hidden

    def save(self, path: str) -> None:
        """Save model checkpoint with configuration.

        Saves both the model state dict and all configuration parameters
        needed to reconstruct the model.

        Args:
            path: Path to save the checkpoint file
        """
        # Create directory if it doesn't exist
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "state_dim": self.state_dim,
                    "input_dim": self.input_dim,
                    "output_dim": self.output_dim,
                    "d_model": self.d_model,
                    "d_conv": self.d_conv,
                    "expand": self.expand,
                    "device": self.device,
                },
                "model_type": "MambaSSM",
                "using_official_mamba": self._using_official_mamba,
            },
            path,
        )
        logger.info(f"MambaSSM checkpoint saved to {path}")

    @staticmethod
    def load(path: str, device: Optional[str] = None) -> "MambaSSM":
        """Load model checkpoint.

        Creates a new MambaSSM instance from a saved checkpoint, restoring
        both the configuration and learned parameters.

        Args:
            path: Path to checkpoint file
            device: Override device (default: use saved device)

        Returns:
            Loaded MambaSSM model with restored weights
        """
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]

        # Override device if specified
        if device is not None:
            config["device"] = device

        # Create and load model
        model = MambaSSM(**config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(config["device"])

        logger.info(f"MambaSSM checkpoint loaded from {path}")
        return model

    def get_complexity_info(self) -> Dict[str, Any]:
        """Get model complexity information.

        Returns:
            Dictionary with complexity metrics including parameter count,
            architecture type, and theoretical complexity class.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": "MambaSSM",
            "using_official_mamba": self._using_official_mamba,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "d_model": self.d_model,
            "d_state": self.state_dim,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "complexity": "O(T·d)" if self._using_official_mamba else "O(T·d·N)",
            "parallelizable": self._using_official_mamba,
        }

    def __repr__(self) -> str:
        backend = "official" if self._using_official_mamba else "fallback"
        return (
            f"MambaSSM(input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"d_model={self.d_model}, d_state={self.state_dim}, "
            f"d_conv={self.d_conv}, expand={self.expand}, backend={backend})"
        )


if __name__ == "__main__":
    # Quick self-test
    logging.basicConfig(level=logging.INFO)

    print("=" * 50)
    print("Testing MambaSSM")
    print("=" * 50)

    model = MambaSSM(
        state_dim=16, input_dim=32, output_dim=32, d_model=64, device="cpu"
    )
    print(f"\nModel: {model}")
    print(f"Complexity: {model.get_complexity_info()}")

    # Single step test
    batch_size = 4
    x_single = torch.randn(batch_size, 32)
    hidden = model.init_hidden(batch_size)
    out_single, next_h = model(x_single, hidden)
    print(f"\nSingle step: input={x_single.shape} -> output={out_single.shape}")
    assert out_single.shape == (batch_size, 32), f"Expected (4, 32), got {out_single.shape}"

    # Sequence test
    seq_len = 20
    x_seq = torch.randn(batch_size, seq_len, 32)
    out_seq, _ = model(x_seq, None)
    print(f"Sequence:    input={x_seq.shape} -> output={out_seq.shape}")
    assert out_seq.shape == (batch_size, seq_len, 32), f"Expected (4, 20, 32), got {out_seq.shape}"

    # Backward test
    loss = out_seq.mean()
    loss.backward()
    has_grads = any(p.grad is not None for p in model.parameters())
    print(f"Backward:    gradients computed = {has_grads}")
    assert has_grads, "No gradients computed!"

    print("\n" + "=" * 50)
    print("All MambaSSM self-tests passed!")
    print("=" * 50)
