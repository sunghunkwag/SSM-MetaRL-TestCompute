"""
SOTA Baseline Methods for Meta-RL Comparison

This module implements baseline methods to compare against SSM-MetaRL:
1. LSTM-MAML: MAML with LSTM policy
2. Transformer-MAML: MAML with Transformer policy  
3. MLP-MAML: MAML with feedforward MLP policy
4. GRU-MAML: MAML with GRU policy

These baselines allow us to demonstrate the value of SSM over other
sequence modeling approaches in the meta-RL setting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class MLPPolicy(nn.Module):
    """Simple feedforward MLP policy (no sequence modeling)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self._stateful = False  # Mark as stateless for MetaMAML
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (stateless).
        
        Args:
            x: Input tensor (batch, state_dim)
        
        Returns:
            Action logits (batch, action_dim)
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class LSTMPolicy(nn.Module):
    """LSTM-based policy for sequence modeling."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
        
        self._stateful = True  # Mark as stateful for MetaMAML
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state (h, c)."""
        device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h, c)
    
    def forward(self, x: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with LSTM.
        
        Args:
            x: Input tensor (batch, state_dim)
            hidden_state: Tuple of (h, c) tensors
        
        Returns:
            Tuple of (action_logits, next_hidden_state)
        """
        # Add sequence dimension if needed
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (batch, 1, state_dim)
        
        # LSTM forward
        out, next_hidden = self.lstm(x, hidden_state)
        
        # Remove sequence dimension
        out = out.squeeze(1)  # (batch, hidden_dim)
        
        # Action logits
        action_logits = self.fc(out)
        
        return action_logits, next_hidden


class GRUPolicy(nn.Module):
    """GRU-based policy for sequence modeling."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
        
        self._stateful = True
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize GRU hidden state."""
        device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)
    
    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with GRU.
        
        Args:
            x: Input tensor (batch, state_dim)
            hidden_state: Hidden state tensor
        
        Returns:
            Tuple of (action_logits, next_hidden_state)
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        
        out, next_hidden = self.gru(x, hidden_state)
        out = out.squeeze(1)
        action_logits = self.fc(out)
        
        return action_logits, next_hidden


class TransformerPolicy(nn.Module):
    """Transformer-based policy for sequence modeling."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, 
                 num_heads: int = 4, num_layers: int = 2, max_seq_len: int = 1000):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Input embedding
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, action_dim)
        
        self._stateful = True
        self._seq_buffer = []  # Buffer to store sequence history
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def init_hidden(self, batch_size: int) -> list:
        """Initialize sequence buffer (acts as 'hidden state')."""
        return []
    
    def forward(self, x: torch.Tensor, hidden_state: list) -> Tuple[torch.Tensor, list]:
        """
        Forward pass with Transformer.
        
        Args:
            x: Input tensor (batch, state_dim)
            hidden_state: Sequence buffer (list of past observations)
        
        Returns:
            Tuple of (action_logits, updated_sequence_buffer)
        """
        device = x.device
        batch_size = x.shape[0]
        
        # Add current observation to buffer
        seq_buffer = hidden_state.copy() if hidden_state else []
        seq_buffer.append(x.detach().cpu())
        
        # Limit buffer size
        if len(seq_buffer) > self.max_seq_len:
            seq_buffer = seq_buffer[-self.max_seq_len:]
        
        # Create sequence tensor
        seq = torch.stack([s.to(device) for s in seq_buffer], dim=1)  # (batch, seq_len, state_dim)
        seq_len = seq.shape[1]
        
        # Project to hidden dimension
        h = self.input_proj(seq)  # (batch, seq_len, hidden_dim)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :].to(device)
        h = h + pos_enc
        
        # Transformer forward
        h = self.transformer(h)  # (batch, seq_len, hidden_dim)
        
        # Take last token output
        h_last = h[:, -1, :]  # (batch, hidden_dim)
        
        # Action logits
        action_logits = self.output_proj(h_last)
        
        return action_logits, seq_buffer


# ============================================================================
# Baseline Registry
# ============================================================================

BASELINE_POLICIES = {
    'mlp': MLPPolicy,
    'lstm': LSTMPolicy,
    'gru': GRUPolicy,
    'transformer': TransformerPolicy,
}


def get_baseline_policy(name: str, state_dim: int, action_dim: int, 
                       hidden_dim: int = 128, **kwargs) -> nn.Module:
    """
    Get a baseline policy by name.
    
    Args:
        name: Policy name ('mlp', 'lstm', 'gru', 'transformer')
        state_dim: State dimension
        action_dim: Action dimension
        hidden_dim: Hidden dimension
        **kwargs: Additional policy-specific arguments
    
    Returns:
        Policy module
    """
    if name not in BASELINE_POLICIES:
        raise ValueError(f"Unknown baseline: {name}. "
                        f"Available: {list(BASELINE_POLICIES.keys())}")
    
    return BASELINE_POLICIES[name](state_dim, action_dim, hidden_dim, **kwargs)


def list_baselines() -> list:
    """List all available baseline policies."""
    return list(BASELINE_POLICIES.keys())


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_model_sizes():
    """Compare parameter counts of different policies."""
    state_dim = 17  # HalfCheetah
    action_dim = 6
    hidden_dim = 128
    
    print("\n" + "="*60)
    print("Model Size Comparison (HalfCheetah-v4)")
    print("="*60)
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Hidden dim: {hidden_dim}")
    print()
    
    for name in list_baselines():
        try:
            model = get_baseline_policy(name, state_dim, action_dim, hidden_dim)
            params = count_parameters(model)
            print(f"{name.upper():15s}: {params:>8,} parameters")
        except Exception as e:
            print(f"{name.upper():15s}: Error - {e}")
    
    # Compare with SSM
    try:
        import sys
        sys.path.insert(0, '/home/ubuntu/SSM-MetaRL-TestCompute')
        from core.ssm import StateSpaceModel
        
        ssm = StateSpaceModel(
            state_dim=hidden_dim,
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim
        )
        params = count_parameters(ssm)
        print(f"{'SSM':15s}: {params:>8,} parameters")
    except Exception as e:
        print(f"{'SSM':15s}: Error - {e}")
    
    print("="*60)


def test_baseline_policy(name: str):
    """Test a baseline policy."""
    print(f"\n{'='*60}")
    print(f"Testing: {name.upper()} Policy")
    print(f"{'='*60}")
    
    state_dim = 17
    action_dim = 6
    hidden_dim = 64
    batch_size = 4
    
    # Create policy
    policy = get_baseline_policy(name, state_dim, action_dim, hidden_dim)
    print(f"Parameters: {count_parameters(policy):,}")
    print(f"Stateful: {hasattr(policy, '_stateful') and policy._stateful}")
    
    # Test forward pass
    x = torch.randn(batch_size, state_dim)
    
    if hasattr(policy, 'init_hidden'):
        hidden = policy.init_hidden(batch_size)
        output, next_hidden = policy(x, hidden)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Hidden state type: {type(next_hidden)}")
    else:
        output = policy(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    
    print(f"✓ {name.upper()} policy test passed")


if __name__ == "__main__":
    print("SSM-MetaRL Baseline Policies")
    print("="*60)
    
    print("\nAvailable baselines:")
    for name in list_baselines():
        print(f"  - {name}")
    
    # Compare model sizes
    compare_model_sizes()
    
    # Test each baseline
    print("\n" + "="*60)
    print("Running tests...")
    print("="*60)
    
    for name in list_baselines():
        try:
            test_baseline_policy(name)
        except Exception as e:
            print(f"✗ Error testing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

