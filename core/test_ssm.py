#!/usr/bin/env python3
"""Test file for core SSM module using pytest assertions."""

import pytest
import torch
import tempfile
import os
from core.ssm import SSM


def test_ssm_import():
    """Test that SSM can be imported successfully."""
    from core.ssm import SSM
    assert SSM is not None


def test_ssm_initialization():
    """Test SSM initialization with correct parameters."""
    ssm = SSM(
        state_dim=64,
        hidden_dim=128,
        output_dim=32
    )
    
    # Verify SSM is a PyTorch module
    assert isinstance(ssm, torch.nn.Module)
    assert type(ssm).__name__ == 'SSM'


def test_ssm_reset():
    """Test SSM reset functionality."""
    ssm = SSM(
        state_dim=64,
        hidden_dim=128,
        output_dim=32
    )
    
    state = ssm.reset()
    
    # Verify state is a torch tensor
    assert isinstance(state, torch.Tensor)
    # Verify state has correct shape (hidden_dim,)
    assert state.shape == (128,)


def test_ssm_forward():
    """Test SSM forward pass."""
    ssm = SSM(
        state_dim=64,
        hidden_dim=128,
        output_dim=32
    )
    
    # Create dummy input: batch_size=1, state_dim=64
    dummy_input = torch.randn(1, 64)
    output = ssm(dummy_input)
    
    # Verify output shape is correct
    assert output.shape == (1, 32)
    # Verify output is a torch tensor
    assert isinstance(output, torch.Tensor)


def test_ssm_forward_batch():
    """Test SSM forward pass with batched input."""
    ssm = SSM(
        state_dim=64,
        hidden_dim=128,
        output_dim=32
    )
    
    # Test with larger batch size
    batch_size = 8
    dummy_input = torch.randn(batch_size, 64)
    output = ssm(dummy_input)
    
    # Verify output shape matches batch size
    assert output.shape == (batch_size, 32)


def test_ssm_save_load():
    """Test SSM state dict save and load functionality."""
    # Create and initialize first model
    ssm = SSM(state_dim=64, hidden_dim=128, output_dim=32)
    dummy_input = torch.randn(1, 64)
    output1 = ssm(dummy_input)
    
    # Save model parameters
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        torch.save(ssm.state_dict(), temp_path)
        assert os.path.exists(temp_path)
        
        # Load into new model
        new_ssm = SSM(state_dim=64, hidden_dim=128, output_dim=32)
        new_ssm.load_state_dict(torch.load(temp_path))
        
        # Verify loaded model produces identical output
        output2 = new_ssm(dummy_input)
        assert torch.allclose(output1, output2)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_ssm_different_dimensions():
    """Test SSM with different dimension configurations."""
    # Test different configurations
    configs = [
        (32, 64, 16),
        (128, 256, 64),
        (16, 32, 8),
    ]
    
    for state_dim, hidden_dim, output_dim in configs:
        ssm = SSM(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        # Test forward pass
        dummy_input = torch.randn(1, state_dim)
        output = ssm(dummy_input)
        
        # Verify output shape
        assert output.shape == (1, output_dim)
        
        # Test reset
        state = ssm.reset()
        assert state.shape == (hidden_dim,)


def test_ssm_gradient_flow():
    """Test that gradients flow through SSM correctly."""
    ssm = SSM(state_dim=64, hidden_dim=128, output_dim=32)
    
    dummy_input = torch.randn(1, 64, requires_grad=True)
    output = ssm(dummy_input)
    
    # Compute a simple loss and backpropagate
    loss = output.sum()
    loss.backward()
    
    # Verify gradients exist for input
    assert dummy_input.grad is not None
    assert dummy_input.grad.shape == dummy_input.shape
    
    # Verify at least some parameters have gradients
    has_grad = any(p.grad is not None for p in ssm.parameters())
    assert has_grad
