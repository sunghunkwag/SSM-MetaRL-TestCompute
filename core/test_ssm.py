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
    
    batch_size = 1
    state_dim = 64
    state = ssm.reset()
    
    # Verify state has correct shape (batch_size, state_dim)
    assert state.shape == (batch_size, state_dim)
    # Verify state is a PyTorch tensor
    assert isinstance(state, torch.Tensor)

def test_ssm_forward():
    """Test SSM forward pass."""
    batch_size = 8
    state_dim = 64
    hidden_dim = 128
    output_dim = 32
    
    ssm = SSM(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    # Create input tensors
    state = torch.randn(batch_size, state_dim)
    obs = torch.randn(batch_size, 10)
    action = torch.randn(batch_size, 5)
    reward = torch.randn(batch_size, 1)
    
    # Forward pass
    next_state, output = ssm.forward(state, obs, action, reward)
    
    # Verify output shapes
    assert next_state.shape == (batch_size, state_dim)
    assert output.shape == (batch_size, output_dim)
    
    # Verify outputs are PyTorch tensors
    assert isinstance(next_state, torch.Tensor)
    assert isinstance(output, torch.Tensor)

def test_ssm_save_load():
    """Test SSM save and load functionality."""
    ssm = SSM(
        state_dim=64,
        hidden_dim=128,
        output_dim=32
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_ssm.pt')
        
        # Generate input tensors
        batch_size = 4
        state = torch.randn(batch_size, 64)
        obs = torch.randn(batch_size, 10)
        action = torch.randn(batch_size, 5)
        reward = torch.randn(batch_size, 1)
        
        # Save model
        torch.save(ssm.state_dict(), save_path)
        
        # Verify file exists
        assert os.path.exists(save_path)
        
        # Create new model and load state
        ssm_loaded = SSM(
            state_dim=64,
            hidden_dim=128,
            output_dim=32
        )
        ssm_loaded.load_state_dict(torch.load(save_path))
        
        # Verify loaded model produces identical output
        ssm.eval()
        ssm_loaded.eval()
        with torch.no_grad():
            next_state1, output1 = ssm(state, obs, action, reward)
            next_state2, output2 = ssm_loaded(state, obs, action, reward)
            
        assert torch.allclose(next_state1, next_state2)
        assert torch.allclose(output1, output2)

def test_ssm_different_dimensions():
    """Test SSM with various dimension configurations."""
    configs = [
        {'state_dim': 32, 'hidden_dim': 64, 'output_dim': 16},
        {'state_dim': 128, 'hidden_dim': 256, 'output_dim': 64},
        {'state_dim': 16, 'hidden_dim': 32, 'output_dim': 8},
    ]
    
    for config in configs:
        ssm = SSM(**config)
        state_dim = config['state_dim']
        hidden_dim = config['hidden_dim']
        output_dim = config['output_dim']
        
        # Test reset
        state = ssm.reset()
        assert state.shape == (1, state_dim)
        
        # Test forward pass
        batch_size = 4
        state = torch.randn(batch_size, state_dim)
        obs = torch.randn(batch_size, 10)
        action = torch.randn(batch_size, 5)
        reward = torch.randn(batch_size, 1)
        
        next_state, output = ssm(state, obs, action, reward)
        assert next_state.shape == (batch_size, state_dim)
        assert output.shape == (batch_size, output_dim)

def test_ssm_gradient_flow():
    """Test that gradients flow through SSM."""
    ssm = SSM(
        state_dim=64,
        hidden_dim=128,
        output_dim=32
    )
    
    # Create input tensors with gradient tracking
    batch_size = 4
    state = torch.randn(batch_size, 64, requires_grad=True)
    obs = torch.randn(batch_size, 10, requires_grad=True)
    action = torch.randn(batch_size, 5, requires_grad=True)
    reward = torch.randn(batch_size, 1, requires_grad=True)
    
    # Forward pass
    next_state, output = ssm(state, obs, action, reward)
    
    # Compute loss and backward pass
    loss = next_state.sum() + output.sum()
    loss.backward()
    
    # Verify gradients exist for all parameters
    for param in ssm.parameters():
        assert param.grad is not None
