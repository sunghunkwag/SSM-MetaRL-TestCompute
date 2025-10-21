#!/usr/bin/env python3
"""
Test file for core SSM module - Moved to tests/ directory.
API under test:
- SSM(state_dim, input_dim, output_dim, hidden_dim=128, device='cpu')
- init_hidden(batch_size=1) -> hidden_state
- forward(x, hidden_state) -> (output, next_hidden_state)
"""
import pytest
import torch
import tempfile
import os
import sys
from pathlib import Path

# Add project root to sys.path to allow importing 'core'
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.ssm import SSM # Now imports correctly from tests/

def test_ssm_import():
    """Test that SSM can be imported successfully."""
    assert SSM is not None

def test_ssm_initialization():
    """
    Test SSM initialization with exact API:
    SSM(state_dim, input_dim, output_dim, hidden_dim=128, device='cpu')
    """
    ssm = SSM(state_dim=10, input_dim=5, output_dim=32, hidden_dim=128, device='cpu')
    assert isinstance(ssm, torch.nn.Module)
    assert type(ssm).__name__ == 'SSM'

def test_init_hidden():
    """Test hidden state initialization."""
    state_dim = 10
    ssm = SSM(state_dim=state_dim, input_dim=5, output_dim=32)
    batch_size = 16
    hidden = ssm.init_hidden(batch_size)
    assert isinstance(hidden, torch.Tensor)
    assert hidden.shape == (batch_size, state_dim)
    assert torch.all(hidden == 0)

def test_ssm_forward_returns_tuple():
    """
    Test that forward() returns a tuple (output, next_hidden_state).
    API: forward(x, hidden_state) -> (output, next_hidden_state)
    """
    state_dim = 10
    input_dim = 5
    output_dim = 32
    ssm = SSM(state_dim=state_dim, input_dim=input_dim, output_dim=output_dim)

    batch_size = 16
    x = torch.randn(batch_size, input_dim)
    hidden = ssm.init_hidden(batch_size)

    # forward() must return tuple
    result = ssm.forward(x, hidden)

    # Critical assertion: output must be tuple of two tensors
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected tuple of length 2, got {len(result)}"

    output, next_hidden = result
    assert isinstance(output, torch.Tensor), f"Expected output Tensor, got {type(output)}"
    assert isinstance(next_hidden, torch.Tensor), f"Expected next_hidden Tensor, got {type(next_hidden)}"

    assert output.shape == (batch_size, output_dim)
    assert next_hidden.shape == (batch_size, state_dim)

def test_ssm_batch_processing():
    """Test SSM handles different batch sizes correctly."""
    state_dim = 10
    input_dim = 5
    output_dim = 32
    ssm = SSM(state_dim=state_dim, input_dim=input_dim, output_dim=output_dim)

    for batch_size in [1, 8, 32, 64]:
        x = torch.randn(batch_size, input_dim)
        hidden = ssm.init_hidden(batch_size)
        output, next_hidden = ssm.forward(x, hidden)

        assert isinstance(output, torch.Tensor)
        assert isinstance(next_hidden, torch.Tensor)
        assert output.shape == (batch_size, output_dim)
        assert next_hidden.shape == (batch_size, state_dim)

def test_ssm_device_placement():
    """Test SSM respects device parameter."""
    state_dim = 10
    input_dim = 5
    output_dim = 32

    # Test CPU
    ssm_cpu = SSM(state_dim=state_dim, input_dim=input_dim, output_dim=output_dim, device='cpu')
    x_cpu = torch.randn(8, input_dim)
    hidden_cpu = ssm_cpu.init_hidden(8)
    output_cpu, next_hidden_cpu = ssm_cpu.forward(x_cpu, hidden_cpu)
    assert output_cpu.device.type == 'cpu'
    assert next_hidden_cpu.device.type == 'cpu'

    # Test CUDA if available
    if torch.cuda.is_available():
        ssm_cuda = SSM(state_dim=state_dim, input_dim=input_dim, output_dim=output_dim, device='cuda')
        x_cuda = torch.randn(8, input_dim).cuda()
        hidden_cuda = ssm_cuda.init_hidden(8) # init_hidden places on correct device
        output_cuda, next_hidden_cuda = ssm_cuda.forward(x_cuda, hidden_cuda)
        assert output_cuda.device.type == 'cuda'
        assert next_hidden_cuda.device.type == 'cuda'

def test_ssm_custom_dimensions():
    """Test SSM with custom dimensions."""
    state_dim = 20
    input_dim = 15
    hidden_dim_nn = 256 # internal hidden dim of NN layers
    output_dim = 64

    ssm = SSM(state_dim=state_dim, input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim_nn)

    batch_size = 16
    x = torch.randn(batch_size, input_dim)
    hidden = ssm.init_hidden(batch_size)
    output, next_hidden = ssm.forward(x, hidden)

    assert isinstance(output, torch.Tensor)
    assert isinstance(next_hidden, torch.Tensor)
    assert output.shape == (batch_size, output_dim)
    assert next_hidden.shape == (batch_size, state_dim)


def test_ssm_gradient_flow():
    """Test that gradients flow through SSM correctly."""
    state_dim = 10
    input_dim = 5
    output_dim = 32
    ssm = SSM(state_dim=state_dim, input_dim=input_dim, output_dim=output_dim)

    x = torch.randn(8, input_dim, requires_grad=True)
    hidden = ssm.init_hidden(8)
    hidden.requires_grad_(True) # Also check grads flow through hidden state

    output, next_hidden = ssm.forward(x, hidden)

    # Compute loss and backward
    loss = output.sum() + next_hidden.sum() # Include next_hidden in loss
    loss.backward()

    # Check gradients exist
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert hidden.grad is not None
    assert hidden.grad.shape == hidden.shape

def test_ssm_save_load():
    """Test SSM can be saved and loaded correctly."""
    ssm_original = SSM(state_dim=10, input_dim=5, output_dim=32, hidden_dim=128, device='cpu')

    # Save model state_dict only
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
        # Save model config separately or pass it when loading
        torch.save(ssm_original.state_dict(), temp_path)

    try:
        # Load model requires config first
        ssm_loaded = SSM(state_dim=10, input_dim=5, output_dim=32, hidden_dim=128, device='cpu')
        ssm_loaded.load_state_dict(torch.load(temp_path))

        # Test outputs match
        x = torch.randn(8, 5)
        hidden = ssm_original.init_hidden(8)

        output_original, next_hidden_original = ssm_original.forward(x, hidden)
        output_loaded, next_hidden_loaded = ssm_loaded.forward(x, hidden)

        assert torch.allclose(output_original, output_loaded, atol=1e-6)
        assert torch.allclose(next_hidden_original, next_hidden_loaded, atol=1e-6)

    finally:
        os.unlink(temp_path)

# Test the static load method as well (if needed, but state_dict is standard)
def test_ssm_static_load_method():
    """Test the static load method which saves/loads config."""
    config = {'state_dim': 10, 'input_dim': 5, 'output_dim': 32, 'hidden_dim': 128, 'device': 'cpu'}
    ssm_original = SSM(**config)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
        ssm_original.save(temp_path) # Uses the instance save method

    try:
        ssm_loaded = SSM.load(temp_path) # Uses the static load method

        # Test config matches
        assert ssm_loaded.state_dim == config['state_dim']
        assert ssm_loaded.input_dim == config['input_dim']
        assert ssm_loaded.output_dim == config['output_dim']
        assert ssm_loaded.hidden_dim == config['hidden_dim']

        # Test outputs match
        x = torch.randn(8, 5)
        hidden = ssm_original.init_hidden(8)
        output_original, _ = ssm_original.forward(x, hidden)
        output_loaded, _ = ssm_loaded.forward(x, hidden)
        assert torch.allclose(output_original, output_loaded, atol=1e-6)

    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
