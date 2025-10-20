#!/usr/bin/env python3
"""
Test file for core SSM module - 100% aligned with core/ssm.py implementation.
API under test:
- SSM(state_dim, input_dim, output_dim, hidden_dim=128, device='cpu')
- forward(x, hidden_state=None) -> single tensor
"""
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
    """
    Test SSM initialization with exact API:
    SSM(state_dim, input_dim, output_dim, hidden_dim=128, device='cpu')
    """
    # Test with exact defaults
    ssm = SSM(state_dim=10, input_dim=5, output_dim=32, hidden_dim=128, device='cpu')
    
    # Verify SSM is a PyTorch module
    assert isinstance(ssm, torch.nn.Module)
    assert type(ssm).__name__ == 'SSM'

def test_ssm_forward_returns_single_tensor():
    """
    Test that forward() returns single tensor, not tuple.
    API: forward(x, hidden_state=None) -> single tensor
    """
    ssm = SSM(state_dim=10, input_dim=5, output_dim=32, hidden_dim=128, device='cpu')
    
    batch_size = 16
    state = torch.randn(batch_size, 5)  # input_dim (5) used
    
    # forward() must return single tensor
    output = ssm.forward(state, hidden_state=None)
    
    # Critical assertion: output must be tensor, not tuple
    assert isinstance(output, torch.Tensor), f"Expected Tensor, got {type(output)}"
    assert output.shape == (batch_size, 32)  # output_dim (32) confirmed

def test_ssm_forward_with_hidden_state():
    """
    Test forward pass with optional hidden_state parameter.
    API: forward(x, hidden_state=None) -> single tensor
    """
    ssm = SSM(state_dim=10, input_dim=5, output_dim=32, hidden_dim=128, device='cpu')
    
    batch_size = 16
    state = torch.randn(batch_size, 5)  # input_dim (5) used
    hidden_state = torch.randn(batch_size, 10)  # state_dim=10
    
    # Call with explicit hidden_state
    output = ssm.forward(state, hidden_state=hidden_state)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 32)  # output_dim (32)

def test_ssm_batch_processing():
    """
    Test SSM handles different batch sizes correctly.
    """
    ssm = SSM(state_dim=10, input_dim=5, output_dim=32, hidden_dim=128, device='cpu')
    
    for batch_size in [1, 8, 32, 64]:
        state = torch.randn(batch_size, 5)  # input_dim (5)
        output = ssm.forward(state)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, 32)  # output_dim (32)

def test_ssm_device_placement():
    """
    Test SSM respects device parameter.
    API: SSM(..., device='cpu') or SSM(..., device='cuda')
    """
    # Test CPU
    ssm_cpu = SSM(state_dim=10, input_dim=5, output_dim=32, hidden_dim=128, device='cpu')
    state_cpu = torch.randn(8, 5)  # input_dim (5)
    output_cpu = ssm_cpu.forward(state_cpu)
    assert output_cpu.device.type == 'cpu'
    
    # Test CUDA if available
    if torch.cuda.is_available():
        ssm_cuda = SSM(state_dim=10, input_dim=5, output_dim=32, hidden_dim=128, device='cuda')
        state_cuda = torch.randn(8, 5).cuda()  # input_dim (5)
        output_cuda = ssm_cuda.forward(state_cuda)
        assert output_cuda.device.type == 'cuda'

def test_ssm_custom_dimensions():
    """
    Test SSM with custom dimensions.
    """
    state_dim = 20
    input_dim = 15
    hidden_dim = 256
    output_dim = 64
    
    ssm = SSM(state_dim=state_dim, input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, device='cpu')
    
    batch_size = 16
    state = torch.randn(batch_size, input_dim)  # input_dim used
    output = ssm.forward(state)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, output_dim)

def test_ssm_gradient_flow():
    """
    Test that gradients flow through SSM correctly.
    """
    ssm = SSM(state_dim=10, input_dim=5, output_dim=32, hidden_dim=128, device='cpu')
    
    state = torch.randn(8, 5, requires_grad=True)  # input_dim (5)
    output = ssm.forward(state)
    
    # Compute loss and backward
    loss = output.sum()
    loss.backward()
    
    # Check gradients exist
    assert state.grad is not None
    assert state.grad.shape == state.shape

def test_ssm_save_load():
    """
    Test SSM can be saved and loaded correctly.
    """
    ssm_original = SSM(state_dim=10, input_dim=5, output_dim=32, hidden_dim=128, device='cpu')
    
    # Save model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
        torch.save(ssm_original.state_dict(), temp_path)
    
    try:
        # Load model
        ssm_loaded = SSM(state_dim=10, input_dim=5, output_dim=32, hidden_dim=128, device='cpu')
        ssm_loaded.load_state_dict(torch.load(temp_path))
        
        # Test outputs match
        state = torch.randn(8, 5)  # input_dim (5)
        output_original = ssm_original.forward(state)
        output_loaded = ssm_loaded.forward(state)
        
        assert torch.allclose(output_original, output_loaded, atol=1e-6)
    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
