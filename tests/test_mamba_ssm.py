"""Unit tests for MambaSSM implementation.

Tests cover:
    - Initialization with different configurations
    - Forward pass shapes (single-step and sequence)
    - Gradient computation and backpropagation
    - Hidden state API compatibility
    - Save and load checkpoint roundtrip
    - Device placement (CPU)
    - API compatibility with legacy SSM
"""

import os
import sys
import tempfile
import pytest
import torch
import torch.nn as nn
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.ssm_mamba import MambaSSM


class TestMambaSSMInit:
    """Tests for MambaSSM initialization."""

    def test_default_init(self):
        """Test creation with default parameters."""
        model = MambaSSM()
        assert isinstance(model, nn.Module)

    def test_custom_dims(self):
        """Test creation with custom dimensions."""
        model = MambaSSM(state_dim=32, input_dim=64, output_dim=48, d_model=128)
        assert model.state_dim == 32
        assert model.input_dim == 64
        assert model.output_dim == 48
        assert model.d_model == 128

    def test_parameter_count(self):
        """Test that model has learnable parameters."""
        model = MambaSSM(state_dim=16, input_dim=32, output_dim=32, d_model=64)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params == total_params

    def test_complexity_info(self):
        """Test complexity info reporting."""
        model = MambaSSM(state_dim=16, input_dim=32, output_dim=32)
        info = model.get_complexity_info()
        assert 'total_params' in info
        assert 'complexity' in info
        assert 'using_official_mamba' in info


class TestMambaSSMForward:
    """Tests for forward pass behavior."""

    @pytest.fixture
    def model(self):
        return MambaSSM(state_dim=16, input_dim=32, output_dim=32, d_model=64)

    def test_single_step_shape(self, model):
        """Test output shape for single timestep input (B, D)."""
        x = torch.randn(4, 32)
        hidden = model.init_hidden(batch_size=4)
        output, next_hidden = model(x, hidden)
        assert output.shape == (4, 32)

    def test_sequence_shape(self, model):
        """Test output shape for sequence input (B, T, D)."""
        x = torch.randn(4, 20, 32)
        output, next_hidden = model(x, None)
        assert output.shape == (4, 20, 32)

    def test_batch_size_1(self, model):
        """Test with batch size 1."""
        x = torch.randn(1, 32)
        output, _ = model(x, None)
        assert output.shape == (1, 32)

    def test_long_sequence(self, model):
        """Test with longer sequences."""
        x = torch.randn(2, 100, 32)
        output, _ = model(x, None)
        assert output.shape == (2, 100, 32)

    def test_hidden_state_api(self, model):
        """Test that init_hidden returns None (Mamba manages state internally)."""
        hidden = model.init_hidden(batch_size=4)
        assert hidden is None

    def test_output_values_finite(self, model):
        """Test that outputs contain only finite values."""
        x = torch.randn(2, 10, 32)
        output, _ = model(x, None)
        assert torch.isfinite(output).all()


class TestMambaSSMGradients:
    """Tests for gradient computation."""

    @pytest.fixture
    def model(self):
        return MambaSSM(state_dim=16, input_dim=32, output_dim=32, d_model=64)

    def test_backward_pass(self, model):
        """Test that backward pass computes without error."""
        x = torch.randn(2, 10, 32)
        output, _ = model(x, None)
        loss = output.mean()
        loss.backward()

    def test_gradients_exist(self, model):
        """Test that gradients are computed for all parameters."""
        x = torch.randn(2, 10, 32)
        output, _ = model(x, None)
        loss = output.mean()
        loss.backward()

        num_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total = sum(1 for _ in model.parameters())
        assert num_with_grad == total, f"Only {num_with_grad}/{total} have gradients"

    def test_gradient_values_finite(self, model):
        """Test that gradients contain only finite values."""
        x = torch.randn(2, 10, 32)
        output, _ = model(x, None)
        loss = output.mean()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


class TestMambaSSMSaveLoad:
    """Tests for checkpoint save and load."""

    def test_save_load_roundtrip(self):
        """Test that save→load produces identical outputs."""
        model = MambaSSM(state_dim=16, input_dim=32, output_dim=32, d_model=64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            model.save(path)

            assert os.path.exists(path)

            loaded = MambaSSM.load(path, device='cpu')

            # Verify identical output
            x = torch.randn(2, 32)
            model.eval()
            loaded.eval()

            with torch.no_grad():
                out1, _ = model(x, None)
                out2, _ = loaded(x, None)

            assert torch.allclose(out1, out2, atol=1e-6)

    def test_save_load_config_preserved(self):
        """Test that model configuration is preserved after load."""
        model = MambaSSM(state_dim=32, input_dim=64, output_dim=48, d_model=128)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            model.save(path)
            loaded = MambaSSM.load(path, device='cpu')

            assert loaded.state_dim == 32
            assert loaded.input_dim == 64
            assert loaded.output_dim == 48
            assert loaded.d_model == 128


class TestMambaSSMCompatibility:
    """Tests for API compatibility with Legacy SSM."""

    def test_same_interface_as_legacy(self):
        """Test that MambaSSM has the same public methods as Legacy SSM."""
        model = MambaSSM()
        required_methods = ['forward', 'init_hidden', 'save', 'load']
        for method in required_methods:
            assert hasattr(model, method), f"Missing method: {method}"

    def test_forward_signature_compatibility(self):
        """Test that forward(x, hidden_state) works like Legacy SSM."""
        model = MambaSSM(state_dim=16, input_dim=32, output_dim=32)
        x = torch.randn(1, 32)
        hidden = model.init_hidden(batch_size=1)

        # Should work with both positional and keyword args
        output, new_hidden = model(x, hidden)
        assert output.shape == (1, 32)

        output2, _ = model(x, hidden_state=hidden)
        assert output2.shape == (1, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
