#!/usr/bin/env python3
"""Integration test for Mamba SSM.

Verifies that the MambaSSM implementation works correctly:
1. Model creation succeeds
2. Forward pass with single timestep (B, D) input
3. Forward pass with sequence (B, T, D) input
4. Backward pass and gradient computation
5. Save/load checkpoint functionality

Expected output:
    ========================================
    Testing Mamba SSM Integration
    ========================================
    ✓ Model created
    ✓ Single timestep: (4, 32) -> (4, 32)
    ✓ Sequence: (4, 20, 32) -> (4, 20, 32)
    ✓ Backward pass successful
    ✓ Gradients computed: True
    ✓ Save/load successful
    ========================================
    All tests passed!
    ========================================
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from core.ssm_mamba import MambaSSM


def run_tests() -> None:
    """Run all Mamba SSM integration tests."""
    print("=" * 40)
    print("Testing Mamba SSM Integration")
    print("=" * 40)

    passed = 0
    total = 6

    # Test 1: Model creation
    try:
        model = MambaSSM(
            state_dim=16,
            input_dim=32,
            output_dim=32,
            d_model=64,
            d_conv=4,
            expand=2,
            device="cpu",
        )
        print(f"✓ Model created")
        print(f"  {model}")
        info = model.get_complexity_info()
        print(f"  Parameters: {info['total_params']:,}")
        print(f"  Backend: {'official mamba-ssm' if info['using_official_mamba'] else 'PyTorch fallback'}")
        print(f"  Complexity: {info['complexity']}")
        passed += 1
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return

    # Test 2: Single timestep forward pass
    try:
        batch_size = 4
        x = torch.randn(batch_size, 32)
        hidden = model.init_hidden(batch_size)
        output, next_hidden = model(x, hidden)
        assert output.shape == (batch_size, 32), (
            f"Expected output shape (4, 32), got {output.shape}"
        )
        print(f"✓ Single timestep: {tuple(x.shape)} -> {tuple(output.shape)}")
        passed += 1
    except Exception as e:
        print(f"✗ Single timestep failed: {e}")

    # Test 3: Sequence forward pass
    try:
        batch_size = 4
        seq_len = 20
        x_seq = torch.randn(batch_size, seq_len, 32)
        output_seq, _ = model(x_seq, None)
        assert output_seq.shape == (batch_size, seq_len, 32), (
            f"Expected output shape (4, 20, 32), got {output_seq.shape}"
        )
        print(f"✓ Sequence: {tuple(x_seq.shape)} -> {tuple(output_seq.shape)}")
        passed += 1
    except Exception as e:
        print(f"✗ Sequence forward failed: {e}")

    # Test 4: Backward pass
    try:
        model.zero_grad()
        x_seq = torch.randn(4, 20, 32)
        output_seq, _ = model(x_seq, None)
        loss = output_seq.mean()
        loss.backward()
        print(f"✓ Backward pass successful")
        passed += 1
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")

    # Test 5: Gradient computation
    try:
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "No gradients were computed"
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for _ in model.parameters())
        print(f"✓ Gradients computed: True ({grad_count}/{total_params} parameters)")
        passed += 1
    except Exception as e:
        print(f"✗ Gradient check failed: {e}")

    # Test 6: Save and load
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "mamba_checkpoint.pt")

            # Save
            model.save(save_path)
            assert os.path.exists(save_path), "Checkpoint file not created"

            # Load
            loaded_model = MambaSSM.load(save_path, device="cpu")

            # Verify loaded model produces same output
            x_test = torch.randn(2, 32)
            model.eval()
            loaded_model.eval()

            with torch.no_grad():
                out_original, _ = model(x_test, None)
                out_loaded, _ = loaded_model(x_test, None)

            assert torch.allclose(out_original, out_loaded, atol=1e-6), (
                f"Output mismatch after load: max diff = "
                f"{(out_original - out_loaded).abs().max().item()}"
            )

            print(f"✓ Save/load successful")
            passed += 1
    except Exception as e:
        print(f"✗ Save/load failed: {e}")

    # Summary
    print("=" * 40)
    if passed == total:
        print("All tests passed!")
    else:
        print(f"FAILED: {passed}/{total} tests passed")
    print("=" * 40)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
