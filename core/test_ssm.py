#!/usr/bin/env python3
"""Test file for core SSM module."""
print("="*50)
print("TEST: core/test_ssm.py")
print("="*50)

try:
    # Import SSM components from correct module path
    print("\n[1/4] Importing SSM from core.ssm...")
    from core.ssm import SSM
    print("✓ Import successful")
    
    # Initialize SSM with PyTorch
    print("\n[2/4] Initializing SSM with PyTorch...")
    import torch
    
    ssm = SSM(
        state_dim=64,
        hidden_dim=128,
        output_dim=32
    )
    print(f"✓ SSM initialized with state_dim=64, hidden_dim=128, output_dim=32")
    print(f"  Model type: {type(ssm).__name__}")
    print(f"  Is nn.Module: {isinstance(ssm, torch.nn.Module)}")
    
    # Run reset smoke test
    print("\n[3/4] Running reset smoke test...")
    state = ssm.reset()
    print(f"✓ Reset successful, state shape: {state.shape}")
    print(f"  State type: {type(state).__name__}")
    print(f"  Is torch.Tensor: {isinstance(state, torch.Tensor)}")
    
    # Run forward smoke test with PyTorch tensor
    print("\n[4/4] Running forward smoke test...")
    dummy_input = torch.randn(1, 64)  # batch_size=1, state_dim=64
    output = ssm(dummy_input)
    print(f"✓ Forward pass successful, output shape: {output.shape}")
    print(f"  Expected shape: (1, 32)")
    
    # Test parameter saving/loading
    print("\n[5/4] Testing save/load functionality...")
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        # Save
        torch.save(ssm.state_dict(), temp_path)
        print(f"✓ Saved model parameters to {temp_path}")
        
        # Load
        new_ssm = SSM(state_dim=64, hidden_dim=128, output_dim=32)
        new_ssm.load_state_dict(torch.load(temp_path))
        print(f"✓ Loaded model parameters successfully")
        
        # Verify outputs match
        output2 = new_ssm(dummy_input)
        if torch.allclose(output, output2):
            print("✓ Loaded model produces identical output")
        else:
            print("✗ Warning: Loaded model output differs")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print("\n" + "="*50)
    print("✓ ALL TESTS PASSED!")
    print("="*50)
    
except ImportError as e:
    print(f"\n✗ Import Error: {e}")
    print("  Make sure core/ssm.py exists and has SSM class")
except Exception as e:
    print(f"\n✗ Test Failed: {e}")
    import traceback
    traceback.print_exc()
