#!/usr/bin/env python3
"""Test file for core SSM module."""

print("=" * 50)
print("TEST: core/test_ssm.py")
print("=" * 50)

try:
    # Import SSM components
    print("\n[1/4] Importing SSM and SSMConfig...")
    from ssm import SSM, SSMConfig
    print("✓ Import successful")
    
    # Initialize parameters
    print("\n[2/4] Initializing SSM parameters...")
    config = SSMConfig(
        state_dim=64,
        hidden_dim=128,
        output_dim=32
    )
    ssm = SSM(config)
    print(f"✓ SSM initialized with state_dim={config.state_dim}, hidden_dim={config.hidden_dim}")
    
    # Run reset smoke test
    print("\n[3/4] Running reset smoke test...")
    state = ssm.reset()
    print(f"✓ Reset successful, state shape: {state.shape if hasattr(state, 'shape') else type(state)}")
    
    # Run act smoke test
    print("\n[4/4] Running act smoke test...")
    import numpy as np
    dummy_input = np.random.randn(config.state_dim)
    output = ssm.act(dummy_input)
    print(f"✓ Act successful, output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
    
    print("\n" + "=" * 50)
    print("✓ ALL TESTS PASSED")
    print("=" * 50)
    
except ImportError as e:
    print(f"✗ Import Error: {e}")
    print("Note: SSM module may not be implemented yet.")
    print("Expected: core/ssm.py with SSM and SSMConfig classes")
    
except Exception as e:
    print(f"✗ Test Error: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
