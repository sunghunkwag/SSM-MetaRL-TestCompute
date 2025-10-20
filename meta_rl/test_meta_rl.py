#!/usr/bin/env python3
"""Test file for meta_rl module."""
print("="*50)
print("TEST: meta_rl/test_meta_rl.py")
print("="*50)

try:
    # Import MAML components from correct module
    print("\n[1/3] Importing MetaMAML...")
    from meta_rl.meta_maml import MetaMAML
    print("✓ Import successful")
    
    # Initialize MetaMAML
    print("\n[2/3] Initializing MetaMAML...")
    import torch
    
    # Create a simple model for testing
    test_model = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )
    
    meta_maml = MetaMAML(
        model=test_model,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5
    )
    print(f"✓ MetaMAML initialized with inner_lr=0.01, outer_lr=0.001, inner_steps=5")
    
    # Run dummy adaptation test
    print("\n[3/3] Testing adaptation with dummy data...")
    
    # Create dummy data
    import torch
    support_x = torch.randn(10, 10)
    support_y = torch.randn(10, 1)
    query_x = torch.randn(5, 10)
    query_y = torch.randn(5, 1)
    
    # Test inner loop adaptation
    adapted_params = meta_maml.inner_loop(support_x, support_y)
    print(f"✓ Inner loop adaptation completed, got {len(adapted_params)} adapted parameters")
    
    # Test outer loop
    loss = meta_maml.outer_loop([(support_x, support_y, query_x, query_y)])
    print(f"✓ Outer loop completed with loss: {loss:.4f}")
    
    print("\n" + "="*50)
    print("✓ ALL TESTS PASSED!")
    print("="*50)
    
except ImportError as e:
    print(f"\n✗ Import Error: {e}")
    print("  Make sure meta_rl/meta_maml.py exists and has MetaMAML class")
except Exception as e:
    print(f"\n✗ Test Failed: {e}")
    import traceback
    traceback.print_exc()
