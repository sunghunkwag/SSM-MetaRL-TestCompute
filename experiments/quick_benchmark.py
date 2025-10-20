#!/usr/bin/env python3
"""
Quick Benchmark Script for SSM-MetaRL
Directly tests Adapter and MetaMAML APIs with correct batch dict structure.
Designed to validate API correctness and basic functionality.

Usage:
    python experiments/quick_benchmark.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from core.ssm import SSM
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from meta_rl.meta_maml import MetaMAML


def generate_dummy_batch(batch_size: int, state_dim: int, action_dim: int) -> Dict[str, torch.Tensor]:
    """
    Generate dummy batch data in dict format as expected by Adapter.adapt().
    
    Args:
        batch_size: Number of samples
        state_dim: State dimension
        action_dim: Action dimension (used as target dimension)
    
    Returns:
        Dictionary with 'observations' and 'targets' keys
    """
    return {
        'observations': torch.randn(batch_size, state_dim),
        'targets': torch.randn(batch_size, action_dim)
    }


def test_adapter_api():
    """
    Test Adapter API with correct AdaptationConfig and dict batch structure.
    """
    print("\n" + "="*60)
    print("Testing Adapter API")
    print("="*60)
    
    # Create SSM model
    state_dim = 8
    action_dim = 4
    hidden_dim = 16
    
    ssm = SSM(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )
    
    print(f"✓ Created SSM model (state_dim={state_dim}, action_dim={action_dim})")
    
    # Create Adapter with AdaptationConfig (correct API)
    adapt_cfg = AdaptationConfig(
        lr=0.01,
        max_steps_per_call=5,
        optimizer='adam'
    )
    
    adapter = Adapter(
        target=ssm,
        cfg=adapt_cfg,
        strategy='none'
    )
    
    print(f"✓ Created Adapter with cfg (lr={adapt_cfg.lr}, max_steps={adapt_cfg.max_steps_per_call})")
    
    # Define loss function
    def loss_fn(model_output: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Loss function matching Adapter.adapt() signature."""
        targets = batch['targets']
        return F.mse_loss(model_output, targets)
    
    # Generate batch as dict (correct API)
    batch_size = 16
    batch = generate_dummy_batch(batch_size, state_dim, action_dim)
    
    print(f"✓ Generated batch dict with keys: {list(batch.keys())}")
    print(f"  - observations shape: {batch['observations'].shape}")
    print(f"  - targets shape: {batch['targets'].shape}")
    
    # Test adaptation
    start_time = time.time()
    
    try:
        # Observe meta-features
        adapter.observe(batch)
        print("✓ Called adapter.observe(batch)")
        
        # Adapt model (batch must be dict)
        adapted_params = adapter.adapt(loss_fn=loss_fn, batch=batch)
        elapsed = time.time() - start_time
        
        print(f"✓ Called adapter.adapt(loss_fn=loss_fn, batch=batch)")
        print(f"✓ Adaptation completed in {elapsed:.3f}s")
        print(f"✓ Returned {len(adapted_params) if adapted_params else 0} adapted parameters")
        
        return True, "Adapter API test passed"
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Adapter API test failed after {elapsed:.3f}s")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_meta_maml_api():
    """
    Test MetaMAML API with correct inner_loop signature (no criterion argument).
    """
    print("\n" + "="*60)
    print("Testing MetaMAML API")
    print("="*60)
    
    # Create SSM model
    state_dim = 8
    action_dim = 4
    hidden_dim = 16
    
    ssm = SSM(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )
    
    print(f"✓ Created SSM model (state_dim={state_dim}, action_dim={action_dim})")
    
    # Create MetaMAML with correct API
    meta_maml = MetaMAML(
        model=ssm,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        first_order=False
    )
    
    print(f"✓ Created MetaMAML (inner_lr=0.01, outer_lr=0.001, inner_steps=5)")
    
    # Generate support and query sets
    batch_size = 16
    support_x = torch.randn(batch_size, state_dim)
    support_y = torch.randn(batch_size, action_dim)
    query_x = torch.randn(batch_size, state_dim)
    query_y = torch.randn(batch_size, action_dim)
    
    print(f"✓ Generated support and query sets")
    print(f"  - support_x shape: {support_x.shape}")
    print(f"  - support_y shape: {support_y.shape}")
    
    start_time = time.time()
    
    try:
        # Test inner loop (no criterion argument in actual API)
        adapted_params = meta_maml.inner_loop(support_x, support_y)
        print(f"✓ Called meta_maml.inner_loop(support_x, support_y) - no criterion arg")
        print(f"✓ Returned {len(adapted_params)} adapted parameters")
        
        # Test outer loop
        loss = meta_maml.outer_loop(query_x, query_y, adapted_params)
        elapsed = time.time() - start_time
        
        print(f"✓ Called meta_maml.outer_loop(query_x, query_y, adapted_params)")
        print(f"✓ Outer loss: {loss.item():.4f}")
        print(f"✓ MetaMAML test completed in {elapsed:.3f}s")
        
        return True, "MetaMAML API test passed"
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ MetaMAML API test failed after {elapsed:.3f}s")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_batch_dict_compliance():
    """
    Test that batch dict structure is correctly used throughout.
    """
    print("\n" + "="*60)
    print("Testing Batch Dict Compliance")
    print("="*60)
    
    batch_size = 8
    state_dim = 4
    action_dim = 2
    
    # Generate batch as dict
    batch = generate_dummy_batch(batch_size, state_dim, action_dim)
    
    print(f"✓ Batch is dict: {isinstance(batch, dict)}")
    print(f"✓ Batch keys: {list(batch.keys())}")
    
    # Verify required keys
    required_keys = ['observations', 'targets']
    missing_keys = [k for k in required_keys if k not in batch]
    
    if missing_keys:
        print(f"✗ Missing required keys: {missing_keys}")
        return False, f"Missing keys: {missing_keys}"
    
    print(f"✓ All required keys present: {required_keys}")
    
    # Verify types
    for key in required_keys:
        if not isinstance(batch[key], torch.Tensor):
            print(f"✗ batch['{key}'] is not a torch.Tensor")
            return False, f"batch['{key}'] is not a torch.Tensor"
    
    print(f"✓ All values are torch.Tensors")
    print(f"✓ Batch dict compliance test passed")
    
    return True, "Batch dict compliance test passed"


def main():
    print("\n" + "#"*60)
    print("# SSM-MetaRL Quick Benchmark")
    print("# Testing API Correctness")
    print("#"*60)
    
    results = []
    
    # Run all tests
    tests = [
        ("Batch Dict Compliance", test_batch_dict_compliance),
        ("Adapter API", test_adapter_api),
        ("MetaMAML API", test_meta_maml_api),
    ]
    
    for test_name, test_func in tests:
        try:
            success, message = test_func()
            results.append((test_name, success, message))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n✗ {test_name} crashed: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, message in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not success:
            print(f"  → {message}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All API correctness tests passed!")
        print("\n💡 Key Points Verified:")
        print("  - Adapter uses cfg=AdaptationConfig(lr=..., max_steps_per_call=...)")
        print("  - Adapter.adapt() accepts batch as dict with 'observations' and 'targets'")
        print("  - MetaMAML.inner_loop() takes only (support_x, support_y) - no criterion arg")
        print("  - All batch structures use dict format, not tuples")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
