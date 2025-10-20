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
        Dict with 'observations' and 'targets' keys
    """
    return {
        'observations': torch.randn(batch_size, state_dim),
        'targets': torch.randn(batch_size, action_dim)
    }

def simple_loss_fn(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Simple MSE loss function that works with dict batches.
    
    Args:
        model: The model to evaluate
        batch: Dict with 'observations' and 'targets'
    
    Returns:
        Loss tensor
    """
    obs = batch['observations']
    targets = batch['targets']
    predictions = model(obs)
    return F.mse_loss(predictions, targets)

def test_adapter(state_dim: int = 10, action_dim: int = 4, batch_size: int = 32):
    """
    Test the Adapter API with correct configuration.
    Only uses lr and max_steps_per_call in AdaptationConfig.
    """
    print("\n" + "="*60)
    print("Testing Adapter API")
    print("="*60)
    
    # Create SSM model
    ssm_model = SSM(
        state_dim=state_dim,
        action_dim=action_dim,
        ssm_dim=64,
        num_layers=2
    )
    
    # Create AdaptationConfig with ONLY lr and max_steps_per_call
    config = AdaptationConfig(
        lr=0.001,
        max_steps_per_call=5
    )
    
    # Create Adapter
    adapter = Adapter(model=ssm_model, config=config)
    
    # Generate batch as dict
    batch = generate_dummy_batch(batch_size, state_dim, action_dim)
    
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Observations shape: {batch['observations'].shape}")
    print(f"Targets shape: {batch['targets'].shape}")
    
    # Test adaptation - pass loss_fn and batch (as dict)
    print("\nCalling adapter.adapt(loss_fn=simple_loss_fn, batch=batch)...")
    start_time = time.time()
    result = adapter.adapt(loss_fn=simple_loss_fn, batch=batch)
    adapt_time = time.time() - start_time
    
    print(f"Adaptation completed in {adapt_time:.4f}s")
    print(f"Result type: {type(result)}")
    print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
    
    # Validate result structure
    if isinstance(result, dict):
        if 'adapted_params' in result:
            print(f"✓ 'adapted_params' present")
        if 'losses' in result:
            print(f"✓ 'losses' present, length: {len(result['losses'])}")
            print(f"  Loss trajectory: {result['losses']}")
    
    print("\n✓ Adapter test completed successfully")
    return adapter, result

def test_meta_maml(state_dim: int = 10, action_dim: int = 4, num_tasks: int = 4, batch_size: int = 16):
    """
    Test the MetaMAML API with correct constructor signature.
    Only uses inner_lr, outer_lr, and first_order parameters.
    """
    print("\n" + "="*60)
    print("Testing MetaMAML API")
    print("="*60)
    
    # Create SSM model
    ssm_model = SSM(
        state_dim=state_dim,
        action_dim=action_dim,
        ssm_dim=64,
        num_layers=2
    )
    
    # Create MetaMAML with ONLY inner_lr, outer_lr, first_order
    print("Creating MetaMAML(model, inner_lr=0.01, outer_lr=0.001, first_order=True)...")
    meta_maml = MetaMAML(
        model=ssm_model,
        inner_lr=0.01,
        outer_lr=0.001,
        first_order=True
    )
    
    # Generate tasks (list of dicts)
    print(f"\nGenerating {num_tasks} tasks...")
    tasks = []
    for i in range(num_tasks):
        support = generate_dummy_batch(batch_size, state_dim, action_dim)
        query = generate_dummy_batch(batch_size, state_dim, action_dim)
        tasks.append({
            'support': support,
            'query': query
        })
    
    print(f"Task 0 keys: {list(tasks[0].keys())}")
    print(f"Task 0 support keys: {list(tasks[0]['support'].keys())}")
    print(f"Task 0 query keys: {list(tasks[0]['query'].keys())}")
    
    # Test meta-update
    print("\nCalling meta_maml.meta_update(tasks, loss_fn=simple_loss_fn)...")
    start_time = time.time()
    result = meta_maml.meta_update(tasks, loss_fn=simple_loss_fn)
    meta_time = time.time() - start_time
    
    print(f"Meta-update completed in {meta_time:.4f}s")
    print(f"Result type: {type(result)}")
    
    # Validate result
    if isinstance(result, dict):
        print(f"Result keys: {list(result.keys())}")
        if 'meta_loss' in result:
            print(f"✓ 'meta_loss' present: {result['meta_loss']:.4f}")
        if 'task_losses' in result:
            print(f"✓ 'task_losses' present, count: {len(result['task_losses'])}")
    else:
        print(f"Result value: {result}")
    
    print("\n✓ MetaMAML test completed successfully")
    return meta_maml, result

def main():
    """
    Main benchmark runner.
    Tests both Adapter and MetaMAML with strict API compliance.
    """
    print("\n" + "#"*60)
    print("# SSM-MetaRL Quick Benchmark")
    print("# Testing API Compliance: Adapter & MetaMAML")
    print("#"*60)
    
    torch.manual_seed(42)
    
    # Test 1: Adapter API
    try:
        adapter, adapter_result = test_adapter()
        print("\n[PASS] Adapter API test")
    except Exception as e:
        print(f"\n[FAIL] Adapter API test: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: MetaMAML API
    try:
        meta_maml, maml_result = test_meta_maml()
        print("\n[PASS] MetaMAML API test")
    except Exception as e:
        print(f"\n[FAIL] MetaMAML API test: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final summary
    print("\n" + "#"*60)
    print("# Benchmark Summary")
    print("#"*60)
    print("✓ All API tests passed")
    print("✓ Adapter: Correct AdaptationConfig (lr, max_steps_per_call only)")
    print("✓ Adapter.adapt: Uses loss_fn and batch dict")
    print("✓ MetaMAML: Correct constructor (inner_lr, outer_lr, first_order only)")
    print("✓ All batch data passed as dicts with 'observations' and 'targets'")
    print("\nBenchmark completed successfully!\n")

if __name__ == "__main__":
    main()
