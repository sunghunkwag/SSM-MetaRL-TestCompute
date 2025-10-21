# -*- coding: utf-8 -*-
"""
Quick benchmark script for SSM-MetaRL-TestCompute.
Benchmarks both MetaMAML and Test-Time Adaptation.

Validates:
- MetaMAML.adapt() returns OrderedDict (fast_weights)
- Adapter.adapt() returns dict with 'loss' key
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter, AdaptationConfig


def benchmark_meta_maml():
    """
    Benchmark MetaMAML adaptation.
    Validates that adapt() returns OrderedDict.
    """
    print("\n" + "="*60)
    print("BENCHMARK: MetaMAML")
    print("="*60)
    
    # Setup
    D_in, D_out = 4, 1
    model = StateSpaceModel(state_dim=4, input_dim=D_in, output_dim=D_out)
    maml = MetaMAML(model=model, inner_lr=0.01, outer_lr=0.001)
    
    # Create task data
    B, T = 8, 10
    
    # [FIX] Reshape data and create targets
    support_x = torch.randn(B, T, D_in).reshape(B * T, D_in)
    support_y = torch.randn(B * T, D_out)
    
    print("\nRunning MetaMAML adaptation...")
    
    # Adapt - should return OrderedDict
    # [FIXED] Call adapt with correct args: support_x, support_y, and 'num_steps'
    fast_weights = maml.adapt(support_x, support_y, num_steps=5)
    
    # Validate return type
    print(f"\nReturn type: {type(fast_weights)}")
    assert isinstance(fast_weights, OrderedDict), \
        f"ERROR: Expected OrderedDict, got {type(fast_weights)}"
    print("✓ PASS: adapt() returns OrderedDict")
    
    # Validate contents
    print(f"\nNumber of parameters: {len(fast_weights)}")
    assert len(fast_weights) > 0, "ERROR: fast_weights is empty"
    print("✓ PASS: fast_weights contains parameters")
    
    # Validate parameter types
    for key, value in fast_weights.items():
        assert isinstance(value, torch.Tensor), \
            f"ERROR: Expected tensor for {key}, got {type(value)}"
    print("✓ PASS: All parameters are tensors")
    
    print("\n" + "="*60)
    print("MetaMAML Benchmark: SUCCESS")
    print("="*60)


def benchmark_adapter():
    """
    Benchmark Test-Time Adapter.
    Validates that adapt() returns dict with 'loss' key.
    """
    print("\n" + "="*60)
    print("BENCHMARK: Test-Time Adapter")
    print("="*60)
    
    # Setup
    model = StateSpaceModel(state_dim=4, input_dim=4, output_dim=1)
    config = AdaptationConfig(
        lr=0.01,
        grad_clip_norm=1.0,
        trust_region_eps=0.01,
        ema_decay=0.99,
        entropy_weight=0.01,
        max_steps_per_call=5
    )
    adapter = Adapter(model, config)
    
    loss_fn = nn.MSELoss()
    
    # Data must be 2D for SSM
    states = torch.randn(8, 4)
    targets = torch.randn(8, 1)
    
    # [FIX] batch_dict keys must match model 'forward(x, ...)'
    batch_dict = {'x': states, 'targets': targets}
    
    # Define wrapper functions for fwd and loss
    def fwd_fn(batch):
        return adapter.target(batch['x'])
        
    def loss_fn_wrapper(outputs, batch):
        return loss_fn(outputs, batch['targets'])
        
    print("\nRunning Adapter adaptation...")
    
    # Adapt - should return dict with 'loss' key
    info = adapter.adapt(loss_fn_wrapper, batch_dict, fwd_fn=fwd_fn)
    
    # Validate return type
    print(f"\nReturn type: {type(info)}")
    assert isinstance(info, dict), \
        f"ERROR: Expected dict, got {type(info)}"
    print("✓ PASS: adapt() returns dict")
    
    # Validate 'loss' key
    print(f"\nKeys in result: {list(info.keys())}")
    assert 'loss' in info, \
        f"ERROR: Expected 'loss' key, got keys: {info.keys()}"
    print("✓ PASS: Result contains 'loss' key")
    
    # Extract and validate loss
    loss = info['loss']
    print(f"\nLoss value: {loss}")
    assert isinstance(loss, (float, int, torch.Tensor)), \
        f"ERROR: Expected numeric loss, got {type(loss)}"
    print("✓ PASS: Loss is numeric value")
    
    # Validate 'steps' key if present
    if 'steps' in info:
        steps = info['steps']
        print(f"Steps taken: {steps}")
        assert isinstance(steps, int), \
            f"ERROR: Expected int for steps, got {type(steps)}"
        print("✓ PASS: Steps is integer value")
    
    print("\n" + "="*60)
    print("Adapter Benchmark: SUCCESS")
    print("="*60)


def run_all_benchmarks():
    """
    Run all benchmarks and report results.
    """
    print("\n" + "#"*60)
    print("# SSM-MetaRL-TestCompute Quick Benchmark Suite")
    print("#" + " "*58 + "#")
    print("# Testing return value consistency:")
    print("#   - MetaMAML.adapt() → OrderedDict (fast_weights)")
    print("#   - Adapter.adapt() → dict with 'loss' key")
    print("#"*60)
    
    try:
        # Run MetaMAML benchmark
        benchmark_meta_maml()
        
        # Run Adapter benchmark
        benchmark_adapter()
        
        # Summary
        print("\n" + "#"*60)
        print("# ALL BENCHMARKS PASSED ✓")
        print("#"*60)
        print("\nSummary:")
        print("  • MetaMAML.adapt() correctly returns OrderedDict")
        print("  • Adapter.adapt() correctly returns dict with 'loss'")
        print("  • All type assertions passed")
        print("  • Return value consistency verified")
        print("\n" + "="*60)
        
        return True
        
    except AssertionError as e:
        print("\n" + "!"*60)
        print("! BENCHMARK FAILED")
        print("!"*60)
        print(f"\nError: {e}")
        print("\n" + "="*60)
        return False
    
    except Exception as e:
        print("\n" + "!"*60)
        print("! UNEXPECTED ERROR")
        print("!"*60)
        print(f"\nError: {e}")
        print("\n" + "="*60)
        return False


if __name__ == "__main__":
    import sys
    success = run_all_benchmarks()
    sys.exit(0 if success else 1)
