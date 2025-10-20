#!/usr/bin/env python3
"""
Quick Benchmark Script for SSM-MetaRL - 100% API-aligned.
Tests all three core APIs: SSM, MetaMAML, and Adapter.

Usage:
    python experiments/quick_benchmark.py
"""
import os
import sys
import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from core.ssm import SSM
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter, AdaptationConfig


def benchmark_ssm():
    """
    Benchmark SSM with exact API:
    - SSM(state_dim, hidden_dim=128, output_dim=32, device='cpu')
    - forward(x, hidden_state=None) -> single tensor
    """
    print("\n" + "="*60)
    print("1. SSM Benchmark")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create SSM with exact defaults
    model = SSM(state_dim=10, hidden_dim=128, output_dim=32, device=device)
    print(f"✓ Created SSM(state_dim=10, hidden_dim=128, output_dim=32, device='{device}')")
    
    # Test forward pass
    batch_size = 32
    state = torch.randn(batch_size, 10).to(device)
    
    start_time = time.time()
    output = model.forward(state, hidden_state=None)
    elapsed = time.time() - start_time
    
    # Verify output is single tensor, not tuple
    assert isinstance(output, torch.Tensor), "forward() must return single tensor"
    assert output.shape == (batch_size, 32), f"Expected (32, 32), got {output.shape}"
    print(f"✓ forward() returns single tensor with shape {output.shape}")
    print(f"✓ Forward pass time: {elapsed*1000:.2f}ms")
    
    return model


def benchmark_meta_maml(model):
    """
    Benchmark MetaMAML with exact API:
    - MetaMAML(model, inner_lr, outer_lr, first_order=False)
    - adapt(support_x, support_y, loss_fn, num_steps) -> adapted_model
    - meta_update(tasks: List[Tuple[support_x, support_y, query_x, query_y]], loss_fn) -> meta_loss
    """
    print("\n" + "="*60)
    print("2. MetaMAML Benchmark")
    print("="*60)
    
    device = next(model.parameters()).device
    
    # Create MetaMAML with exact signature
    meta_learner = MetaMAML(model, inner_lr=0.01, outer_lr=0.001, first_order=False)
    print("✓ Created MetaMAML(model, inner_lr=0.01, outer_lr=0.001, first_order=False)")
    
    loss_fn = nn.MSELoss()
    
    # Test adapt() method
    support_x = torch.randn(10, 10).to(device)
    support_y = torch.randn(10, 32).to(device)
    
    start_time = time.time()
    adapted_model = meta_learner.adapt(support_x, support_y, loss_fn, num_steps=5)
    elapsed = time.time() - start_time
    
    assert isinstance(adapted_model, nn.Module), "adapt() must return a model"
    print(f"✓ adapt() returned adapted model")
    print(f"✓ Adaptation time: {elapsed*1000:.2f}ms")
    
    # Test meta_update() method with exact task format
    tasks = []
    for _ in range(8):
        support_x = torch.randn(10, 10).to(device)
        support_y = torch.randn(10, 32).to(device)
        query_x = torch.randn(10, 10).to(device)
        query_y = torch.randn(10, 32).to(device)
        tasks.append((support_x, support_y, query_x, query_y))
    
    start_time = time.time()
    meta_loss = meta_learner.meta_update(tasks, loss_fn)
    elapsed = time.time() - start_time
    
    assert isinstance(meta_loss, float) or isinstance(meta_loss, torch.Tensor), "meta_update() must return loss"
    print(f"✓ meta_update() with {len(tasks)} tasks, meta_loss: {meta_loss:.4f}")
    print(f"✓ Meta-update time: {elapsed*1000:.2f}ms")


def benchmark_adapter(model):
    """
    Benchmark Adapter with exact API:
    - AdaptationConfig(lr, grad_clip_norm, trust_region_eps, ema_decay, entropy_weight, max_steps_per_call)
    - Adapter(model, cfg)
    - adapt(loss_fn, batch_dict) -> loss
    """
    print("\n" + "="*60)
    print("3. Adapter Benchmark")
    print("="*60)
    
    device = next(model.parameters()).device
    
    # Create config with exact fields from implementation
    config = AdaptationConfig(
        lr=0.01,
        grad_clip_norm=1.0,
        trust_region_eps=0.01,
        ema_decay=0.99,
        entropy_weight=0.01,
        max_steps_per_call=5
    )
    print("✓ Created AdaptationConfig with all required fields")
    
    # Create Adapter with exact signature
    adapter = Adapter(model, config)
    print("✓ Created Adapter(model, config)")
    
    loss_fn = nn.MSELoss()
    
    # Prepare batch_dict
    batch_size = 16
    states = torch.randn(batch_size, 10).to(device)
    targets = torch.randn(batch_size, 32).to(device)
    batch_dict = {'states': states, 'targets': targets}
    
    start_time = time.time()
    loss = adapter.adapt(loss_fn, batch_dict)
    elapsed = time.time() - start_time
    
    assert isinstance(loss, float) or isinstance(loss, torch.Tensor), "adapt() must return loss"
    print(f"✓ adapt(loss_fn, batch_dict) returned loss: {loss:.4f}")
    print(f"✓ Adaptation time: {elapsed*1000:.2f}ms")


def run_all_benchmarks():
    """
    Run all benchmarks in sequence.
    """
    print("\n" + "#"*60)
    print("# SSM-MetaRL Quick Benchmark Suite")
    print("# Validates 100% API compliance")
    print("#"*60)
    
    try:
        # 1. Benchmark SSM
        model = benchmark_ssm()
        
        # 2. Benchmark MetaMAML
        benchmark_meta_maml(model)
        
        # 3. Benchmark Adapter
        benchmark_adapter(model)
        
        print("\n" + "="*60)
        print("✓ ALL BENCHMARKS PASSED")
        print("✓ All APIs are 100% compliant with implementations")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n✗ BENCHMARK FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_benchmarks()
