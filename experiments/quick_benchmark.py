# -*- coding: utf-8 -*-
"""
Quick benchmark script for SSM-MetaRL-TestCompute.
Benchmarks both MetaMAML and Test-Time Adaptation.
Validates:
- MetaMAML.adapt_task() returns OrderedDict (fast_weights)
- Adapter.update_step() returns dict with 'loss' key
Now passes time series input (B, T, D) without flattening to MAML.
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
    Validates that adapt_task() returns OrderedDict.
    Now passes time series data (B, T, D) without flattening.
    """
    print("\n" + "="*60)
    print("BENCHMARK: MetaMAML")
    print("="*60)
    
    # Setup
    D_in, D_out = 4, 1
    model = StateSpaceModel(state_dim=4, input_dim=D_in, output_dim=D_out)
    maml = MetaMAML(model=model, inner_lr=0.01, outer_lr=0.001)
    
    # Create task data - keep as time series (B, T, D)
    B, T = 8, 10
    
    # Create time series data without flattening
    support_x = torch.randn(B, T, D_in)  # (B, T, D_in) - time series format
    support_y = torch.randn(B, T, D_out)  # (B, T, D_out) - time series format
    query_x = torch.randn(B, T, D_in)  # (B, T, D_in) - time series format
    query_y = torch.randn(B, T, D_out)  # (B, T, D_out) - time series format
    
    # Initialize hidden state for batch B (not B*T)
    initial_hidden = model.init_hidden(batch_size=B)
    
    print(f"Support X shape: {support_x.shape}")
    print(f"Support Y shape: {support_y.shape}")
    print(f"Hidden state shape: {initial_hidden.shape}")
    
    # Adapt to task - now receives time series data
    print("\nAdapting to task...")
    fast_weights = maml.adapt_task(
        x_support=support_x,
        y_support=support_y,
        hidden_state=initial_hidden
    )
    
    # Validate output type
    assert isinstance(fast_weights, OrderedDict), f"Expected OrderedDict, got {type(fast_weights)}"
    print(f"✓ adapt_task() returned OrderedDict with {len(fast_weights)} parameters")
    
    # Test meta_update - now receives time series data
    print("\nPerforming meta-update...")
    loss = maml.meta_update(
        x_support=support_x,
        y_support=support_y,
        x_query=query_x,
        y_query=query_y
    )
    print(f"✓ Meta loss: {loss:.4f}")
    
    print("\n" + "="*60)
    print("MetaMAML benchmark completed successfully!")
    print("="*60)

def benchmark_test_time_adaptation():
    """
    Benchmark Test-Time Adaptation.
    Validates that Adapter.update_step() returns dict with 'loss' key.
    Now manages hidden_state through fwd_fn at each step.
    """
    print("\n" + "="*60)
    print("BENCHMARK: Test-Time Adaptation")
    print("="*60)
    
    # Setup
    D_in, D_out = 4, 1
    model = StateSpaceModel(state_dim=4, input_dim=D_in, output_dim=D_out)
    config = AdaptationConfig(learning_rate=0.01, num_steps=10)
    adapter = Adapter(model=model, config=config)
    
    # Create test data
    batch_size = 8
    x = torch.randn(batch_size, D_in)
    y = torch.randn(batch_size, D_out)
    
    # Initialize hidden state
    hidden_state = model.init_hidden(batch_size=batch_size)
    
    # Define forward function that manages hidden state
    def fwd_fn(x_input, h_state):
        """Forward function that takes input and hidden state, returns output and next hidden state."""
        output, next_h = model(x_input, h_state)
        return output, next_h
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Hidden state shape: {hidden_state.shape}")
    
    # Perform adaptation steps
    print("\nPerforming adaptation steps...")
    for step in range(5):
        # Use fwd_fn to get output and update hidden state
        output, hidden_state = fwd_fn(x, hidden_state)
        
        # Perform update step - now with hidden state management
        result = adapter.update_step(
            x=x,
            y=y,
            hidden_state=hidden_state
        )
        
        # Validate output type
        assert isinstance(result, (dict, tuple)), f"Expected dict or tuple, got {type(result)}"
        
        # Handle both dict and tuple returns
        if isinstance(result, dict):
            assert 'loss' in result, f"Expected 'loss' key in result dict, got keys: {result.keys()}"
            loss_val = result['loss']
            print(f"  Step {step}: loss = {loss_val:.4f}")
        else:  # tuple
            loss_val, steps_taken = result
            print(f"  Step {step}: loss = {loss_val:.4f}, steps_taken = {steps_taken}")
    
    print("\n✓ All adaptation steps completed successfully!")
    
    print("\n" + "="*60)
    print("Test-Time Adaptation benchmark completed successfully!")
    print("="*60)

def main():
    """
    Run all benchmarks.
    """
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + " "*10 + "SSM-MetaRL-TestCompute Quick Benchmark" + " "*10 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    try:
        benchmark_meta_maml()
        benchmark_test_time_adaptation()
        
        print("\n" + "#"*60)
        print("#" + " "*58 + "#")
        print("#" + " "*15 + "ALL BENCHMARKS PASSED!" + " "*22 + "#")
        print("#" + " "*58 + "#")
        print("#"*60 + "\n")
        
    except Exception as e:
        print(f"\n\n{'='*60}")
        print(f"BENCHMARK FAILED: {e}")
        print(f"{'='*60}\n")
        raise

if __name__ == "__main__":
    main()
