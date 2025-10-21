# -*- coding: utf-8 -*-
"""
Quick benchmark script for SSM-MetaRL-TestCompute.
Benchmarks both MetaMAML and Test-Time Adaptation.
FIXED: Calls MetaMAML API correctly matching main.py and meta_maml.py definition.
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter, AdaptationConfig

def benchmark_meta_maml():
    """
    Benchmark MetaMAML adaptation.
    FIXED: Calls adapt_task and meta_update with correct arguments
           (positional args for adapt_task, tasks list and state for meta_update).
    """
    print("\n" + "="*60)
    print("BENCHMARK: MetaMAML")
    print("="*60)

    # Setup
    D_in, D_out = 4, 1 # Example dimension, ensure model output matches target
    # Output dim needs to match target 'y', which might be next_state (D_in) or reward (1)
    # Let's assume the target is next state for consistency with main.py
    model = StateSpaceModel(state_dim=4, input_dim=D_in, output_dim=D_in) # Output dim matches input for next_obs prediction
    maml = MetaMAML(model=model, inner_lr=0.01, outer_lr=0.001)

    # Create task data - keep as time series (B, T, D)
    B, T = 8, 10 # Batch size B, Time steps T

    # Create time series data
    support_x = torch.randn(B, T, D_in)  # (B, T, D_in)
    support_y = torch.randn(B, T, D_in)  # Target is next observation (B, T, D_in)
    query_x = torch.randn(B, T, D_in)    # (B, T, D_in)
    query_y = torch.randn(B, T, D_in)    # Target is next observation (B, T, D_in)

    # Initialize hidden state for batch B
    initial_hidden = model.init_hidden(batch_size=B)

    print(f"Support X shape: {support_x.shape}")
    print(f"Support Y shape: {support_y.shape}")
    print(f"Hidden state shape: {initial_hidden.shape}")

    # --- FIX: Call adapt_task correctly ---
    # Use positional arguments: support_x, support_y, initial_hidden_state
    print("\nAdapting to task...")
    fast_weights = maml.adapt_task(
        support_x,
        support_y,
        initial_hidden_state=initial_hidden,
        loss_fn=F.mse_loss, # Example loss
        num_steps=3 # Example inner steps
    )

    # Validate output type
    assert isinstance(fast_weights, OrderedDict), f"Expected OrderedDict, got {type(fast_weights)}"
    print(f"✓ adapt_task() returned OrderedDict with {len(fast_weights)} parameters")

    # --- FIX: Call meta_update correctly ---
    # Prepare tasks as List[Tuple]
    # For benchmark, let's treat each item in batch B as a separate task
    tasks = []
    for i in range(B):
        tasks.append((
            support_x[i:i+1], # Keep batch dim: (1, T, D_in)
            support_y[i:i+1], # (1, T, D_in)
            query_x[i:i+1],   # (1, T, D_in)
            query_y[i:i+1]    # (1, T, D_in)
        ))

    # Pass tasks list and initial_hidden_state
    # Note: Using the same initial_hidden for all tasks in the batch here
    # A more rigorous benchmark might require per-task initial states if tasks differ significantly
    print("\nPerforming meta-update...")
    loss = maml.meta_update(
        tasks,
        initial_hidden_state=initial_hidden, # Pass initial state
        loss_fn=F.mse_loss # Example loss
    )
    print(f"✓ Meta loss: {loss:.4f}")

    print("\n" + "="*60)
    print("MetaMAML benchmark completed successfully!")
    print("="*60)


def benchmark_test_time_adaptation():
    """
    Benchmark Test-Time Adaptation.
    Validates that Adapter.update_step() returns tuple (loss, steps).
    Uses correct API call.
    """
    print("\n" + "="*60)
    print("BENCHMARK: Test-Time Adaptation")
    print("="*60)

    # Setup
    D_in = 4
    # Output dim must match target 'y', assume next observation prediction
    D_out = D_in
    model = StateSpaceModel(state_dim=4, input_dim=D_in, output_dim=D_out)
    # Use num_steps from adapter config for internal loop count
    config = AdaptationConfig(learning_rate=0.01, num_steps=5)
    adapter = Adapter(model=model, config=config)

    # Create test data (batch for adaptation)
    batch_size = 8
    x = torch.randn(batch_size, D_in)
    y = torch.randn(batch_size, D_out) # Target (e.g., next observation)

    # Initialize hidden state
    hidden_state = model.init_hidden(batch_size=batch_size) # state_t

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Hidden state shape: {hidden_state.shape}")

    # Perform adaptation steps
    print("\nPerforming adaptation steps...")
    total_adaptation_steps = 10 # Number of times we call update_step
    for step in range(total_adaptation_steps):
        # Store current state for adaptation
        current_hidden_state_for_adapt = hidden_state

        # Get model output and next state (not used for action here)
        with torch.no_grad():
             output, hidden_state = model(x, current_hidden_state_for_adapt)

        # Perform one call to update_step, which runs config.num_steps internally
        # Pass state_t (current_hidden_state_for_adapt)
        result = adapter.update_step(
            x=x,
            y=y, # Target
            hidden_state=current_hidden_state_for_adapt # state_t
        )

        # Validate output type (should be tuple)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected tuple of length 2 (loss, steps), got {len(result)}"

        loss_val, steps_taken = result
        print(f"  Adaptation Call {step}: loss = {loss_val:.4f}, internal_steps = {steps_taken}")

        # In a real scenario, you'd get new x, y, and potentially reset hidden_state
        # For benchmark, we reuse x, y and use the updated hidden_state for next call's prediction step
        # (This benchmark doesn't simulate environment interaction, just the adapter call)


    print("\n✓ All adaptation calls completed successfully!")

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
        import traceback
        traceback.print_exc() # Print traceback for debugging
        raise

if __name__ == "__main__":
    # Import F for loss function
    import torch.nn.functional as F
    main()
