# -*- coding: utf-8 -*-
"""
Unit tests for Test-Time Adaptation implementation.
Tests that update_step() returns dict with 'loss', 'steps', etc.
Now includes parameter mutation checks to verify actual weight updates.
"""
import pytest
import torch
import torch.nn as nn
import copy

from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from core.ssm import StateSpaceModel

class TestAdapter:
    """Test suite for Adapter implementation."""
    
    @pytest.fixture
    def model(self):
        """Create a simple test model."""
        return StateSpaceModel(
            state_dim=4,
            input_dim=4,
            output_dim=1
        )
    
    @pytest.fixture
    def config(self):
        """Create AdaptationConfig."""
        return AdaptationConfig(
            learning_rate=0.01,
            num_steps=5
        )
    
    def test_adapter_initialization(self, model, config):
        """Test that Adapter initializes correctly."""
        adapter = Adapter(model=model, config=config)
        assert adapter.model is model
        assert adapter.config is config
        print("✓ Adapter initialization successful")
    
    def test_update_step_return_type(self, model, config):
        """Test that update_step() returns correct type (dict or tuple)."""
        adapter = Adapter(model=model, config=config)
        
        # Create test data
        batch_size = 8
        x = torch.randn(batch_size, 4)
        y = torch.randn(batch_size, 1)
        hidden_state = model.init_hidden(batch_size=batch_size)
        
        # Call update_step
        result = adapter.update_step(x=x, y=y, hidden_state=hidden_state)
        
        # Validate return type
        assert isinstance(result, (dict, tuple)), (
            f"Expected dict or tuple, got {type(result)}"
        )
        
        # If dict, check for required keys
        if isinstance(result, dict):
            assert 'loss' in result, f"Expected 'loss' key in result dict"
            print(f"✓ update_step() returned dict with keys: {result.keys()}")
        else:  # tuple
            assert len(result) >= 1, "Expected at least loss value in tuple"
            print(f"✓ update_step() returned tuple with {len(result)} elements")
    
    def test_parameter_mutation(self, model, config):
        """
        Test that update_step() actually mutates model parameters.
        Uses deepcopy before/after and torch.equal to verify weight updates.
        """
        adapter = Adapter(model=model, config=config)
        
        # Create test data
        batch_size = 8
        x = torch.randn(batch_size, 4)
        y = torch.randn(batch_size, 1)
        hidden_state = model.init_hidden(batch_size=batch_size)
        
        # Deep copy model parameters BEFORE update
        params_before = {}
        for name, param in model.named_parameters():
            params_before[name] = copy.deepcopy(param.data)
        
        print("\nParameter values BEFORE update_step:")
        for name, param in list(params_before.items())[:3]:  # Show first 3 params
            print(f"  {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
        
        # Perform update step
        result = adapter.update_step(x=x, y=y, hidden_state=hidden_state)
        
        # Deep copy model parameters AFTER update
        params_after = {}
        for name, param in model.named_parameters():
            params_after[name] = copy.deepcopy(param.data)
        
        print("\nParameter values AFTER update_step:")
        for name, param in list(params_after.items())[:3]:  # Show first 3 params
            print(f"  {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
        
        # Verify that at least some parameters have changed
        params_changed = 0
        params_unchanged = 0
        
        for name in params_before.keys():
            param_before = params_before[name]
            param_after = params_after[name]
            
            # Check if parameters are different using torch.equal
            if torch.equal(param_before, param_after):
                params_unchanged += 1
                print(f"  WARNING: Parameter '{name}' unchanged after update")
            else:
                params_changed += 1
                # Calculate difference magnitude
                diff = (param_after - param_before).abs().mean().item()
                print(f"  ✓ Parameter '{name}' changed (avg diff: {diff:.6f})")
        
        print(f"\nSummary: {params_changed} parameters changed, {params_unchanged} unchanged")
        
        # Assert that at least one parameter has changed
        assert params_changed > 0, (
            f"Expected at least one parameter to change after update_step, "
            f"but all {params_unchanged} parameters remained unchanged. "
            f"This indicates that the adapter is not actually updating model weights."
        )
        
        print(f"✓ Parameter mutation test passed: {params_changed} parameters updated")
    
    def test_multiple_update_steps(self, model, config):
        """Test multiple consecutive update steps."""
        adapter = Adapter(model=model, config=config)
        
        # Create test data
        batch_size = 8
        x = torch.randn(batch_size, 4)
        y = torch.randn(batch_size, 1)
        hidden_state = model.init_hidden(batch_size=batch_size)
        
        # Perform multiple update steps
        losses = []
        for i in range(3):
            result = adapter.update_step(x=x, y=y, hidden_state=hidden_state)
            
            # Extract loss value
            if isinstance(result, dict):
                loss_val = result['loss']
            else:  # tuple
                loss_val = result[0]
            
            losses.append(loss_val)
            print(f"  Step {i}: loss = {loss_val:.4f}")
        
        print(f"✓ Multiple update steps completed: losses = {[f'{l:.4f}' for l in losses]}")
    
    def test_adaptation_with_hidden_state(self, model, config):
        """Test adaptation with proper hidden state management."""
        adapter = Adapter(model=model, config=config)
        
        # Create test data
        batch_size = 4
        x = torch.randn(batch_size, 4)
        y = torch.randn(batch_size, 1)
        
        # Initialize hidden state
        hidden_state = model.init_hidden(batch_size=batch_size)
        
        # Define forward function that manages hidden state
        def fwd_fn(x_input, h_state):
            """Forward function that takes input and hidden state, returns output and next hidden state."""
            output, next_h = model(x_input, h_state)
            return output, next_h
        
        # Perform adaptation with hidden state management
        output, hidden_state = fwd_fn(x, hidden_state)
        result = adapter.update_step(x=x, y=y, hidden_state=hidden_state)
        
        # Validate
        assert hidden_state is not None, "Hidden state should be maintained"
        assert output is not None, "Output should be produced"
        
        print("✓ Adaptation with hidden state management successful")

def test_full_adaptation_pipeline():
    """Integration test for full adaptation pipeline."""
    print("\n" + "="*60)
    print("Integration Test: Full Adaptation Pipeline")
    print("="*60)
    
    # Setup
    model = StateSpaceModel(state_dim=4, input_dim=4, output_dim=1)
    config = AdaptationConfig(learning_rate=0.01, num_steps=10)
    adapter = Adapter(model=model, config=config)
    
    # Create test data
    batch_size = 8
    x = torch.randn(batch_size, 4)
    y = torch.randn(batch_size, 1)
    hidden_state = model.init_hidden(batch_size=batch_size)
    
    # Deep copy parameters before adaptation
    params_before = {}
    for name, param in model.named_parameters():
        params_before[name] = copy.deepcopy(param.data)
    
    # Run adaptation loop
    print("\nRunning adaptation loop...")
    for step in range(5):
        # Forward pass with hidden state management
        output, hidden_state = model(x, hidden_state)
        
        # CRITICAL FIX: Detach hidden_state to prevent "backward through the graph a second time" error
        # This breaks the connection to the previous computational graph
        hidden_state = hidden_state.detach()
        
        # Adaptation step
        result = adapter.update_step(x=x, y=y, hidden_state=hidden_state)
        
        # Extract loss
        if isinstance(result, dict):
            loss_val = result['loss']
        else:
            loss_val = result[0]
        
        print(f"  Step {step}: loss = {loss_val:.4f}")
    
    # Verify parameters changed
    params_changed = 0
    for name in params_before.keys():
        if not torch.equal(params_before[name], model.state_dict()[name]):
            params_changed += 1
    
    assert params_changed > 0, "Expected parameters to change during adaptation"
    print(f"\n✓ Full adaptation pipeline successful: {params_changed} parameters updated")
    print("="*60)

if __name__ == "__main__":
    # Run tests manually (for development)
    print("Running adaptation tests...\n")
    
    test_adapter = TestAdapter()
    model = StateSpaceModel(state_dim=4, input_dim=4, output_dim=1)
    config = AdaptationConfig(learning_rate=0.01, num_steps=5)
    
    print("Test 1: Adapter Initialization")
    test_adapter.test_adapter_initialization(model, config)
    
    print("\nTest 2: Update Step Return Type")
    test_adapter.test_update_step_return_type(model, config)
    
    print("\nTest 3: Parameter Mutation Check")
    test_adapter.test_parameter_mutation(model, config)
    
    print("\nTest 4: Multiple Update Steps")
    test_adapter.test_multiple_update_steps(model, config)
    
    print("\nTest 5: Adaptation with Hidden State")
    test_adapter.test_adaptation_with_hidden_state(model, config)
    
    print("\nTest 6: Full Adaptation Pipeline")
    test_full_adaptation_pipeline()
    
    print("\n" + "="*60)
    print("All tests passed successfully!")
    print("="*60)
