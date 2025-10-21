# -*- coding: utf-8 -*-
"""
Unit tests for MetaMAML implementation.
Tests that adapt_task() returns OrderedDict (fast_weights).
"""

import pytest
import torch
import torch.nn as nn
from collections import OrderedDict
from meta_rl.meta_maml import MetaMAML
from core.ssm import StateSpaceModel


class TestMetaMAML:
    """Test suite for MetaMAML implementation."""
    
    @pytest.fixture
    def model(self):
        """Create a simple test model."""
        return StateSpaceModel(
            state_dim=4,
            input_dim=4,
            output_dim=1
        )
    
    @pytest.fixture
    def maml(self, model):
        """Create MetaMAML instance."""
        return MetaMAML(
            model=model,
            inner_lr=0.01,
            outer_lr=0.001
        )
    
    def test_adapt_returns_ordered_dict(self, maml):
        """
        Test that adapt_task() returns OrderedDict (fast_weights).
        This is the core requirement for consistency.
        """
        # Create task data
        B, T, D_in = 8, 10, 4
        D_out = 1
        
        # Reshape data and create targets
        support_x = torch.randn(B, T, D_in).reshape(B * T, D_in)
        support_y = torch.randn(B * T, D_out)
        
        # Call adapt_task using support_y and 'num_steps'
        result = maml.adapt_task(support_x, support_y, num_steps=5)
        
        # Verify return type is OrderedDict
        assert isinstance(result, OrderedDict), \
            f"Expected OrderedDict from adapt_task(), got {type(result)}"
    
    def test_adapt_returns_weights(self, maml):
        """
        Test that adapt_task() returns a dictionary containing model parameters.
        """
        B, T, D_in = 8, 10, 4
        D_out = 1
        
        # Reshape data and create targets
        support_x = torch.randn(B, T, D_in).reshape(B * T, D_in)
        support_y = torch.randn(B * T, D_out)
        
        # using support_y and 'num_steps'
        fast_weights = maml.adapt_task(support_x, support_y, num_steps=3)
        
        # Verify it's an OrderedDict
        assert isinstance(fast_weights, OrderedDict)
        
        # Verify it contains tensor values (weights)
        assert len(fast_weights) > 0, "fast_weights should not be empty"
        
        for key, value in fast_weights.items():
            assert isinstance(value, torch.Tensor), \
                f"Expected tensor for key {key}, got {type(value)}"
    
    def test_adapt_with_multiple_steps(self, maml):
        """
        Test adapt_task() with different numbers of adaptation steps.
        """
        B, T, D_in = 8, 10, 4
        D_out = 1
        
        # Reshape data and create targets
        support_x = torch.randn(B, T, D_in).reshape(B * T, D_in)
        support_y = torch.randn(B * T, D_out)
        
        for n_steps in [1, 3, 5, 10]:
            # using support_y and 'num_steps'
            fast_weights = maml.adapt_task(support_x, support_y, num_steps=n_steps)
            
            # Always returns OrderedDict regardless of n_steps
            assert isinstance(fast_weights, OrderedDict), \
                f"Expected OrderedDict for n_steps={n_steps}, got {type(fast_weights)}"
    
    def test_adapt_usage_pattern(self, maml):
        """
        Test the correct usage pattern for adapt_task() result.
        Users should receive OrderedDict and use it with forward_with_weights.
        """
        B, T, D_in = 8, 10, 4
        D_out = 1
        
        # Reshape data and create targets
        support_x = torch.randn(B, T, D_in).reshape(B * T, D_in)
        support_y = torch.randn(B * T, D_out)
        
        test_x = torch.randn(4, 10, 4).reshape(4 * 10, 4)
        
        # Get fast_weights from adapt_task
        # using support_y and 'num_steps'
        fast_weights = maml.adapt_task(support_x, support_y, num_steps=5)
        
        # Verify type
        assert isinstance(fast_weights, OrderedDict)
        
        # Typical usage: pass fast_weights to functional_forward
        output = maml.functional_forward(test_x, fast_weights)
        assert output.shape == (test_x.shape[0], D_out)
        
        assert hasattr(fast_weights, 'keys'), "Should have dict-like interface"
        assert hasattr(fast_weights, 'items'), "Should have dict-like interface"


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
