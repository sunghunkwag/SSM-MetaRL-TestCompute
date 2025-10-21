# -*- coding: utf-8 -*-
"""
Unit tests for Test-Time Adaptation implementation.
Tests that adapt() returns dict with 'loss', 'steps', etc.
"""

import pytest
import torch
import torch.nn as nn
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
            lr=0.01,
            grad_clip_norm=1.0,
            trust_region_eps=0.01,
            ema_decay=0.99,
            entropy_weight=0.01,
            max_steps_per_call=5
        )
    
    @pytest.fixture
    def adapter(self, model, config):
        """Create Adapter instance."""
        return Adapter(model, config)

    # [FIX] Helper function for correct fwd/loss calls
    def _run_adapt(self, adapter, loss_fn, batch_dict):
        def fwd_fn(batch):
            # Model forward expects 'x'
            return adapter.target(batch['x'])
            
        def loss_fn_wrapper(outputs, batch):
            # Loss fn compares against 'targets'
            return loss_fn(outputs, batch['targets'])
        
        return adapter.adapt(loss_fn_wrapper, batch_dict, fwd_fn=fwd_fn)

    def test_adapt_returns_dict(self, adapter):
        """
        Test that adapt() returns dict (info).
        This is the core requirement for consistency.
        """
        loss_fn = nn.MSELoss()
        states = torch.randn(8, 4)
        targets = torch.randn(8, 1)
        # [FIX] Model forward expects 'x', not 'states'
        batch_dict = {'x': states, 'targets': targets}
        
        # Call adapt [FIX] using wrapper
        result = self._run_adapt(adapter, loss_fn, batch_dict)
        
        # Verify return type is dict
        assert isinstance(result, dict), \
            f"Expected dict from adapt(), got {type(result)}"
    
    def test_adapt_contains_loss_key(self, adapter):
        """
        Test that adapt() result contains 'loss' key.
        """
        loss_fn = nn.MSELoss()
        states = torch.randn(8, 4)
        targets = torch.randn(8, 1)
        # [FIX] Model forward expects 'x'
        batch_dict = {'x': states, 'targets': targets}
        
        # [FIX] using wrapper
        info = self._run_adapt(adapter, loss_fn, batch_dict)
        
        # Verify it's a dict
        assert isinstance(info, dict)
        
        # Verify 'loss' key exists
        assert 'loss' in info, \
            f"Expected 'loss' key in result, got keys: {info.keys()}"
        
        # Verify loss is a numeric value
        loss = info['loss']
        assert isinstance(loss, (float, int, torch.Tensor)), \
            f"Expected numeric loss value, got {type(loss)}"
    
    def test_adapt_contains_steps_key(self, adapter):
        """
        Test that adapt() result contains 'steps' key.
        """
        loss_fn = nn.MSELoss()
        states = torch.randn(8, 4)
        targets = torch.randn(8, 1)
        # [FIX] Model forward expects 'x'
        batch_dict = {'x': states, 'targets': targets}
        
        # [FIX] using wrapper
        info = self._run_adapt(adapter, loss_fn, batch_dict)
        
        # Verify 'steps' key exists
        assert 'steps' in info, \
            f"Expected 'steps' key in result, got keys: {info.keys()}"
        
        # Verify steps is an integer
        steps = info['steps']
        assert isinstance(steps, int), \
            f"Expected int for steps, got {type(steps)}"
    
    def test_adapt_usage_pattern(self, adapter):
        """
        Test the correct usage pattern for adapt() result.
        Users should extract loss from the returned dict.
        """
        loss_fn = nn.MSELoss()
        states = torch.randn(8, 4)
        targets = torch.randn(8, 1)
        # [FIX] Model forward expects 'x'
        batch_dict = {'x': states, 'targets': targets}
        
        # Get info dict from adapt [FIX] using wrapper
        info = self._run_adapt(adapter, loss_fn, batch_dict)
        
        # Verify type
        assert isinstance(info, dict)
        
        # Extract loss and steps (typical usage pattern)
        loss = info['loss']
        steps = info['steps']
        
        # Verify extracted values are valid
        assert isinstance(loss, (float, int, torch.Tensor))
        assert isinstance(steps, int)
        assert steps >= 0, "Steps should be non-negative"
    
    def test_adapt_multiple_calls(self, adapter):
        """
        Test that multiple adapt() calls consistently return dict.
        """
        loss_fn = nn.MSELoss()
        
        for i in range(5):
            states = torch.randn(8, 4)
            targets = torch.randn(8, 1)
            # [FIX] Model forward expects 'x'
            batch_dict = {'x': states, 'targets': targets}
            
            # [FIX] using wrapper
            info = self._run_adapt(adapter, loss_fn, batch_dict)
            
            # Always returns dict
            assert isinstance(info, dict), \
                f"Expected dict on call {i}, got {type(info)}"
            
            # Always has 'loss' key
            assert 'loss' in info, \
                f"Expected 'loss' key on call {i}, got keys: {info.keys()}"


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
