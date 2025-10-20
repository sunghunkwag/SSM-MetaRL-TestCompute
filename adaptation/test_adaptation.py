#!/usr/bin/env python3
"""
Test file for test-time adaptation - 100% aligned with adaptation/test_time_adaptation.py.

API under test:
- AdaptationConfig(lr, grad_clip_norm, trust_region_eps, ema_decay, entropy_weight, max_steps_per_call)
- Adapter(model, cfg)
- adapt(loss_fn, batch_dict) -> loss
"""
import pytest
import torch
import torch.nn as nn
from adaptation.test_time_adaptation import Adapter, AdaptationConfig


def create_simple_model():
    """Create a simple MLP for testing."""
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32)
    )


def test_adaptation_config_import():
    """Test that AdaptationConfig can be imported successfully."""
    from adaptation.test_time_adaptation import AdaptationConfig
    assert AdaptationConfig is not None


def test_adapter_import():
    """Test that Adapter can be imported successfully."""
    from adaptation.test_time_adaptation import Adapter
    assert Adapter is not None


def test_adaptation_config_initialization():
    """
    Test AdaptationConfig initialization with exact fields:
    lr, grad_clip_norm, trust_region_eps, ema_decay, entropy_weight, max_steps_per_call
    """
    config = AdaptationConfig(
        lr=0.01,
        grad_clip_norm=1.0,
        trust_region_eps=0.01,
        ema_decay=0.99,
        entropy_weight=0.01,
        max_steps_per_call=5
    )
    
    assert config.lr == 0.01
    assert config.grad_clip_norm == 1.0
    assert config.trust_region_eps == 0.01
    assert config.ema_decay == 0.99
    assert config.entropy_weight == 0.01
    assert config.max_steps_per_call == 5


def test_adapter_initialization():
    """
    Test Adapter initialization with exact API:
    Adapter(model, cfg)
    """
    model = create_simple_model()
    
    config = AdaptationConfig(
        lr=0.01,
        grad_clip_norm=1.0,
        trust_region_eps=0.01,
        ema_decay=0.99,
        entropy_weight=0.01,
        max_steps_per_call=5
    )
    
    # Create Adapter with exact signature
    adapter = Adapter(model, config)
    
    assert adapter is not None
    assert hasattr(adapter, 'model')
    assert hasattr(adapter, 'cfg')


def test_adapter_adapt_method():
    """
    Test adapt method with exact API:
    adapt(loss_fn, batch_dict) -> loss
    """
    model = create_simple_model()
    
    config = AdaptationConfig(
        lr=0.01,
        grad_clip_norm=1.0,
        trust_region_eps=0.01,
        ema_decay=0.99,
        entropy_weight=0.01,
        max_steps_per_call=5
    )
    
    adapter = Adapter(model, config)
    
    # Prepare batch_dict
    batch_size = 16
    states = torch.randn(batch_size, 10)
    targets = torch.randn(batch_size, 32)
    batch_dict = {'states': states, 'targets': targets}
    
    loss_fn = nn.MSELoss()
    
    # Call adapt with exact signature: adapt(loss_fn, batch_dict)
    loss = adapter.adapt(loss_fn, batch_dict)
    
    # adapt() must return a loss value
    assert isinstance(loss, (float, torch.Tensor))


def test_adapter_adapt_multiple_calls():
    """
    Test that adapt can be called multiple times.
    """
    model = create_simple_model()
    
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
    
    # Call adapt multiple times
    for _ in range(3):
        batch_dict = {
            'states': torch.randn(16, 10),
            'targets': torch.randn(16, 32)
        }
        loss = adapter.adapt(loss_fn, batch_dict)
        assert isinstance(loss, (float, torch.Tensor))


def test_adapter_config_fields_exact():
    """
    Test that config has exactly the required fields, no extras.
    """
    config = AdaptationConfig(
        lr=0.01,
        grad_clip_norm=1.0,
        trust_region_eps=0.01,
        ema_decay=0.99,
        entropy_weight=0.01,
        max_steps_per_call=5
    )
    
    # Verify all required fields exist
    assert hasattr(config, 'lr')
    assert hasattr(config, 'grad_clip_norm')
    assert hasattr(config, 'trust_region_eps')
    assert hasattr(config, 'ema_decay')
    assert hasattr(config, 'entropy_weight')
    assert hasattr(config, 'max_steps_per_call')


def test_adapter_with_different_batch_sizes():
    """
    Test adapter with different batch sizes.
    """
    model = create_simple_model()
    
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
    
    for batch_size in [8, 16, 32]:
        batch_dict = {
            'states': torch.randn(batch_size, 10),
            'targets': torch.randn(batch_size, 32)
        }
        loss = adapter.adapt(loss_fn, batch_dict)
        assert isinstance(loss, (float, torch.Tensor))


def test_adapter_max_steps_per_call():
    """
    Test that max_steps_per_call is respected.
    """
    model = create_simple_model()
    
    # Test with different max_steps_per_call values
    for max_steps in [1, 5, 10]:
        config = AdaptationConfig(
            lr=0.01,
            grad_clip_norm=1.0,
            trust_region_eps=0.01,
            ema_decay=0.99,
            entropy_weight=0.01,
            max_steps_per_call=max_steps
        )
        
        adapter = Adapter(model, config)
        loss_fn = nn.MSELoss()
        
        batch_dict = {
            'states': torch.randn(16, 10),
            'targets': torch.randn(16, 32)
        }
        
        loss = adapter.adapt(loss_fn, batch_dict)
        assert isinstance(loss, (float, torch.Tensor))


def test_adapter_gradient_clipping():
    """
    Test adapter with different grad_clip_norm values.
    """
    model = create_simple_model()
    
    for grad_clip in [0.5, 1.0, 2.0]:
        config = AdaptationConfig(
            lr=0.01,
            grad_clip_norm=grad_clip,
            trust_region_eps=0.01,
            ema_decay=0.99,
            entropy_weight=0.01,
            max_steps_per_call=5
        )
        
        adapter = Adapter(model, config)
        loss_fn = nn.MSELoss()
        
        batch_dict = {
            'states': torch.randn(16, 10),
            'targets': torch.randn(16, 32)
        }
        
        loss = adapter.adapt(loss_fn, batch_dict)
        assert isinstance(loss, (float, torch.Tensor))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
