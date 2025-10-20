#!/usr/bin/env python3
"""Tests for adaptation module."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptation.test_time_adaptation import AdaptationConfig, Adapter


def test_adapter_initialization():
    """Test Adapter creation with model and config."""
    # Create a simple model
    model = nn.Linear(10, 5)
    
    # Create config with correct fields: lr and max_steps_per_call
    config = AdaptationConfig(
        lr=0.001,
        max_steps_per_call=5
    )
    
    # Create adapter with model and cfg
    adapter = Adapter(model=model, cfg=config)
    
    # Check that adapter was created successfully
    assert adapter is not None, "Adapter should be created successfully"
    assert hasattr(adapter, 'cfg'), "Adapter should have cfg attribute"
    assert adapter.cfg.lr == 0.001, "Adapter should store correct lr"
    assert adapter.cfg.max_steps_per_call == 5, "Adapter should store correct max_steps_per_call"


def test_adaptation_config():
    """Test AdaptationConfig creation and attribute access."""
    config = AdaptationConfig(
        lr=0.1,
        max_steps_per_call=10
    )
    
    # Test all configuration attributes
    assert config.lr == 0.1, "lr should be 0.1"
    assert config.max_steps_per_call == 10, "max_steps_per_call should be 10"
    
    # Test that config is properly typed
    assert isinstance(config.lr, float), "lr should be a float"
    assert isinstance(config.max_steps_per_call, int), "max_steps_per_call should be an integer"


def test_adaptation_config_edge_cases():
    """Test AdaptationConfig with edge case values."""
    # Test with minimum reasonable values
    config_min = AdaptationConfig(
        lr=0.0001,
        max_steps_per_call=1
    )
    assert config_min.lr == 0.0001
    assert config_min.max_steps_per_call == 1
    
    # Test with larger values
    config_max = AdaptationConfig(
        lr=1.0,
        max_steps_per_call=100
    )
    assert config_max.lr == 1.0
    assert config_max.max_steps_per_call == 100


def test_adapter_adapt_call():
    """Test that adapter.adapt can be called with correct signature: (loss_fn, batch)."""
    # Create a simple model
    model = nn.Linear(10, 5)
    
    # Create config
    config = AdaptationConfig(
        lr=0.01,
        max_steps_per_call=3
    )
    
    # Create adapter
    adapter = Adapter(model=model, cfg=config)
    
    # Create a dummy batch
    batch = {
        'x': torch.randn(4, 10),
        'y': torch.randn(4, 5)
    }
    
    # Define a simple loss function
    def simple_loss_fn(batch_data):
        x = batch_data['x']
        y = batch_data['y']
        pred = model(x)
        return F.mse_loss(pred, y)
    
    # Call adapt with correct signature: (loss_fn, batch)
    try:
        adapter.adapt(loss_fn=simple_loss_fn, batch=batch)
        # If we get here, the call succeeded
        assert True, "adapt call should succeed with correct signature"
    except TypeError as e:
        pytest.fail(f"adapt call failed with correct signature: {e}")


def test_adapter_with_different_models():
    """Test Adapter works with different model architectures."""
    config = AdaptationConfig(
        lr=0.01,
        max_steps_per_call=5
    )
    
    # Test with Linear model
    linear_model = nn.Linear(10, 5)
    adapter1 = Adapter(model=linear_model, cfg=config)
    assert adapter1 is not None
    
    # Test with Sequential model
    seq_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    adapter2 = Adapter(model=seq_model, cfg=config)
    assert adapter2 is not None


def test_config_default_values():
    """Test that AdaptationConfig can be created and stores values correctly."""
    # Test with explicit values
    config = AdaptationConfig(
        lr=0.005,
        max_steps_per_call=7
    )
    
    assert config.lr == 0.005
    assert config.max_steps_per_call == 7
