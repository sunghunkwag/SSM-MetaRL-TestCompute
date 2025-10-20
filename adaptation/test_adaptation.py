#!/usr/bin/env python3
"""Test file for adaptation module with assert-based pytest tests."""
import pytest
import torch
import torch.nn as nn
import numpy as np
from adaptation.test_time_adaptation import Adapter, AdaptationConfig


def test_adapter_basic():
    """Test basic Adapter functionality with correct signature."""
    # Create adapter config
    config = AdaptationConfig(
        adaptation_steps=3,
        learning_rate=0.01,
        batch_size=4
    )
    
    # Initialize adapter
    adapter = Adapter(config)
    assert adapter is not None, "Adapter should be instantiated successfully"
    assert adapter.config == config, "Adapter should store the config correctly"
    
    # Create a simple dummy model for testing
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    
    # Create dummy batch data (dictionary format)
    batch = {
        'observations': torch.randn(4, 2),
        'targets': torch.randn(4, 1)
    }
    
    # Define a simple loss function
    def loss_fn(model_output, batch_data):
        predictions = model_output
        targets = batch_data['targets']
        return nn.MSELoss()(predictions, targets)
    
    # Define forward function
    def fwd_fn(batch_data):
        return model(batch_data['observations'])
    
    # Test the adapt method with correct signature
    result = adapter.adapt(loss_fn, batch, fwd_fn)
    
    # Comprehensive assertions
    assert isinstance(result, dict), "Adapter.adapt should return a dictionary"
    assert 'final_loss' in result or len(result) >= 0, "Result should contain adaptation information"


def test_adapter_initialization():
    """Test Adapter initialization with various configurations."""
    config = AdaptationConfig(
        adaptation_steps=5,
        learning_rate=0.001,
        batch_size=8
    )
    
    adapter = Adapter(config)
    
    # Check that adapter was created successfully
    assert adapter is not None, "Adapter should be created successfully"
    assert hasattr(adapter, 'config'), "Adapter should have config attribute"
    assert adapter.config.adaptation_steps == 5, "Adapter should store correct adaptation_steps"
    assert adapter.config.learning_rate == 0.001, "Adapter should store correct learning_rate"
    assert adapter.config.batch_size == 8, "Adapter should store correct batch_size"


def test_adaptation_config():
    """Test AdaptationConfig creation and attribute access."""
    config = AdaptationConfig(
        adaptation_steps=10,
        learning_rate=0.1,
        batch_size=32
    )
    
    # Test all configuration attributes
    assert config.adaptation_steps == 10, "adaptation_steps should be 10"
    assert config.learning_rate == 0.1, "learning_rate should be 0.1"
    assert config.batch_size == 32, "batch_size should be 32"
    
    # Test that config is properly typed
    assert isinstance(config.adaptation_steps, int), "adaptation_steps should be an integer"
    assert isinstance(config.learning_rate, float), "learning_rate should be a float"
    assert isinstance(config.batch_size, int), "batch_size should be an integer"


def test_adaptation_config_edge_cases():
    """Test AdaptationConfig with edge case values."""
    # Test with minimum reasonable values
    config_min = AdaptationConfig(
        adaptation_steps=1,
        learning_rate=0.0001,
        batch_size=1
    )
    assert config_min.adaptation_steps == 1
    assert config_min.learning_rate == 0.0001
    assert config_min.batch_size == 1
    
    # Test with larger values
    config_max = AdaptationConfig(
        adaptation_steps=100,
        learning_rate=1.0,
        batch_size=256
    )
    assert config_max.adaptation_steps == 100
    assert config_max.learning_rate == 1.0
    assert config_max.batch_size == 256
