#!/usr/bin/env python3
"""Test file for adaptation module with correct Adapter usage."""

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
    
    # Basic assertions
    assert isinstance(result, dict), "Adapter.adapt should return a dictionary"
    
    print("✓ test_adapter_basic passed")


def test_adapter_initialization():
    """Test Adapter initialization."""
    config = AdaptationConfig(
        adaptation_steps=5,
        learning_rate=0.001,
        batch_size=8
    )
    
    adapter = Adapter(config)
    
    # Check that adapter was created successfully
    assert adapter is not None, "Adapter should be created successfully"
    
    print("✓ test_adapter_initialization passed")


def test_adaptation_config():
    """Test AdaptationConfig creation."""
    config = AdaptationConfig(
        adaptation_steps=10,
        learning_rate=0.1,
        batch_size=32
    )
    
    assert config.adaptation_steps == 10
    assert config.learning_rate == 0.1
    assert config.batch_size == 32
    
    print("✓ test_adaptation_config passed")


if __name__ == "__main__":
    print("=" * 50)
    print("TEST: adaptation/test_adaptation.py (pytest style)")
    print("=" * 50)
    
    try:
        test_adapter_initialization()
        test_adaptation_config()
        test_adapter_basic()
        
        print("\n✓ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise
    
    print("=" * 50)
