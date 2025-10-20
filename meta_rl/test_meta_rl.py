#!/usr/bin/env python3
"""Test file for meta_rl module using pytest."""
import pytest
import torch
from meta_rl.meta_maml import MetaMAML

def test_meta_maml_import():
    """Test that MetaMAML can be imported successfully."""
    assert MetaMAML is not None, "MetaMAML class should be importable"

def test_meta_maml_initialization():
    """Test MetaMAML initialization with various parameters."""
    test_model = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )
    
    inner_lr = 0.01
    outer_lr = 0.001
    
    meta_maml = MetaMAML(
        model=test_model,
        inner_lr=inner_lr,
        outer_lr=outer_lr
    )
    
    assert meta_maml is not None, "MetaMAML should initialize successfully"
    assert meta_maml.inner_lr == inner_lr, f"Inner learning rate should be {inner_lr}"
    assert meta_maml.model is not None, "Model should be set"

def test_meta_maml_adapt():
    """Test adapt method with dummy data."""
    test_model = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )
    
    meta_maml = MetaMAML(
        model=test_model,
        inner_lr=0.01,
        outer_lr=0.001
    )
    
    support_x = torch.randn(10, 10)
    support_y = torch.randn(10, 1)
    
    adapted_params = meta_maml.adapt(support_x, support_y, num_steps=5)
    
    assert adapted_params is not None, "Adapted parameters should not be None"
    assert isinstance(adapted_params, dict), "Adapted parameters should be a dict (OrderedDict)"
    assert len(adapted_params) > 0, "Should have at least one adapted parameter"
    
    model_params = dict(test_model.named_parameters())
    assert len(adapted_params) == len(model_params), f"Expected {len(model_params)} adapted parameters, got {len(adapted_params)}"

def test_meta_maml_meta_update():
    """Test meta_update method with task batch."""
    test_model = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )
    
    meta_maml = MetaMAML(
        model=test_model,
        inner_lr=0.01,
        outer_lr=0.001
    )
    
    support_x = torch.randn(10, 10)
    support_y = torch.randn(10, 1)
    query_x = torch.randn(5, 10)
    query_y = torch.randn(5, 1)
    
    task_batch = [{
        'support_x': support_x,
        'support_y': support_y,
        'query_x': query_x,
        'query_y': query_y
    }]
    loss = meta_maml.meta_update(task_batch)
    
    assert loss is not None, "Loss should not be None"
    assert isinstance(loss, float), f"Loss should be a float, got {type(loss)}"
    assert loss >= 0, f"Loss should be non-negative, got {loss}"
    assert not torch.isnan(torch.tensor(loss)), "Loss should not be NaN"

def test_meta_maml_multiple_tasks():
    """Test meta_update with multiple tasks in batch."""
    test_model = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )
    
    meta_maml = MetaMAML(
        model=test_model,
        inner_lr=0.01,
        outer_lr=0.001
    )
    
    num_tasks = 3
    task_batch = []
    for _ in range(num_tasks):
        support_x = torch.randn(10, 10)
        support_y = torch.randn(10, 1)
        query_x = torch.randn(5, 10)
        query_y = torch.randn(5, 1)
        task_batch.append({
            'support_x': support_x,
            'support_y': support_y,
            'query_x': query_x,
            'query_y': query_y
        })
    
    assert len(task_batch) == num_tasks, f"Task batch should have {num_tasks} tasks"
    
    loss = meta_maml.meta_update(task_batch)
    
    assert loss is not None, "Loss should not be None"
    assert isinstance(loss, float), f"Loss should be a float, got {type(loss)}"
    assert loss >= 0, f"Loss should be non-negative, got {loss}"
    assert not torch.isnan(torch.tensor(loss)), "Loss should not be NaN"

def test_meta_maml_output_shapes():
    """Test that adapted parameters have correct shapes."""
    input_dim = 10
    output_dim = 1
    test_model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, output_dim)
    )
    
    meta_maml = MetaMAML(
        model=test_model,
        inner_lr=0.01,
        outer_lr=0.001
    )
    
    batch_size = 8
    support_x = torch.randn(batch_size, input_dim)
    support_y = torch.randn(batch_size, output_dim)
    
    adapted_params = meta_maml.adapt(support_x, support_y, num_steps=5)
    
    original_params = dict(test_model.named_parameters())
    for name, adapted_p in adapted_params.items():
        assert name in original_params, f"Parameter {name} not found in original model"
        original_p = original_params[name]
        assert adapted_p.shape == original_p.shape, f"Adapted parameter shape {adapted_p.shape} should match original {original_p.shape}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
