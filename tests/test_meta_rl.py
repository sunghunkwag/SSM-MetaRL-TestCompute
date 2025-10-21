# -*- coding: utf-8 -*-
"""
Unit tests for MetaMAML implementation.
Tests adapt_task() with stateful models.
"""

import pytest
import torch
import torch.nn as nn
from collections import OrderedDict
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from meta_rl.meta_maml import MetaMAML
from core.ssm import StateSpaceModel # Assuming forward returns (out, next_hidden)

class TestMetaMAMLStateful:
    """Test suite for MetaMAML with stateful models like SSM."""

    @pytest.fixture
    def stateful_model(self):
        """Create a stateful SSM model."""
        return StateSpaceModel(
            state_dim=4,
            input_dim=4,
            output_dim=1,
            hidden_dim=8 # Smaller hidden dim for testing
        )

    @pytest.fixture
    def maml(self, stateful_model):
        """Create MetaMAML instance."""
        return MetaMAML(
            model=stateful_model,
            inner_lr=0.01,
            outer_lr=0.001
        )

    def test_adapt_task_returns_ordered_dict_stateful(self, maml, stateful_model):
        """Test adapt_task returns OrderedDict for stateful model."""
        B, D_in = 8, 4 # No time dimension for this basic check
        D_out = 1
        D_state = stateful_model.state_dim

        support_x = torch.randn(B, D_in)
        support_y = torch.randn(B, D_out)
        initial_hidden = stateful_model.init_hidden(B)

        # Call adapt_task
        result = maml.adapt_task(support_x, support_y, initial_hidden_state=initial_hidden, num_steps=5)

        assert isinstance(result, OrderedDict), f"Expected OrderedDict, got {type(result)}"

    def test_adapt_task_stateful_sequential(self, maml, stateful_model):
        """Test adapt_task with sequential data (Time dimension)."""
        B, T, D_in = 8, 10, 4
        D_out = 1
        D_state = stateful_model.state_dim

        support_x = torch.randn(B, T, D_in) # Data with time dimension
        support_y = torch.randn(B, T, D_out) # Targets per time step
        initial_hidden = stateful_model.init_hidden(B)

        fast_weights = maml.adapt_task(support_x, support_y, initial_hidden_state=initial_hidden, num_steps=3)

        assert isinstance(fast_weights, OrderedDict)
        assert len(fast_weights) > 0

        # Check if functional forward works with adapted weights and state over time
        test_x = torch.randn(B, T, D_in)
        hidden_state = initial_hidden
        outputs = []
        with torch.no_grad(): # Just check shapes and execution
            for t in range(T):
                x_t = test_x[:, t, :]
                output_t, hidden_state = maml.functional_forward(x_t, hidden_state, fast_weights)
                outputs.append(output_t)
        final_output = torch.stack(outputs, dim=1)

        assert final_output.shape == (B, T, D_out)

    def test_meta_update_stateful_sequential(self, maml, stateful_model):
        """Test meta_update with stateful model and sequential data."""
        B, T, D_in = 4, 5, 4 # Smaller batch/time for faster test
        D_out = 1
        D_state = stateful_model.state_dim

        # Create dummy tasks (List of tuples)
        tasks = []
        for _ in range(3): # 3 tasks
            support_x = torch.randn(B, T, D_in)
            support_y = torch.randn(B, T, D_out)
            query_x = torch.randn(B, T, D_in)
            query_y = torch.randn(B, T, D_out)
            tasks.append((support_x, support_y, query_x, query_y))

        initial_hidden = stateful_model.init_hidden(B)

        # Run meta update
        initial_params = maml.get_fast_weights()
        meta_loss = maml.meta_update(tasks, initial_hidden_state=initial_hidden)
        updated_params = maml.get_fast_weights()

        assert isinstance(meta_loss, float)
        # Check if parameters were updated
        assert any(not torch.equal(initial_params[name], updated_params[name])
                   for name in initial_params if updated_params[name].requires_grad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
