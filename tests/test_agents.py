import os
import torch
import pytest
from multi_agent.tools.ssm_tool import SSMTool

def test_ssm_tool_creation():
    """Test that SSMTool creates and saves a model correctly."""
    tool = SSMTool()

    # Test parameters
    state_dim = 32
    input_dim = 10
    output_dim = 5
    hidden_dim = 64

    # Run tool
    result = tool._run(state_dim, input_dim, output_dim, hidden_dim)

    # Verify result structure
    assert result["status"] == "success"
    assert "model_path" in result
    assert os.path.exists(result["model_path"])

    # Verify saved artifact
    checkpoint = torch.load(result["model_path"])
    assert "state_dict" in checkpoint
    assert "config" in checkpoint

    config = checkpoint["config"]
    assert config["state_dim"] == state_dim
    assert config["input_dim"] == input_dim
    assert config["output_dim"] == output_dim

    # Cleanup
    if os.path.exists(result["model_path"]):
        os.remove(result["model_path"])

def test_ssm_tool_error_handling():
    """Test that SSMTool handles errors gracefully."""
    tool = SSMTool()

    # Invalid parameters (negative dimensions) should raise an error in PyTorch or validation
    # passing strings where ints are expected might cause issues if not handled,
    # but here we assume the tool takes args.
    # Let's try to force an error by passing invalid dimensions to the underlying model

    # Note: The tool converts args to int, but let's try a case that fails inside SSM init
    # e.g. negative dimension
    result = tool._run(-10, 10, 5, 64)

    assert result["status"] == "error"
    assert "error_message" in result
