"""Core SSM Module: State Space Model Implementation

This module provides a full-featured State Space Model (SSM) class
for meta-reinforcement learning applications. The SSM includes:
- Linear state transition dynamics
- Recurrent cell processing
- Observation and output modules
- Model persistence (save/load)
- Comprehensive testing

Author: SSM-MetaRL Team
Date: 2025-10-20
"""

import numpy as np
import pickle
import os
from typing import Tuple, Optional, Dict, Any


class StateSpaceModel:
    """State Space Model with linear dynamics and recurrent processing.
    
    The model follows the standard SSM formulation:
        h_t = A @ h_{t-1} + B @ u_t
        y_t = C @ h_t + D @ u_t
    
    Where:
        h_t: hidden state at time t
        u_t: input/observation at time t
        y_t: output at time t
        A, B, C, D: learnable parameter matrices
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 seed: Optional[int] = None):
        """Initialize the State Space Model.
        
        Args:
            input_dim: Dimension of input observations
            hidden_dim: Dimension of hidden state
            output_dim: Dimension of model output
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize state transition matrix A (hidden to hidden)
        # Use smaller values for stability
        self.A = np.random.randn(hidden_dim, hidden_dim) * 0.1
        
        # Initialize input matrix B (input to hidden)
        self.B = np.random.randn(hidden_dim, input_dim) * 0.1
        
        # Initialize output matrix C (hidden to output)
        self.C = np.random.randn(output_dim, hidden_dim) * 0.1
        
        # Initialize feedforward matrix D (input to output)
        self.D = np.random.randn(output_dim, input_dim) * 0.1
        
        # Initialize hidden state
        self.hidden_state = np.zeros(hidden_dim)
        
    def reset_state(self) -> None:
        """Reset the hidden state to zero."""
        self.hidden_state = np.zeros(self.hidden_dim)
    
    def forward(self, observation: np.ndarray) -> np.ndarray:
        """Perform forward pass through the SSM.
        
        Args:
            observation: Input observation vector of shape (input_dim,)
            
        Returns:
            output: Model output of shape (output_dim,)
        """
        # Ensure observation has correct shape
        observation = np.asarray(observation).flatten()
        assert observation.shape[0] == self.input_dim, \
            f"Expected input dimension {self.input_dim}, got {observation.shape[0]}"
        
        # State transition: h_t = A @ h_{t-1} + B @ u_t
        self.hidden_state = self.A @ self.hidden_state + self.B @ observation
        
        # Output computation: y_t = C @ h_t + D @ u_t
        output = self.C @ self.hidden_state + self.D @ observation
        
        return output
    
    def forward_sequence(self, 
                        observations: np.ndarray,
                        reset: bool = True) -> np.ndarray:
        """Process a sequence of observations.
        
        Args:
            observations: Sequence of observations with shape (seq_len, input_dim)
            reset: Whether to reset hidden state before processing
            
        Returns:
            outputs: Sequence of outputs with shape (seq_len, output_dim)
        """
        if reset:
            self.reset_state()
        
        observations = np.asarray(observations)
        seq_len = observations.shape[0]
        outputs = np.zeros((seq_len, self.output_dim))
        
        for t in range(seq_len):
            outputs[t] = self.forward(observations[t])
        
        return outputs
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get all model parameters.
        
        Returns:
            Dictionary containing all parameter matrices
        """
        return {
            'A': self.A.copy(),
            'B': self.B.copy(),
            'C': self.C.copy(),
            'D': self.D.copy()
        }
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set model parameters.
        
        Args:
            params: Dictionary containing parameter matrices
        """
        if 'A' in params:
            self.A = params['A'].copy()
        if 'B' in params:
            self.B = params['B'].copy()
        if 'C' in params:
            self.C = params['C'].copy()
        if 'D' in params:
            self.D = params['D'].copy()
    
    def save(self, filepath: str) -> None:
        """Save model parameters and configuration to disk.
        
        Args:
            filepath: Path where model should be saved
        """
        model_data = {
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim
            },
            'parameters': self.get_parameters(),
            'hidden_state': self.hidden_state.copy()
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', 
                   exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'StateSpaceModel':
        """Load model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded StateSpaceModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new model instance
        config = model_data['config']
        model = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim']
        )
        
        # Restore parameters and state
        model.set_parameters(model_data['parameters'])
        model.hidden_state = model_data['hidden_state'].copy()
        
        print(f"Model loaded from {filepath}")
        return model
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"StateSpaceModel(input_dim={self.input_dim}, "
                f"hidden_dim={self.hidden_dim}, "
                f"output_dim={self.output_dim})")


# ============================================================================
# STANDALONE TEST CASES
# ============================================================================

def test_ssm_basic():
    """Test basic SSM functionality."""
    print("\n" + "="*60)
    print("TEST 1: Basic SSM Initialization and Forward Pass")
    print("="*60)
    
    # Create model
    model = StateSpaceModel(input_dim=4, hidden_dim=8, output_dim=2, seed=42)
    print(f"Created model: {model}")
    
    # Single forward pass
    observation = np.random.randn(4)
    output = model.forward(observation)
    
    print(f"Input shape: {observation.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden state shape: {model.hidden_state.shape}")
    assert output.shape == (2,), "Output shape mismatch"
    print("✓ Test passed!")


def test_ssm_sequence():
    """Test sequence processing."""
    print("\n" + "="*60)
    print("TEST 2: Sequence Processing")
    print("="*60)
    
    model = StateSpaceModel(input_dim=3, hidden_dim=5, output_dim=2, seed=42)
    
    # Create a sequence
    seq_len = 10
    observations = np.random.randn(seq_len, 3)
    
    # Process sequence
    outputs = model.forward_sequence(observations)
    
    print(f"Input sequence shape: {observations.shape}")
    print(f"Output sequence shape: {outputs.shape}")
    assert outputs.shape == (seq_len, 2), "Output sequence shape mismatch"
    print("✓ Test passed!")


def test_ssm_state_persistence():
    """Test that hidden state persists across calls."""
    print("\n" + "="*60)
    print("TEST 3: State Persistence")
    print("="*60)
    
    model = StateSpaceModel(input_dim=2, hidden_dim=4, output_dim=1, seed=42)
    
    # First forward pass
    obs1 = np.array([1.0, 0.0])
    out1 = model.forward(obs1)
    state1 = model.hidden_state.copy()
    
    # Second forward pass (state should have changed)
    obs2 = np.array([0.0, 1.0])
    out2 = model.forward(obs2)
    state2 = model.hidden_state.copy()
    
    print(f"State after step 1: {state1}")
    print(f"State after step 2: {state2}")
    assert not np.allclose(state1, state2), "State should change between steps"
    
    # Reset and verify state is zeroed
    model.reset_state()
    print(f"State after reset: {model.hidden_state}")
    assert np.allclose(model.hidden_state, np.zeros(4)), "State should be zero after reset"
    print("✓ Test passed!")


def test_ssm_save_load():
    """Test model saving and loading."""
    print("\n" + "="*60)
    print("TEST 4: Save and Load")
    print("="*60)
    
    # Create and configure model
    model1 = StateSpaceModel(input_dim=3, hidden_dim=6, output_dim=2, seed=42)
    obs = np.random.randn(3)
    out1 = model1.forward(obs)
    
    # Save model
    filepath = '/tmp/test_ssm.pkl'
    model1.save(filepath)
    
    # Load model
    model2 = StateSpaceModel.load(filepath)
    
    # Verify loaded model produces same output
    model2.hidden_state = model1.hidden_state.copy()  # Ensure same state
    out2 = model2.forward(obs)
    
    print(f"Original output: {out1}")
    print(f"Loaded model output: {out2}")
    assert np.allclose(out1, out2), "Loaded model should produce same outputs"
    print("✓ Test passed!")
    
    # Cleanup
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Cleaned up test file: {filepath}")


def test_ssm_parameter_access():
    """Test parameter getter and setter."""
    print("\n" + "="*60)
    print("TEST 5: Parameter Access")
    print("="*60)
    
    model = StateSpaceModel(input_dim=2, hidden_dim=3, output_dim=1, seed=42)
    
    # Get parameters
    params = model.get_parameters()
    print(f"Parameter keys: {params.keys()}")
    print(f"Matrix A shape: {params['A'].shape}")
    print(f"Matrix B shape: {params['B'].shape}")
    print(f"Matrix C shape: {params['C'].shape}")
    print(f"Matrix D shape: {params['D'].shape}")
    
    # Modify parameters
    new_params = {k: v * 0.5 for k, v in params.items()}
    model.set_parameters(new_params)
    
    # Verify modification
    updated_params = model.get_parameters()
    assert np.allclose(updated_params['A'], params['A'] * 0.5), "Parameters not updated correctly"
    print("✓ Test passed!")


if __name__ == "__main__":
    """Run all tests when script is executed directly."""
    print("\n" + "#"*60)
    print("#" + " "*18 + "SSM MODULE TESTS" + " "*22 + "#")
    print("#"*60)
    
    # Run all tests
    test_ssm_basic()
    test_ssm_sequence()
    test_ssm_state_persistence()
    test_ssm_save_load()
    test_ssm_parameter_access()
    
    print("\n" + "#"*60)
    print("#" + " "*15 + "ALL TESTS PASSED! ✓" + " "*20 + "#")
    print("#"*60 + "\n")
