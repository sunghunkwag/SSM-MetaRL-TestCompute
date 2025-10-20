#!/usr/bin/env python3
"""Test file for adaptation module."""
print("=" * 50)
print("TEST: adaptation/test_adaptation.py")
print("=" * 50)
try:
    # Import Adapter components
    print("\n[1/4] Importing Adapter and AdaptationConfig...")
    from adaptation.test_time_adaptation import Adapter, AdaptationConfig
    print("✓ Import successful")
    
    # Initialize adapter
    print("\n[2/4] Initializing Adapter...")
    config = AdaptationConfig(
        adaptation_steps=5,
        learning_rate=0.01,
        batch_size=16
    )
    adapter = Adapter(config)
    print(f"✓ Adapter initialized with {config.adaptation_steps} steps")
    
    # Create stub policy
    print("\n[3/4] Creating stub policy...")
    import numpy as np
    
    class StubPolicy:
        """Minimal policy for testing"""
        def __init__(self):
            self.params = np.random.randn(10)
        
        def predict(self, state):
            return np.tanh(self.params[:len(state)].dot(state))
        
        def update(self, gradient):
            self.params -= gradient
    
    policy = StubPolicy()
    print(f"✓ Policy created with {len(policy.params)} parameters")
    
    # Test adaptation
    print("\n[4/4] Running adaptation...")
    test_state = np.random.randn(5)
    result = adapter.adapt(policy, test_state)
    print(f"✓ Adaptation completed. Result type: {type(result).__name__}")
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED")
    print("=" * 50)
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {type(e).__name__}")
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    print("=" * 50)
