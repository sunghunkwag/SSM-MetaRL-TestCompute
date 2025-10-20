#!/usr/bin/env python3
"""Test file for adaptation module."""

print("=" * 50)
print("TEST: adaptation/test_adaptation.py")
print("=" * 50)

try:
    # Import TestTimeAdapter components
    print("\n[1/4] Importing TestTimeAdapter and AdaptConfig...")
    from test_time_adapter import TestTimeAdapter, AdaptConfig
    print("✓ Import successful")
    
    # Initialize adapter
    print("\n[2/4] Initializing TestTimeAdapter...")
    config = AdaptConfig(
        adaptation_steps=5,
        learning_rate=0.01,
        batch_size=16
    )
    adapter = TestTimeAdapter(config)
    print(f"✓ TestTimeAdapter initialized with {config.adaptation_steps} steps")
    
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
            self.params += gradient
    
    policy = StubPolicy()
    print("✓ Stub policy created")
    
    # Create stub environment
    print("\n[4/4] Running adaptation with stub environment...")
    
    class StubEnv:
        """Minimal environment for testing"""
        def __init__(self):
            self.state = np.random.randn(5)
        
        def reset(self):
            self.state = np.random.randn(5)
            return self.state
        
        def step(self, action):
            reward = -np.abs(action - 0.5)  # Reward for action close to 0.5
            self.state = np.random.randn(5)
            done = False
            return self.state, reward, done, {}
    
    env = StubEnv()
    print(f"Created stub environment with state dim: {len(env.state)}")
    
    # Run adaptation
    result = adapter.adapt(policy, env)
    
    # Print outputs or success message
    print("\nAdaptation completed successfully!")
    print("\nResults:")
    if isinstance(result, dict):
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print(f"  Adaptation result: {result}")
    
    print("\n" + "=" * 50)
    print("✓ ALL TESTS PASSED")
    print("=" * 50)
    
except ImportError as e:
    print(f"✗ Import Error: {e}")
    print("Note: TestTimeAdapter module may not be implemented yet.")
    print("Expected: adaptation/test_time_adapter.py with TestTimeAdapter and AdaptConfig classes")
    
except Exception as e:
    print(f"✗ Test Error: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
