#!/usr/bin/env python3
"""Test file for meta_rl module."""

print("=" * 50)
print("TEST: meta_rl/test_meta_rl.py")
print("=" * 50)

try:
    # Import MetaLearner components
    print("\n[1/3] Importing MetaLearner and MetaConfig...")
    from meta_learner import MetaLearner, MetaConfig
    print("✓ Import successful")
    
    # Initialize MetaLearner
    print("\n[2/3] Initializing MetaLearner...")
    config = MetaConfig(
        num_tasks=5,
        inner_steps=10,
        outer_steps=3,
        learning_rate=0.001
    )
    meta_learner = MetaLearner(config)
    print(f"✓ MetaLearner initialized with {config.num_tasks} tasks, {config.outer_steps} outer steps")
    
    # Run dummy outer_step with placeholder tasks
    print("\n[3/3] Running dummy outer_step with placeholder tasks...")
    
    # Create placeholder tasks
    class DummyTask:
        def __init__(self, task_id):
            self.task_id = task_id
            self.name = f"Task_{task_id}"
        
        def sample_batch(self):
            """Return dummy batch data"""
            import numpy as np
            return {
                'states': np.random.randn(32, 64),
                'actions': np.random.randn(32, 16),
                'rewards': np.random.randn(32)
            }
    
    tasks = [DummyTask(i) for i in range(config.num_tasks)]
    print(f"Created {len(tasks)} placeholder tasks")
    
    # Run outer step
    metrics = meta_learner.outer_step(tasks)
    
    # Print metrics/results
    print("\nOuter step completed successfully!")
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    print("✓ ALL TESTS PASSED")
    print("=" * 50)
    
except ImportError as e:
    print(f"✗ Import Error: {e}")
    print("Note: MetaLearner module may not be implemented yet.")
    print("Expected: meta_rl/meta_learner.py with MetaLearner and MetaConfig classes")
    
except Exception as e:
    print(f"✗ Test Error: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
