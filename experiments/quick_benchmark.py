#!/usr/bin/env python3
"""
Quick Benchmark Script for SSM-MetaRL

Runs a small, fast experiment to validate the training pipeline and compare
performance with/without improvements. Designed to complete in under 2 minutes.

Usage:
    python experiments/quick_benchmark.py
"""

import os
import sys
import time
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

def run_experiment(name, args, timeout=120):
    """
    Run a single experiment with given arguments.
    
    Args:
        name: Experiment name for display
        args: List of command-line arguments to pass to main.py
        timeout: Maximum runtime in seconds (default 120)
    
    Returns:
        dict: Results including time, status, and output
    """
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, 'main.py'] + args
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time
        
        # Parse output for key metrics
        output_lines = result.stdout.split('\n')
        metrics = {}
        
        for line in output_lines:
            if 'loss' in line.lower() or 'reward' in line.lower():
                print(line)
        
        return {
            'name': name,
            'status': 'success' if result.returncode == 0 else 'failed',
            'time': elapsed,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"⚠️  Timeout after {elapsed:.1f}s")
        return {
            'name': name,
            'status': 'timeout',
            'time': elapsed,
            'returncode': -1
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error: {e}")
        return {
            'name': name,
            'status': 'error',
            'time': elapsed,
            'error': str(e)
        }

def print_summary(results):
    """
    Print a summary table of all experiments.
    
    Args:
        results: List of result dictionaries from run_experiment
    """
    print(f"\n\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}\n")
    
    print(f"{'Experiment':<35} {'Status':<10} {'Time (s)':<10}")
    print("-" * 60)
    
    for r in results:
        status_icon = {
            'success': '✓',
            'failed': '✗',
            'timeout': '⏱',
            'error': '❌'
        }.get(r['status'], '?')
        
        print(f"{r['name']:<35} {status_icon} {r['status']:<9} {r['time']:>8.1f}")
    
    print("\n" + "="*60)
    
    # Calculate statistics
    successful = [r for r in results if r['status'] == 'success']
    if len(successful) > 1:
        print("\nPerformance Comparison:")
        baseline = successful[0]
        for exp in successful[1:]:
            speedup = baseline['time'] / exp['time'] if exp['time'] > 0 else 0
            diff = exp['time'] - baseline['time']
            print(f"  {exp['name']} vs {baseline['name']}:")
            print(f"    Time difference: {diff:+.2f}s ({speedup:.2f}x)")

def main():
    """
    Main benchmark routine.
    """
    print("""\n
    ╔═══════════════════════════════════════════════════════╗
    ║   SSM-MetaRL Quick Benchmark                          ║
    ║   Testing pipeline & performance tuning               ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # Define experiments
    # Using very small parameters for speed (<2min total)
    experiments = [
        {
            'name': 'Baseline (minimal)',
            'args': [
                '--steps', '100',
                '--batch_size', '32',
                '--lr', '0.001'
            ]
        },
        {
            'name': 'With improvements',
            'args': [
                '--steps', '100',
                '--batch_size', '32',
                '--lr', '0.001',
                '--improve'
            ]
        },
        {
            'name': 'Larger batch',
            'args': [
                '--steps', '100',
                '--batch_size', '64',
                '--lr', '0.001'
            ]
        },
    ]
    
    results = []
    
    for exp in experiments:
        result = run_experiment(exp['name'], exp['args'])
        results.append(result)
        
        # Show immediate result
        if result['status'] == 'success':
            print(f"\n✓ Completed in {result['time']:.2f}s")
        elif result['status'] == 'failed':
            print(f"\n✗ Failed (exit code {result.get('returncode', 'unknown')})")
            if result.get('stderr'):
                print(f"Error output:\n{result['stderr'][:500]}")
    
    # Print final summary
    print_summary(results)
    
    print("\n💡 Performance Tuning Tips:")
    print("   - Increase --batch_size for better GPU utilization")
    print("   - Adjust --lr based on convergence speed")
    print("   - Use --improve flag to enable optimizations")
    print("   - Monitor GPU memory with nvidia-smi")
    print("\n📊 For full experiments, increase --steps and --episodes\n")
    
    # Exit with error if any experiments failed
    if any(r['status'] != 'success' for r in results):
        print("\n⚠️  Some experiments failed!")
        sys.exit(1)
    
    print("\n✓ All benchmarks passed!\n")
    sys.exit(0)

if __name__ == '__main__':
    main()
