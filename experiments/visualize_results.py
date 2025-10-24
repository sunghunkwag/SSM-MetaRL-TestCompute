"""
Visualization Tools for Benchmark Results

This script generates publication-quality plots from benchmark results.

Usage:
    python experiments/visualize_results.py --results-dir results
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_results(results_dir: str) -> Dict[str, Dict]:
    """Load all result files from directory."""
    results_path = Path(results_dir)
    all_results = {}
    
    for filepath in results_path.glob('*_results.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
            key = f"{data['task_distribution']}_{data['method']}"
            all_results[key] = data
    
    return all_results


def plot_training_curves(results: Dict[str, Dict], output_path: str):
    """Plot training loss curves for all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for key, data in results.items():
        method = data['method']
        metrics = data['metrics']
        
        if 'train_loss' in metrics and metrics['train_loss']:
            epochs = metrics.get('epoch', list(range(len(metrics['train_loss']))))
            ax.plot(epochs, metrics['train_loss'], label=method.upper(), linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss (MSE)', fontsize=12)
    ax.set_title('Meta-Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to: {output_path}")
    plt.close()


def plot_test_performance(results: Dict[str, Dict], output_path: str):
    """Plot test performance comparison."""
    methods = []
    final_rewards = []
    
    for key, data in results.items():
        method = data['method']
        metrics = data['metrics']
        
        if 'test_reward' in metrics and metrics['test_reward']:
            methods.append(method.upper())
            final_rewards.append(metrics['test_reward'][-1])
    
    if not methods:
        print("No test reward data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    bars = ax.bar(methods, final_rewards, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Final Test Reward', fontsize=12)
    ax.set_title('Test Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved test performance to: {output_path}")
    plt.close()


def plot_adaptation_curves(results: Dict[str, Dict], output_path: str):
    """Plot adaptation curves showing loss reduction."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for key, data in results.items():
        method = data['method']
        metrics = data['metrics']
        
        if 'test_initial_loss' in metrics and 'test_final_loss' in metrics:
            if metrics['test_initial_loss'] and metrics['test_final_loss']:
                initial = metrics['test_initial_loss']
                final = metrics['test_final_loss']
                
                # Plot adaptation trajectory
                ax.plot([0, 1], [np.mean(initial), np.mean(final)], 
                       label=method.upper(), linewidth=2, marker='o', markersize=8)
    
    ax.set_xlabel('Adaptation Progress', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Test-Time Adaptation', fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Initial', 'After Adaptation'])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved adaptation curves to: {output_path}")
    plt.close()


def generate_summary_table(results: Dict[str, Dict], output_path: str):
    """Generate a summary table of all results."""
    with open(output_path, 'w') as f:
        f.write("# Benchmark Results Summary\n\n")
        
        for key, data in results.items():
            task = data['task_distribution']
            method = data['method']
            metrics = data['metrics']
            
            f.write(f"## {task.upper()} - {method.upper()}\n\n")
            
            # Training metrics
            if 'train_loss' in metrics and metrics['train_loss']:
                initial_loss = metrics['train_loss'][0]
                final_loss = metrics['train_loss'][-1]
                f.write(f"- **Training Loss**: {initial_loss:.4f} → {final_loss:.4f}\n")
            
            # Test metrics
            if 'test_reward' in metrics and metrics['test_reward']:
                final_reward = metrics['test_reward'][-1]
                f.write(f"- **Test Reward**: {final_reward:.2f}\n")
            
            # Adaptation metrics
            if 'test_initial_loss' in metrics and 'test_final_loss' in metrics:
                if metrics['test_initial_loss'] and metrics['test_final_loss']:
                    initial = np.mean(metrics['test_initial_loss'])
                    final = np.mean(metrics['test_final_loss'])
                    improvement = ((initial - final) / initial * 100) if initial > 0 else 0
                    f.write(f"- **Adaptation**: {initial:.4f} → {final:.4f} ({improvement:.1f}% improvement)\n")
            
            # Timing
            if 'total_time' in metrics:
                f.write(f"- **Runtime**: {metrics['total_time']:.2f}s\n")
            
            f.write("\n")
    
    print(f"Saved summary table to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing result JSON files')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Directory to save figures')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} result files")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot_training_curves(results, str(output_path / 'training_curves.png'))
    plot_test_performance(results, str(output_path / 'test_performance.png'))
    plot_adaptation_curves(results, str(output_path / 'adaptation_curves.png'))
    generate_summary_table(results, str(output_path / 'summary.md'))
    
    print("\n✓ All visualizations generated successfully!")


if __name__ == "__main__":
    main()

