"""Plot benchmark results from JSON output.

Generates publication-quality comparison plots:
    1. Training loss curves per model/algorithm
    2. Final performance bar charts
    3. Training time comparison
    4. Parameter efficiency analysis

Usage:
    python experiments/plot_results.py results/benchmark_*.json
    python experiments/plot_results.py results/benchmark_*.json --output_dir figures/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

# Try importing matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON.

    Args:
        filepath: Path to benchmark JSON file

    Returns:
        Parsed results dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_loss_curves(results: Dict[str, Any], output_dir: str) -> None:
    """Plot training loss curves for each configuration.

    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save plots
    """
    if not _MATPLOTLIB_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, 8))
    color_idx = 0

    for run in results.get('raw_results', []):
        losses = run.get('meta_losses', [])
        if not losses:
            continue

        config = run['config']
        label = f"{config['model_type']}_{config['algorithm']}_{config['env_name']}_s{config['seed']}"

        ax.plot(losses, label=label, color=colors[color_idx % len(colors)], alpha=0.8)
        color_idx += 1

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Meta Loss', fontsize=12)
    ax.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(output_dir, 'loss_curves.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_performance_comparison(results: Dict[str, Any], output_dir: str) -> None:
    """Plot bar chart of final performance by configuration.

    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save plots
    """
    if not _MATPLOTLIB_AVAILABLE:
        return

    configs = results.get('configurations', {})
    if not configs:
        return

    names = []
    means = []
    stds = []

    for name, stats in configs.items():
        if stats.get('final_loss_mean') is not None:
            names.append(name)
            means.append(stats['final_loss_mean'])
            stds.append(stats.get('final_loss_std', 0))

    if not names:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                  color=plt.cm.Set2(np.linspace(0, 1, len(names))))

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Final Meta Loss (lower = better)', fontsize=12)
    ax.set_title('Model × Algorithm Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    filepath = os.path.join(output_dir, 'performance_comparison.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_training_time(results: Dict[str, Any], output_dir: str) -> None:
    """Plot training time comparison.

    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save plots
    """
    if not _MATPLOTLIB_AVAILABLE:
        return

    configs = results.get('configurations', {})
    if not configs:
        return

    names = []
    time_means = []
    time_stds = []

    for name, stats in configs.items():
        names.append(name)
        time_means.append(stats.get('training_time_mean', 0))
        time_stds.append(stats.get('training_time_std', 0))

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(names))
    bars = ax.barh(x, time_means, xerr=time_stds, capsize=3, alpha=0.8,
                   color=plt.cm.Paired(np.linspace(0, 1, len(names))))

    ax.set_ylabel('Configuration', fontsize=12)
    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.grid(True, axis='x', alpha=0.3)

    filepath = os.path.join(output_dir, 'training_time.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_param_efficiency(results: Dict[str, Any], output_dir: str) -> None:
    """Plot parameter count vs performance scatter.

    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save plots
    """
    if not _MATPLOTLIB_AVAILABLE:
        return

    configs = results.get('configurations', {})
    if not configs:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, stats in configs.items():
        if stats.get('final_loss_mean') is not None:
            ax.scatter(
                stats['total_params'],
                stats['final_loss_mean'],
                s=100,
                label=name,
                alpha=0.8,
                zorder=5,
            )

    ax.set_xlabel('Total Parameters', fontsize=12)
    ax.set_ylabel('Final Meta Loss', fontsize=12)
    ax.set_title('Parameter Efficiency', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(output_dir, 'param_efficiency.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def generate_all_plots(results_path: str, output_dir: str = 'figures') -> None:
    """Generate all benchmark plots from a results file.

    Args:
        results_path: Path to the benchmark JSON results
        output_dir: Directory for saving plots
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)
    print(f"  Found {results.get('num_runs', 0)} runs")

    print(f"\nGenerating plots to: {output_dir}/")
    plot_loss_curves(results, output_dir)
    plot_performance_comparison(results, output_dir)
    plot_training_time(results, output_dir)
    plot_param_efficiency(results, output_dir)

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument('results_file', type=str, help='Path to benchmark JSON file')
    parser.add_argument('--output_dir', type=str, default='figures',
                        help='Output directory for plots')

    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"Error: {args.results_file} not found")
        sys.exit(1)

    generate_all_plots(args.results_file, args.output_dir)


if __name__ == "__main__":
    main()
