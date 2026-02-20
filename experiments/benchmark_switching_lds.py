"""Non-stationary Switching Linear Dynamical System (LDS) Benchmark

Tests SSM ability to track regime changes in dynamical systems.
The system switches between K distinct linear dynamics regimes,
each with different state transition and observation matrices.

This is the PRIMARY benchmark: evaluating whether Mamba SSM or Legacy SSM
can better adapt to non-stationary dynamics via MAML meta-learning.

Mathematical formulation:
    State:       x_{t+1} = A_k * x_t + B_k * u_t + process_noise
    Observation: y_t     = C_k * x_t + obs_noise
    where k = regime index, switching every T_switch steps

Usage:
    python experiments/benchmark_switching_lds.py
"""

import sys
import time
import json
import platform
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.ssm import StateSpaceModel
from core.ssm_mamba import MambaSSM
from meta_rl.meta_maml import MetaMAML

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================================
# Switching LDS Data Generator
# ============================================================================

@dataclass
class LDSRegime:
    """A single linear dynamical system regime.

    Attributes:
        A: State transition matrix (state_dim, state_dim)
        B: Input matrix (state_dim, input_dim)
        C: Observation matrix (obs_dim, state_dim)
        process_noise_std: Standard deviation of process noise
        obs_noise_std: Standard deviation of observation noise
    """
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    process_noise_std: float = 0.01
    obs_noise_std: float = 0.05


class SwitchingLDS:
    """Non-stationary Switching Linear Dynamical System.

    Generates time series data where the underlying dynamics switch
    between K distinct regimes. This tests whether an SSM can detect
    and adapt to abrupt changes in system dynamics.

    Args:
        state_dim: Dimension of hidden state x
        input_dim: Dimension of control input u
        obs_dim: Dimension of observed output y
        num_regimes: Number of distinct dynamic regimes (K)
        switch_interval: Steps between regime switches
        spectral_radius: Max eigenvalue magnitude for stable A matrices
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        state_dim: int = 4,
        input_dim: int = 2,
        obs_dim: int = 6,
        num_regimes: int = 3,
        switch_interval: int = 50,
        spectral_radius: float = 0.95,
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.obs_dim = obs_dim
        self.num_regimes = num_regimes
        self.switch_interval = switch_interval

        self.rng = np.random.RandomState(seed)
        self.regimes = self._generate_regimes(spectral_radius)

        logger.info(
            f"SwitchingLDS created: {num_regimes} regimes, "
            f"state_dim={state_dim}, obs_dim={obs_dim}, "
            f"switch every {switch_interval} steps"
        )

    def _generate_regimes(self, spectral_radius: float) -> List[LDSRegime]:
        """Generate K dynamically distinct but stable regimes.

        Each regime has a different A matrix with controlled spectral radius
        to ensure stability, and different B, C matrices for distinct
        input-output behavior.
        """
        regimes = []

        for k in range(self.num_regimes):
            # Generate stable A: random rotation + scaling
            # Use QR decomposition to get orthogonal matrix, then scale
            raw = self.rng.randn(self.state_dim, self.state_dim)
            Q, R = np.linalg.qr(raw)

            # Generate eigenvalues with controlled magnitude
            # Different regimes get different eigenvalue patterns
            if k == 0:
                # Regime 0: slow decay, smooth dynamics
                eig_magnitudes = spectral_radius * np.ones(self.state_dim) * 0.98
            elif k == 1:
                # Regime 1: oscillatory dynamics
                eig_magnitudes = spectral_radius * np.ones(self.state_dim) * 0.90
            else:
                # Regime 2+: mixed dynamics with varying speeds
                eig_magnitudes = spectral_radius * self.rng.uniform(
                    0.7, 0.99, self.state_dim
                )

            # Create diagonal eigenvalue matrix and reconstruct A
            Lambda = np.diag(eig_magnitudes)
            A = Q @ Lambda @ Q.T

            # Verify stability
            max_eig = np.max(np.abs(np.linalg.eigvals(A)))
            assert max_eig < 1.0, f"Regime {k}: unstable A (max eigenvalue = {max_eig})"

            # Different B matrices for different input responses
            B = self.rng.randn(self.state_dim, self.input_dim) * 0.5

            # Different C matrices for different observation projections
            C = self.rng.randn(self.obs_dim, self.state_dim) * 0.3

            # Vary noise levels across regimes
            process_noise = 0.01 * (1 + 0.5 * k)
            obs_noise = 0.05 * (1 + 0.3 * k)

            regimes.append(LDSRegime(
                A=A, B=B, C=C,
                process_noise_std=process_noise,
                obs_noise_std=obs_noise,
            ))

            logger.debug(
                f"  Regime {k}: max_eig={max_eig:.4f}, "
                f"process_noise={process_noise:.3f}, obs_noise={obs_noise:.3f}"
            )

        return regimes

    def generate_sequence(
        self,
        total_steps: int = 300,
        regime_schedule: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate a switching LDS sequence.

        Args:
            total_steps: Total number of timesteps
            regime_schedule: Optional explicit regime schedule.
                If None, cycles through regimes every switch_interval steps.

        Returns:
            observations: (total_steps, obs_dim)
            inputs: (total_steps, input_dim)
            states: (total_steps, state_dim) — ground truth hidden states
            regime_labels: (total_steps,) — which regime was active
        """
        # Build regime schedule
        if regime_schedule is None:
            regime_labels = np.zeros(total_steps, dtype=int)
            for t in range(total_steps):
                regime_labels[t] = (t // self.switch_interval) % self.num_regimes
        else:
            regime_labels = np.array(regime_schedule[:total_steps])

        # Initialize
        x = self.rng.randn(self.state_dim) * 0.1
        states = np.zeros((total_steps, self.state_dim))
        observations = np.zeros((total_steps, self.obs_dim))
        inputs = self.rng.randn(total_steps, self.input_dim) * 0.3

        # Simulate
        for t in range(total_steps):
            regime = self.regimes[regime_labels[t]]

            # Observation
            y = regime.C @ x + self.rng.randn(self.obs_dim) * regime.obs_noise_std
            observations[t] = y
            states[t] = x

            # State transition
            x = (regime.A @ x
                 + regime.B @ inputs[t]
                 + self.rng.randn(self.state_dim) * regime.process_noise_std)

        return observations, inputs, states, regime_labels

    def generate_meta_tasks(
        self,
        num_tasks: int = 20,
        steps_per_task: int = 300,
    ) -> List[Dict[str, np.ndarray]]:
        """Generate multiple tasks for meta-learning.

        Each task has a different random seed and potentially different
        regime switching patterns, but the same underlying regimes.

        Args:
            num_tasks: Number of meta-learning tasks
            steps_per_task: Steps per task sequence

        Returns:
            List of task dictionaries with observations, inputs, etc.
        """
        tasks = []
        for i in range(num_tasks):
            # Vary the switching pattern per task
            self.rng = np.random.RandomState(42 + i * 137)
            obs, inputs, states, labels = self.generate_sequence(steps_per_task)

            tasks.append({
                'observations': obs,
                'inputs': inputs,
                'states': states,
                'regime_labels': labels,
                'task_id': i,
            })

        return tasks


# ============================================================================
# Benchmark Runner
# ============================================================================

def prepare_maml_data(
    task: Dict[str, np.ndarray],
    support_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a switching LDS task into MAML support/query sets.

    Uses observation prediction: given y_t, predict y_{t+1}.

    Args:
        task: Dictionary from SwitchingLDS.generate_meta_tasks()
        support_ratio: Fraction of data to use as support set

    Returns:
        (support_x, support_y, query_x, query_y) tensors
    """
    obs = task['observations']  # (T, obs_dim)

    # Create (input, target) pairs: predict next observation
    x = obs[:-1]  # (T-1, obs_dim)
    y = obs[1:]   # (T-1, obs_dim)

    split = int(len(x) * support_ratio)

    support_x = torch.tensor(x[:split], dtype=torch.float32).unsqueeze(0)
    support_y = torch.tensor(y[:split], dtype=torch.float32).unsqueeze(0)
    query_x = torch.tensor(x[split:], dtype=torch.float32).unsqueeze(0)
    query_y = torch.tensor(y[split:], dtype=torch.float32).unsqueeze(0)

    return support_x, support_y, query_x, query_y


def compute_regime_switch_mse(
    model: nn.Module,
    task: Dict[str, np.ndarray],
    window: int = 10,
) -> Dict[str, float]:
    """Compute prediction MSE specifically around regime switch points.

    This measures how quickly the model adapts after a regime change.

    Args:
        model: Trained SSM model
        task: Task data dictionary
        window: Number of steps around switch to evaluate

    Returns:
        Dictionary with MSE at switches vs. steady-state
    """
    obs = task['observations']
    labels = task['regime_labels']

    # Find switch points
    switch_points = []
    for t in range(1, len(labels)):
        if labels[t] != labels[t - 1]:
            switch_points.append(t)

    if not switch_points:
        return {'switch_mse': 0.0, 'steady_mse': 0.0, 'ratio': 1.0}

    # Predict full sequence
    x = torch.tensor(obs[:-1], dtype=torch.float32).unsqueeze(0)
    y_true = torch.tensor(obs[1:], dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        hidden = model.init_hidden(batch_size=1)
        y_pred, _ = model(x, hidden)

    errors = (y_pred - y_true).pow(2).mean(dim=-1).squeeze().numpy()

    # Separate switch-region errors from steady-state errors
    switch_mask = np.zeros(len(errors), dtype=bool)
    for sp in switch_points:
        start = max(0, sp - 1)
        end = min(len(errors), sp + window)
        switch_mask[start:end] = True

    switch_mse = float(errors[switch_mask].mean()) if switch_mask.any() else 0.0
    steady_mse = float(errors[~switch_mask].mean()) if (~switch_mask).any() else 0.0
    ratio = switch_mse / max(steady_mse, 1e-8)

    return {
        'switch_mse': switch_mse,
        'steady_mse': steady_mse,
        'ratio': ratio,
        'num_switches': len(switch_points),
    }


# ============================================================================
# Switch-Aligned Evaluation (User Request)
# ============================================================================

from adaptation.test_time_adaptation import Adapter, AdaptationConfig
import copy

class SwitchEvaluator:
    """Evaluates SSM adaptation performance around regime switches.

    Aligns evaluation at switch points (delta_t = 0) and computes
    windowed MSE and Area Under Adaptation (AUA) metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        window_size: int = 50,
        device: str = 'cpu'
    ):
        self.model = model
        self.window_size = window_size
        self.device = device
        self.window_defs = {
            'Pre': (-5, -1),
            'Shock': (1, 5),
            'Recover': (6, 20),
            'Steady': (21, 50)
        }

    def evaluate_task(
        self,
        task: Dict[str, np.ndarray],
        use_adaptation: bool = False,
        lr: float = 0.01
    ) -> Dict:
        """Evaluate a single task with sliding windows aligned to switches."""
        obs = task['observations']
        labels = task['regime_labels']

        # Aligned switch points (where label changes)
        switch_indices = []
        for t in range(1, len(labels)):
            if labels[t] != labels[t-1]:
                switch_indices.append(t)

        if not switch_indices:
            return []

        num_steps = len(obs) - 1
        x_tensor = torch.tensor(obs[:-1], dtype=torch.float32).to(self.device)
        y_true_tensor = torch.tensor(obs[1:], dtype=torch.float32).to(self.device)

        mse_time_series = np.zeros(num_steps)
        
        # Work on a copy of the model to avoid global parameter leakage
        eval_model = copy.deepcopy(self.model).to(self.device)
        
        # Setup Refine Mamba TTA
        params_to_adapt = None
        lr = 0.01
        if isinstance(eval_model, MambaSSM):
            lr = 0.001
            if hasattr(eval_model, 'output_projection'):
                params_to_adapt = list(eval_model.output_projection.parameters())
        elif isinstance(eval_model, StateSpaceModel):
            # For Legacy, also restrict to output head for fair comparison
            if hasattr(eval_model, 'output_network'):
                params_to_adapt = list(eval_model.output_network.parameters())
        
        if use_adaptation:
            config = AdaptationConfig(
                learning_rate=lr,
                num_steps=1,
                grad_clip_norm=1.0
            )
            adapter = Adapter(eval_model, config, device=self.device, params_to_adapt=params_to_adapt)
        else:
            adapter = None

        eval_model.eval()

        # We process step-by-step to allow for adaptation and hidden state flow
        hidden = eval_model.init_hidden(batch_size=1)

        for t in range(num_steps):
            inp = x_tensor[t:t+1]      # (1, obs_dim)
            target = y_true_tensor[t:t+1] # (1, obs_dim)

            # 1. Prediction (before adaptation for this step's performance)
            with torch.no_grad():
                # We need [1, 1, obs_dim] for sequence models if they expect it
                # Our models expect (batch, time, dim)
                pred, next_hidden = eval_model(inp.unsqueeze(0), hidden)
                mse = F.mse_loss(pred.squeeze(0), target).item()
                mse_time_series[t] = mse

            # 2. Adaptation Step (TTA)
            if use_adaptation:
                # Adapter.update_step handles model.train() and optimizer internally
                # It also returns next_hidden (detached)
                _, _ = adapter.update_step(inp.unsqueeze(0), target.unsqueeze(0), hidden)
                # Re-run forward to get updated hidden for next step
                # Note: adapter.update_step already updated model params
                with torch.no_grad():
                    _, hidden = eval_model(inp.unsqueeze(0), hidden)
            else:
                hidden = next_hidden

        # Component Metrics per Switch
        results_per_switch = []
        for ts in switch_indices:
            # We align at ts. Delta_t = t - ts.
            switch_res = {}
            for name, (start_dt, end_dt) in self.window_defs.items():
                s_idx = ts + start_dt
                e_idx = ts + end_dt  # Inclusive range implementation
                # Boundary checks
                s_idx = max(0, min(num_steps - 1, s_idx))
                e_idx = max(0, min(num_steps - 1, e_idx))

                if s_idx <= e_idx:
                    window_mse = np.mean(mse_time_series[s_idx:e_idx+1])
                else:
                    window_mse = 0.0
                switch_res[name] = window_mse

            # AUA: sum_{Δt=0..20} MSE(Δt)
            s_aua = ts + 0
            e_aua = ts + 20
            s_aua = max(0, min(num_steps - 1, s_aua))
            e_aua = max(0, min(num_steps - 1, e_aua))
            switch_res['AUA'] = np.sum(mse_time_series[s_aua:e_aua+1])

            # Recovery Time: first Δt where MSE ≤ 1.1 * Steady
            steady_val = switch_res['Steady']
            threshold = 1.1 * steady_val
            rec_time = -1
            for dt in range(0, 51): # Check up to 50 steps
                idx = ts + dt
                if idx < num_steps:
                    if mse_time_series[idx] <= threshold:
                        rec_time = dt
                        break
            switch_res['RecTime'] = rec_time if rec_time != -1 else 50.0

            results_per_switch.append(switch_res)

        return results_per_switch

    def run_benchmark(self, tasks: List[Dict], use_adaptation: bool = False):
        """Run evaluation across all tasks and aggregate stats."""
        all_switch_results = []
        for task in tasks:
            task_res = self.evaluate_task(task, use_adaptation=use_adaptation)
            all_switch_results.extend(task_res)

        # Aggregate stats (mean +/- std)
        metrics = ['Pre', 'Shock', 'Recover', 'Steady', 'AUA', 'RecTime']
        summary = {}
        for m in metrics:
            vals = [r[m] for r in all_switch_results if m in r]
            if vals:
                summary[f"{m}_mean"] = np.mean(vals)
                summary[f"{m}_std"] = np.std(vals)
            else:
                summary[f"{m}_mean"] = 0.0
                summary[f"{m}_std"] = 0.0

        return summary


def main():
    print("=" * 70)
    print("  Benchmark 1: Non-stationary Switching LDS (Synthetic)")
    print("  SSM regime-switching detection & adaptation")
    print("=" * 70)

    # Use a shared LDS environment for both models
    lds = SwitchingLDS(
        state_dim=4,
        input_dim=2,
        obs_dim=6,
        num_regimes=3,
        switch_interval=50,
        seed=42
    )

    models_trained = {}
    
    # 0. State Persistence Sanity Check
    print("\nRunning state persistence sanity check...", flush=True)
    temp_model = MambaSSM(state_dim=16, input_dim=6, output_dim=6, d_model=64)
    h = temp_model.init_hidden(1)
    norms = []
    if h is not None:
        norms.append(torch.norm(h).item())
        for _ in range(3):
            _, h = temp_model(torch.randn(1, 1, 6), h)
            norms.append(torch.norm(h).item())
        
        # Check difference and non-zero
        diffs = [abs(norms[i+1] - norms[i]) > 1e-6 for i in range(len(norms)-1)]
        is_nonzero = all(n > 1e-6 for n in norms[1:])
        if all(diffs) and is_nonzero:
            print(f"State persistence sanity check: PASS | h norms: {['{:.4f}'.format(n) for n in norms]}", flush=True)
        else:
            print(f"State persistence sanity check: FAIL | h norms: {norms}", flush=True)
    else:
        print("State persistence sanity check: SKIPPED (No hidden state returned)", flush=True)

    # 1. Training Phase
    for model_type in ['legacy', 'mamba']:
        print(f"\nTraining {model_type.upper()}...")
        sys.stdout.flush()
        
        if model_type == 'mamba':
            model = MambaSSM(state_dim=16, input_dim=6, output_dim=6, d_model=64)
        else:
            model = StateSpaceModel(state_dim=16, input_dim=6, output_dim=6, hidden_dim=64)

        meta_learner = MetaMAML(model=model, inner_lr=0.01, outer_lr=0.001)
        tasks = lds.generate_meta_tasks(num_tasks=16, steps_per_task=300)

        num_epochs = 30
        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_start = time.time()
            batch_tasks = [prepare_maml_data(tasks[i]) for i in np.random.choice(16, 8, replace=False)]
            loss = meta_learner.meta_update(batch_tasks, initial_hidden_state=model.init_hidden(1), loss_fn=nn.MSELoss())
            
            # ETA computation
            elapsed = time.time() - start_time
            avg_epoch_time = elapsed / (epoch + 1)
            remaining_epochs = num_epochs - (epoch + 1)
            eta = avg_epoch_time * remaining_epochs
            
            print(f"  Epoch {epoch+1:2d}/{num_epochs} | Loss: {loss:.6f} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", flush=True)

        models_trained[model_type] = model
        
        # Save model
        model_path = project_root / 'results' / f'model_{model_type}_lds.pt'
        model_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"  Saved {model_type} model to {model_path}", flush=True)

    # 2. Aligned Evaluation
    eval_tasks = lds.generate_meta_tasks(num_tasks=10, steps_per_task=500)
    print("\nStarting switch-aligned evaluation (10 tasks, 500 steps each)...")

    results_eval = {}
    for model_type, model in models_trained.items():
        evaluator = SwitchEvaluator(model)
        print(f"  Evaluating {model_type}...")

        # (a) Adaptation OFF
        res_off = evaluator.run_benchmark(eval_tasks, use_adaptation=False)
        # (b) Adaptation ON
        res_on = evaluator.run_benchmark(eval_tasks, use_adaptation=True)

        results_eval[model_type] = {'OFF': res_off, 'ON': res_on}

    # Assertion: Ensure both models were evaluated
    if 'legacy' not in results_eval or 'mamba' not in results_eval:
        msg = f"CRITICAL FAILURE: Missing evaluation results. Found keys: {list(results_eval.keys())}"
        print(f"\n{msg}", flush=True)
        raise RuntimeError(msg)

    # 3. Final Output Tables (Markdown)
    for model_type in ['legacy', 'mamba']:
        print(f"\n### Model: {model_type.upper()}")
        for mode in ['OFF', 'ON']:
            print(f"\n#### Mode: TTA {mode}")
            print("| Metric | mean ± std |")
            print("| :--- | :--- |")
            
            res = results_eval[model_type][mode]
            for m in ['Pre', 'Shock', 'Recover', 'Steady', 'AUA', 'RecTime']:
                label = m
                if m == 'RecTime': label = "Time-to-Recover (Δt)"
                
                val = res[m+'_mean']
                std = res[m+'_std']
                
                if m in ['AUA', 'RecTime']:
                    print(f"| {label} | {val:.2f} ± {std:.2f} |")
                else:
                    print(f"| {label} | {val:.4f} ± {std:.4f} |")
            print()

    print("\nWindow Definitions:")
    print("  Pre: Δt ∈ [-5, -1], Shock: Δt ∈ [1, 5], Recover: Δt ∈ [6, 20], Steady: Δt ∈ [21, 50]")
    print("  AUA: sum of MSE across Δt ∈ [0, 20]")
    print("TTA Updates Applied:")
    print("  Adapt OFF: No (fixed state updates, no parameter updates)")
    print("  Adapt ON: Yes (test-time SGD updates per step, lr=0.01)")

    # 4. Save results to JSON
    results_path = project_root / 'results' / 'switching_lds_benchmark.json'
    results_path.parent.mkdir(exist_ok=True)

    save_data = {
        "benchmark": "Non-stationary Switching LDS (Enhanced Aligned Eval)",
        "config": {
            "num_regimes": 3,
            "switch_interval": 50,
            "state_dim": 4,
            "obs_dim": 6
        },
        "results": results_eval,
        "window_defs": {
            "Pre": [-5, -1],
            "Shock": [1, 5],
            "Recover": [6, 20],
            "Steady": [21, 50]
        }
    }

    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved comprehensive results to {results_path}", flush=True)


if __name__ == "__main__":
    main()

