"""
Test-Time Adaptation Module for SSM-MetaRL-TestCompute
... (docstring comments) ...
Usage Overview:
- Construct an Adapter with a target module and configuration
- Call adapter.observe(x, y_pred, y_true, meta) to log meta-features
- Call adapter.update_step(loss_fn, batch) during inference steps to update weights
- Use adapter.context(model) as a context manager to temporarily enable adaptation
Note: Designed for PyTorch; adaptation is opt-in and safe by default.
"""
from __future__ import annotations
import contextlib
import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Any
import math
import time
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - allow import when torch unavailable
    torch = None
    nn = object  # type: ignore
    F = None
# ... (rest of the file from line 41 to 244 is identical) ...
# ------------------------------
# Core Adapter
# ------------------------------
class Adapter:
    def __init__(self, target: "nn.Module", cfg: Optional[AdaptationConfig] = None, strategy: str = "none"):
        if torch is None:
            raise RuntimeError("PyTorch is required for test-time adaptation")
        self.target = target
        self.cfg = cfg or AdaptationConfig()
        self.strategy_name = strategy
        self.strategy = _STRATEGY_REGISTRY.get(strategy, _STRATEGY_REGISTRY["none"])(self)
        self.shift = ShiftTracker()
        self.meta_log: List[MetaFeatures] = []
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
        self._opt: Optional[torch.optim.Optimizer] = None
        self._ema: Optional[Dict[str, torch.Tensor]] = None
        self._frozen_bn: List[Tuple[nn.Module, torch.Tensor, torch.Tensor]] = []
        self._select_params()
    def _select_params(self):
        allow = self.cfg.allow_layers
        deny = self.cfg.deny_layers
        for name, p in self.target.named_parameters():
            req = True
            if allow is not None:
                req = any(tok in name for tok in allow)
            if deny is not None and any(tok in name for tok in deny):
                req = False
            p.requires_grad = req
    def _ensure_opt(self):
        if self._opt is None:
            params = [p for p in self.target.parameters() if p.requires_grad]
            self._opt = torch.optim.Adam(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
    @contextlib.contextmanager
    def context(self, model: Optional["nn.Module"] = None) -> Iterator["Adapter"]:
        # Freeze BN stats optionally
        if self.cfg.freeze_batchnorm_stats:
            self._freeze_bn()
        try:
            yield self
        finally:
            if self.cfg.freeze_batchnorm_stats:
                self._unfreeze_bn()
    def _freeze_bn(self):
        self._frozen_bn.clear()
        for m in self.target.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                self._frozen_bn.append((m, m.running_mean.clone(), m.running_var.clone()))
                m.eval()
    def _unfreeze_bn(self):
        for m, mean, var in self._frozen_bn:
            m.running_mean.copy_(mean)
            m.running_var.copy_(var)
        self._frozen_bn.clear()
    def observe(self, outputs: Mapping[str, Any], batch: Mapping[str, Any], features: Optional["torch.Tensor"] = None):
        bs = 0
        if batch:
            try:
                first = next(iter(batch.values()))
                bs = len(first) if hasattr(first, "__len__") else 0
            except Exception:
                bs = 0
        mf = MetaFeatures(timestamp=time.time(), batch_size=bs)
        if outputs is not None:
            if isinstance(outputs, Mapping) and "logits" in outputs and torch.is_tensor(outputs["logits"]):
                logits = outputs["logits"]
                ent = predictive_entropy(logits).mean().item()
                mf.entropy = float(ent)
                mf.mean_confidence = float(confidence(logits.softmax(-1)).mean().item())
            if isinstance(outputs, Mapping) and "probs" in outputs and torch.is_tensor(outputs["probs"]):
                probs = outputs["probs"]
                mf.mean_confidence = float(confidence(probs).mean().item())
        if features is not None and torch.is_tensor(features) and features.ndim >= 2:
            mf.feature_shift = self.shift.update(features)
            mf.prediction_var = float(batch_variance(features).item())
        self.meta_log.append(mf)
        return mf
    
    def update_step(self, loss_fn: Callable[[Any, Mapping[str, Any]], "torch.Tensor"], batch: Mapping[str, Any], fwd_fn: Optional[Callable[[Mapping[str, Any]], Any]] = None) -> Dict[str, Any]:
        """
        Performs a single test-time adaptation update step.
        """
        if not self.cfg.enabled:
            return {"updated": False, "reason": "disabled", "loss": None}
        self._ensure_opt()
        self.target.train()
        updated = False
        info: Dict[str, Any] = {}
        steps = max(1, int(self.cfg.max_steps_per_call))
        current_loss: Optional[float] = None
        for _ in range(steps):
            self._opt.zero_grad(set_to_none=True)
            outputs = fwd_fn(batch) if fwd_fn else self.target(**batch)
            loss = loss_fn(outputs, batch)
            loss = self.strategy.post_loss(loss, outputs, batch)
            loss.backward()
            # track last step loss.item()
            try:
                current_loss = float(loss.item())
            except Exception:
                current_loss = None
            self._stability_guards()
            self._opt.step()
            self._trust_region()
            self._ema_update()
            updated = True
        info.update({"updated": updated, "steps": steps, "loss": current_loss})
        return info
    
    def _stability_guards(self):
# ... (rest of the file from line 301 to 376 is identical) ...
# ------------------------------
# Example docstring usage
# ------------------------------
__doc__ += """
Architecture:
- Adapter controls parameter selection, optimizer, stability guards, EMA, and trust region.
- Strategy augments the loss with adaptation regularizers (e.g., entropy minimization).
- Hooks expose integration points with SSMs and Meta-RL components for feature collection and loss computation.
Example:
    adapter = create_adapter(model, AdaptationConfig(lr=5e-5, entropy_weight=0.01), strategy="tent")
    ssm_hook = SSMHook(adapter).attach(model)
    def fwd(batch):
        return model(**batch)
    def loss_fn(outputs, batch):
        # cross-entropy with optional Tent regularization
        y = batch["labels"]
        logits = outputs["logits"]
        return F.cross_entropy(logits, y)
    with adapter.context(model):
        outputs = fwd(batch)
        feats = ssm_hook.features(batch)
        mf = adapter.observe(outputs, batch, feats)
        info = adapter.update_step(loss_fn, batch, fwd_fn=fwd)
        print(mf)
"""
