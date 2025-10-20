"""
Test-Time Adaptation Module for SSM-MetaRL-TestCompute
This module implements online adaptation mechanisms during inference for state-space models (SSMs)
and Meta-Reinforcement Learning (Meta-RL) agents. It provides:
- Online weight update hooks with safe, bounded parameter updates
- Meta-feature monitoring utilities (e.g., distribution shift indicators)
- Uncertainty estimation (entropy, variance, epistemic proxies)
- Adaptive gradient steps with stability guards (grad clipping, EMA, trust region)
- Integration glue to SSM core, Meta-RL policy/value nets, and environment wrappers
- Modular interface for extending adaptation strategies via a registry pattern
References (non-exhaustive):
- Test-Time Training: S. Sun et al., ICML 2020
- Tent: Test-time entropy minimization, D. Wang et al., ICLR 2021
- T3A: Test-time Template Adjustments, I. Nado et al., NeurIPS 2020 Workshop
- Meta-Learning: Finn et al., Model-Agnostic Meta-Learning (MAML), ICML 2017
- SSMs: Gu et al., Low-Rank State-Space Models, NeurIPS 2021; Smith et al., HiPPO, NeurIPS 2020
- RL Uncertainty: Kendall & Gal, What Uncertainties Do We Need in Bayesian Deep Learning for CV?, NIPS 2017
Usage Overview:
- Construct an Adapter with a target module and configuration
- Call adapter.observe(x, y_pred, y_true, meta) to log meta-features
- Call adapter.adapt(loss_fn, batch) during inference steps to update weights
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
# ------------------------------
# Configs and registries
# ------------------------------
@dataclass
class AdaptationConfig:
    enabled: bool = True
    lr: float = 1e-4
    max_step_scale: float = 0.1  # fraction of parameter norm per step
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    trust_region_eps: float = 1e-3
    ema_decay: float = 0.0  # 0 disables EMA
    entropy_weight: float = 0.0  # for Tent-like strategies
    max_steps_per_call: int = 1
    allow_layers: Optional[List[str]] = None  # regex or name substrings
    deny_layers: Optional[List[str]] = None
    freeze_batchnorm_stats: bool = True
    seed: Optional[int] = None
_STRATEGY_REGISTRY: Dict[str, Callable[["Adapter"], "AdaptationStrategy"]] = {}
def register_strategy(name: str):
    def deco(fn: Callable[["Adapter"], "AdaptationStrategy"]):
        _STRATEGY_REGISTRY[name] = fn
        return fn
    return deco
# ------------------------------
# Meta-feature monitoring and uncertainty
# ------------------------------
@dataclass
class MetaFeatures:
    timestamp: float
    batch_size: int
    entropy: Optional[float] = None
    mean_confidence: Optional[float] = None
    prediction_var: Optional[float] = None
    feature_shift: Optional[float] = None
    notes: Dict[str, Any] = field(default_factory=dict)
def predictive_entropy(logits: "torch.Tensor") -> "torch.Tensor":
    probs = logits.softmax(dim=-1)
    ent = -(probs * (probs.clamp(min=1e-12).log())).sum(dim=-1)
    return ent
def confidence(probs: "torch.Tensor") -> "torch.Tensor":
    return probs.max(dim=-1).values
def batch_variance(x: "torch.Tensor") -> "torch.Tensor":
    return x.float().var(dim=0).mean()
class ShiftTracker:
    def __init__(self, momentum: float = 0.99):
        self.momentum = momentum
        self._ref_mean: Optional["torch.Tensor"] = None
        self._ref_var: Optional["torch.Tensor"] = None
    def update(self, feats: "torch.Tensor") -> float:
        with torch.no_grad():
            cur_mean = feats.mean(dim=0)
            cur_var = feats.var(dim=0, unbiased=False)
            if self._ref_mean is None:
                self._ref_mean = cur_mean
                self._ref_var = cur_var
                return 0.0
            # population update
            self._ref_mean = self.momentum * self._ref_mean + (1 - self.momentum) * cur_mean
            self._ref_var = self.momentum * self._ref_var + (1 - self.momentum) * cur_var
            # standardized shift magnitude
            delta = (cur_mean - self._ref_mean).pow(2) / (self._ref_var + 1e-8)
            return float(delta.mean().sqrt().item())
# ------------------------------
# Strategies
# ------------------------------
class AdaptationStrategy:
    def __init__(self, adapter: "Adapter"):
        self.adapter = adapter
    def pre_forward(self, *args, **kwargs):
        pass
    def post_loss(self, loss: "torch.Tensor", outputs: Any, batch: Mapping[str, Any]):
        return loss
    def parameters(self) -> List["nn.Parameter"]:
        return [p for p in self.adapter.target.parameters() if p.requires_grad]
@register_strategy("tent")
class TentStrategy(AdaptationStrategy):
    """Entropy minimization at test time (Wang et al., 2021).
    Assumes classification logits in outputs["logits"].
    """
    def post_loss(self, loss: "torch.Tensor", outputs: Any, batch: Mapping[str, Any]):
        if not self.adapter.cfg.entropy_weight or "logits" not in outputs:
            return loss
        logits = outputs["logits"]
        ent = predictive_entropy(logits).mean()
        return loss + self.adapter.cfg.entropy_weight * ent
@register_strategy("none")
class NoOpStrategy(AdaptationStrategy):
    pass
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
    def adapt(self, loss_fn: Callable[[Any, Mapping[str, Any]], "torch.Tensor"], batch: Mapping[str, Any], fwd_fn: Optional[Callable[[Mapping[str, Any]], Any]] = None) -> Dict[str, Any]:
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
        # grad clip by norm
        if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.strategy.parameters(), self.cfg.grad_clip_norm)
        # bound max parameter step
        if self.cfg.max_step_scale and self.cfg.max_step_scale > 0:
            for p in self.strategy.parameters():
                if p.grad is None:
                    continue
                step_max = self.cfg.max_step_scale * (p.data.norm().item() + 1e-12)
                gnorm = p.grad.data.norm().item() + 1e-12
                scale = min(1.0, step_max / (self.cfg.lr * gnorm)) if self.cfg.lr > 0 else 1.0
                p.grad.data.mul_((scale))
    def _trust_region(self):
        # simple trust region using parameter EMA as anchor
        if self.cfg.trust_region_eps <= 0:
            return
        if self._ema is None:
            return
        with torch.no_grad():
            eps = self.cfg.trust_region_eps
            for name, p in self.target.named_parameters():
                if not p.requires_grad:
                    continue
                ref = self._ema.get(name)
                if ref is None:
                    continue
                delta = p.data - ref
                nrm = delta.norm().item()
                if nrm > eps:
                    p.data.copy_(ref + delta * (eps / (nrm + 1e-12)))
    def _ema_update(self):
        d = self.cfg.ema_decay
        if d <= 0:
            return
        if self._ema is None:
            self._ema = {}
        with torch.no_grad():
            for name, p in self.target.named_parameters():
                if not p.requires_grad:
                    continue
                ref = self._ema.get(name)
                if ref is None:
                    self._ema[name] = p.data.detach().clone()
                else:
                    self._ema[name].mul_(d).add_(p.data, alpha=1 - d)
# ------------------------------
# Integration hooks
# ------------------------------
class SSMHook:
    def __init__(self, adapter: Adapter):
        self.adapter = adapter
        self.extractor = None
    def attach(self, ssm_model: "nn.Module"):
        # Example: expose internal states or features for shift tracking
        if hasattr(ssm_model, "extract_features"):
            self.extractor = ssm_model.extract_features
        else:
            self.extractor = None
        return self
    def features(self, batch: Mapping[str, Any]) -> Optional["torch.Tensor"]:
        if self.extractor is None:
            return None
        with torch.no_grad():
            try:
                return self.extractor(batch)
            except Exception:
                return None
class MetaRLHook:
    def __init__(self, adapter: Adapter):
        self.adapter = adapter
    def policy_loss(self, outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> "torch.Tensor":
        # Placeholder: integrate with policy/value objectives
        # Expect outputs to include: logits for policy, value for baseline, etc.
        logits = outputs.get("logits")
        actions = batch.get("actions")
        advantages = batch.get("advantages")
        if logits is None or actions is None or advantages is None:
            raise ValueError("Missing keys for policy loss: logits/actions/advantages")
        logp = F.log_softmax(logits, dim=-1)
        picked = logp.gather(dim=-1, index=actions.long().unsqueeze(-1)).squeeze(-1)
        return -(picked * advantages).mean()
# ------------------------------
# Public API for extending strategies
# ------------------------------
def create_adapter(target: "nn.Module", cfg: Optional[AdaptationConfig] = None, strategy: str = "none") -> Adapter:
    return Adapter(target, cfg, strategy)

def register_adaptation_strategy(name: str, builder: Callable[[Adapter], AdaptationStrategy]):
    _STRATEGY_REGISTRY[name] = builder
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
        info = adapter.adapt(loss_fn, batch, fwd_fn=fwd)
        print(mf)
"""
