import os

import numpy as np
import torch
import torch.nn.functional as F

from methods.diagnostics.base import Diagnostic, DiagnosticInfo, LogType, Summary
from methods.diagnostics.standard import ForwardPass, PerSampleLossError

class _PerSampleProgress(Diagnostic):
    def __init__(self, manager, loader_key, label_source):
        self.loader_key = loader_key
        self.label_source = label_source
        self.forward_pass = manager.build(ForwardPass, manager, loader_key)
        super().__init__(manager)
    
    def progress_projection(self, log_probs, targets):
        N = self.method.num_classes
        uncertainty_dist = torch.full((N,), 1/N) # (1/N, ..., 1/N)
        true_dists = F.one_hot(targets, N)
        pred_dists = torch.exp(log_probs)

        # Shift probability simplex to origin
        def shift(probs):
            return probs - uncertainty_dist
        def shift_inv(probs):
            return probs + uncertainty_dist

        # Get probabilites' projections along "truth" line
        shifted_true = shift(true_dists)
        progress = torch.sum(shift(pred_dists) * shifted_true, axis=1) / torch.sum(shifted_true ** 2, axis=1)

        projected_dists = shift_inv(progress[:, None] * shifted_true)

        return projected_dists, progress

    def _run(self):
        fp = self.forward_pass.run().info
        log_probs, predictions, targets = fp["log_probs"], fp["predictions"], fp["targets"]

        # Calculate "new" progress per-sample
        projected_dists, progress = self.progress_projection(log_probs, targets)
        return DiagnosticInfo("per sample progress", {
            "projected distributions": projected_dists,
            "progress": progress
        })

    def __eq__(self, other):
        return (
            isinstance(other, _PerSampleProgress)
            and self.loader_key == other.loader_key
            and self.label_source == other.label_source
        )

class PerSampleProgressSummary(Summary):
    dependency_cls = _PerSampleProgress
    info_key = "progress"

class _PerSampleVolatility(Diagnostic):
    def __init__(self, manager, loader_key, label_source):
        self.loader_key = loader_key
        self.label_source = label_source
        self.forward_pass = manager.build(ForwardPass, manager, loader_key)
        self.last_log_probs = None
        super().__init__(manager)
    
    def _run(self):
        fp = self.forward_pass.run().info
        log_probs = fp["log_probs"]

        name = "per sample prediction volatility"
        # Cannot compute volatility on the first time we see predictions
        if self.last_log_probs is None:
            self.last_log_probs = log_probs
            return DiagnosticInfo(name, None)
        
        kl_divs = F.kl_div(log_probs, self.last_log_probs, reduction="none", log_target=True)
        
        return DiagnosticInfo(name, kl_divs)
    
    def __eq__(self, other):
        return isinstance(other, _PerSampleVolatility)
    
class PerSampleVolatilitySummary(Summary):
    dependency_cls = _PerSampleVolatility
