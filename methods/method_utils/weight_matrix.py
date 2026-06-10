import torch

from methods.method_utils.diagnostics_context import DiagnosticsRunContext


class WeightMatrixDiagnostics:
    def __init__(
        self,
        logger,
        context: DiagnosticsRunContext,
        enabled: bool = False,
    ):
        self.logger = logger
        self.context = context
        self.enabled = enabled

    def _get_weight_info(self, name, p):
        if len(p.shape) != 2:
            return None

        frobenius = torch.linalg.norm(p, ord='fro').detach().cpu().item()
        spectral = torch.linalg.matrix_norm(p, ord=2).detach().cpu().item()
        alignment = spectral / frobenius

        return {'frobenius': frobenius, 'spectral': spectral, 'alignment': alignment}

    def log_metrics(self, model, param_names=None):
        # param_names is intentionally not config-driven; kept as an optional kwarg for future use
        if not self.enabled:
            return {}

        if param_names is None:
            params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        else:
            params = [
                (n, p) for n, p in model.named_parameters()
                if p.requires_grad and n in param_names
            ]

        log_data = {}
        for name, p in params:
            info = self._get_weight_info(name, p)
            if info is None:
                continue
            for metric, value in info.items():
                log_data[f'diagnostics/weight_norms/{name}/{metric}'] = value

        return log_data
