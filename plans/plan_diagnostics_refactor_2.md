Analysis of replacing DiagnosticsLogger with the DiagnosticsManager/Diagnostic design.

## What the current system does

`DiagnosticsLogger.log_diagnostics()` is called after every batch and at run start. It:
- Decides whether this step is a logging checkpoint (logarithmic or per-epoch schedule)
- Calls `model.eval()`, runs all sub-diagnostics (snapshots, probes, NTK, param/grad norms, weight matrix norms), logs to W&B, restores `model.train()`
- Returns `best_acc`, `best_epoch`, `is_best` back to `SelectionMethod` so it can trigger checkpointing

`SelectionMethod.__init__` passes a rich `DiagnosticsRunContext` (frozen dataclass: save_dir, fixed_train_loader, test_loader, artifact_stem, seed, checkpoint_saver, etc.) to the logger and all sub-diagnostics.

---

## Problems with the proposed design

### 1. `TrainState` is too thin

`TrainState` only has `epoch`, `batch_idx`, `total_epochs`, `total_batches`. But sub-diagnostics need the model, device, current lr, total_time, time_this_epoch, checkpoint_state for saving, selected_indexes for noisy-point tracking, and all the stuff currently in `DiagnosticsRunContext` (data loaders, save paths, etc.). None of this is specified — presumably it all moves into `shared_context`, but then the typed safety of `DiagnosticsRunContext` is lost. `total_step` is also missing from `TrainState`, even though the logarithmic schedule and W&B step alignment both depend on it.

[[Does it need _all_ the context in DiagnosticsRunContext? I want my individual Diagnostics objects and their associated managers to manage a minimal context. Definitely add `total_step` to my TrainState class though. What would the minimal shared context look like?]]

### 2. `should_run` on `Diagnostic` has no arguments

It is declared as `lambda: True` — a zero-argument callable. But deciding whether to log requires knowing the current `total_step` and `batch_idx` (to implement the logarithmic schedule). If it "uses manager's state," `TrainState` would need to be readable from within the lambda, which means either closing over the manager or adding `total_step` to `TrainState`.

### 3. The cache check in `run()` is broken as written

> "If `current_state` is `last_run_state`, return `last_run_state`"

Two issues: (a) `is` on a dataclass tests identity, not equality — a freshly constructed `TrainState` with the same values won't match. (b) The return value should be `last_run_diagnostic` (a `DiagnosticInfo`), not `last_run_state` (a `TrainState`). The caching semantics need clarification.

### 4. `DiagnosticsManager.should_run` is a plain `bool`

`Diagnostic.should_run` is a callable. `DiagnosticsManager.should_run` is a bare `bool`. It's unclear what gates it or who sets it, and whether it's meant to globally suppress all diagnostics.

### 5. Dependency ordering is declared but not enforced

`Diagnostic.dependencies: List[Diagnostic]` exists, but `run_diagnostics` has no described mechanism to topologically sort or otherwise respect dependencies before running. Diagnostics that depend on others (e.g., a logit-norm diagnostic that depends on a snapshot already being computed) could run in the wrong order.

### 6. `update_shared_context` defaulting to `vars()`

`vars()` called inside a method returns the method's local variables, not anything meaningful. This seems like a placeholder but would silently produce wrong behavior if left as-is. The intended default is probably `lambda self: vars(self)` (the manager's instance dict) or just `{}`.

### 7. `fetch_context()` raises on duplicate keys

Raising an exception on overlap is strict. Currently `DiagnosticsRunContext` is a frozen typed dataclass — no overlap possible. With two dicts that both might have common keys (e.g., `model`, `device`), this will require careful discipline to avoid crashes, especially as new diagnostics are added.

### 8. Best-acc / checkpointing flow is unspecified

`SelectionMethod.compute_diagnostics` reads `best_acc`, `best_epoch`, `is_best` from the return value of `log_diagnostics` and uses them to trigger `save_model`. The new design returns `DiagnosticInfo(name, info: Any)` — it's not clear how `SelectionMethod` gets those scalar values back, or which `Diagnostic` is responsible for producing them.

### 9. `model.eval()` / `model.train()` state management

Currently there is exactly one `model.eval()` call per logging step, and training state is restored afterward. If each `Diagnostic` independently switches eval/train mode, you either do it redundantly on every diagnostic or push this responsibility to `DiagnosticsManager.run_diagnostics` — but that's not described in the plan.

### 10. W&B step alignment

All current W&B logging goes through a single `logger.wandb_log(log_data, step=int(total_step))` call, so all metrics for one checkpoint land on the same W&B step. If individual `Diagnostic` objects call `wandb_log` independently, they each need to know `total_step` and use it consistently, or W&B will scatter metrics across steps.

---

## Summary

The main structural gap is that the interface between `DiagnosticsManager` and `SelectionMethod` is underspecified: how does the manager receive the model, device, lr, and checkpoint state it needs? How do best-acc updates flow back? How is `total_step` threaded through? Resolving these — likely by expanding `TrainState` and formalizing what goes in `shared_context` vs `context` — would address most of the above issues.
