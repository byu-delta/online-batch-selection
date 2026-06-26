New Diagnostics system.

---

## `TrainState` (dataclass)

- `epoch: int`
- `batch_idx: int`
- `total_epochs: int`
- `total_batches: int`
- `total_steps: int`

---

## `DiagnosticInfo` (dataclass)

- `name: str`
- `info: Any`

---

## `Diagnostic` (partially abstract)

**Attributes**

- `manager: DiagnosticsManager` — given in constructor; auto-registers self with manager
- `log_path: str | None` — given in constructor; defaults to `None`; `file_log` is a no-op if unset
- `should_run: Callable -> bool` — given in constructor; defaults to `lambda: True`
- `last_run_state: TrainState | None` — initially `None`
- `last_run_diagnostic: DiagnosticInfo | None` — initially `None`

**Methods**

- `get_state() -> TrainState` — returns `manager.current_state`
- `get_context() -> dict` — returns `manager.shared_context`
- `_run() -> DiagnosticInfo` — **abstract.** Should take no args. Child implementations should call `dep.run()` for each dependency at the start, then compute and return a `DiagnosticInfo`. Note: `dep.run()` caches by state, so calling it here is cheap if the dep has already run this step.
- `run() -> DiagnosticInfo` — if `get_state() == last_run_state`, returns cached `last_run_diagnostic`; otherwise calls `_run(*args)`, updates `last_run_state` and `last_run_diagnostic`, and returns the result. Note: cache is keyed on state only — diagnostics should not use `*args` for values that vary independently of state; prefer passing such values through `shared_context`.
- `conditional_run()` — calls `should_run(get_state())`; if `True`, calls `run()`. To be used by DiagnosticsManager.
- `wandb_log(infos: List[DiagnosticInfo])` — logs `last_run_diagnostic` with necessary elements of `last_run_state` to W&B
- `file_log(infos: List[DiagnosticInfo])` — logs `last_run_diagnostic` with necessary elements of `last_run_state` to `log_path`
- `log()` — calls `wandb_log([last_run_diagnostic])` and `file_log([last_run_diagnostic])`.
- `__eq__` — raises `NotImplementedError` by default; children must override if they may be deduplicated in `create_diagnostics.py` (i.e., if multiple instances with the same parameters could be created)

---

## `DiagnosticsManager`

**Attributes**

- `diagnostics: List[Diagnostic]` — top-level diagnostics only; dependency diagnostics are not registered here
- `current_state: TrainState`
- `should_run: bool` — master kill-switch for all diagnostics in this manager
- `shared_context: dict`

**Methods**

- `_update_state(state: TrainState)` — sets `current_state`
- `_update_shared_context(**kwargs)` — merges kwargs into `shared_context`
- `run_diagnostics(state: TrainState, *, **kwargs)` — Runs `_update_state(state)`, runs `_update_shared_context(**kwargs)`, calls `conditional_run()` on each diagnostic in `diagnostics`. Then runs `_log_diagnostics()`.
- `_log_diagnostics()` Calls `log()` on each diagnostic in `diagnostics`.

*Multiple managers exist for different training phases (pre-batch, post-batch, validation, etc.).*

---

## `DiagnosticsBuilder`

**Attributes**

- `all_diagnostics: defaultdict(list)` — keyed by diagnostic class; values are lists of instances

**Methods**

```python
def fetch_duplicate_diagnostic(self, diagnostic) -> Diagnostic | None:
    # Raises NotImplementedError if type(diagnostic) has not overridden __eq__
    matches = [x for x in self.all_diagnostics[type(diagnostic)] if x == diagnostic]
    if not matches:
        return None
    elif len(matches) > 1:
        raise ValueError(f"Multiple identical diagnostics of type {type(diagnostic).__name__}")
    else:
        return matches[0]

def create_diagnostic(self, diagnostic_class, *args, **kwargs) -> Diagnostic:
    new_diagnostic = diagnostic_class(*args, **kwargs)
    duplicate = self.fetch_duplicate_diagnostic(new_diagnostic)
    if duplicate:
        return duplicate
    self.all_diagnostics[diagnostic_class].append(new_diagnostic)
    return new_diagnostic
```

---

## `create_diagnostics.py`

- Initializes a `DiagnosticsBuilder` and the appropriate `DiagnosticsManager`s
- Reads diagnostics config YAML; keys are Diagnostics class names available in the scope of this file
- Calls `DiagnosticsBuilder.create_diagnostic()` for each top-level diagnostic; shared dependencies constructed internally by those diagnostics are deduplicated automatically via `__eq__`
- Registers top-level diagnostics with the appropriate manager; dependency diagnostics are not added to any manager's `diagnostics` list

## Diagnostics config yaml
- Config specifies
    - which top-level diagnostics to create
    - their parameters
    - optionally a per-diagnostic `should_run` override; a global default frequency can also be set
Example:
```yaml
diagnostics:
    logging_defaults:
        log_interval: logarithmic
        save_init: 5
        save_freq: 4
    diagnostics:
        GradNorm: # this is made up
            logging: # Not necessary, just to override the defaults
                log_interval: ...
            params:
                norm: 2 # Parameters to be passed to GradNorm's constuctor as kwargs
                ...
        ...
```