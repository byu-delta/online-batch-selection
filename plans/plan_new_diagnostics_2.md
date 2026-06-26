New Diagnostics system.

dataclass `TrainState`
- `epoch: int`
- `batch_idx: int`
- `total_epochs: int`
- `total_batches: int`
- `total_steps: int`


dataclass `DiagnosticInfo`
- `name: str`
- `info: Any`

class `Diagnostic` (partially abstract)
- `manager: DiagnosticsManager` Given in constructor. Automatically adds diagnostic to manager with `manager.add_diagnostic(self)`.
- `log_path` Given in constructor. Defaults to None, in which case `file_log` does nothing.
- `should_run -> bool` Given in constructor, defaults to `lambda: True`. Internally would normally use manager's state through `get_state` when overridden
- `last_run_state: TrainState`
- `last_run_diagnostic: DiagnosticInfo`
- `get_context()` Gets managers' context as a dict
- `get_state() -> TrainState`
- `_run() -> DiagnosticInfo` Abstract method. Arg-free. Runs the diagnostic. First runs dependencies, if present. 
- `run()` First checks if `get_state() == last_run_state` (using `==`, not `is`). If so, returns `last_run_diagnostic`. Otherwise, updates `last_run_state`, updates `last_run_diagnostic` with `_run(*args)`, and returns `last_run_diagnostic`
- `conditional_run()` Runs `should_run`. If `True`, runs `run()`
- `wandb_log(List[DiagnosticInfo])` logs to W&B
- `file_log(List[DiagnosticInfo])` logs to `log_path`
- `log()` calls `wandb_log([last_run_diagnostic])` and `file_log([last_run_diagnostic])`
- `__eq__`: raises `NotImplementedError` by default, forcing children to override it if there are to be multiple instances of it in create_diagnostics.py
- Children should have their dependencies (`Diagnostic`s) as attributes, like `self.diagnostic_dep`. At the beginning of each child's `self._run(...)`, it should `d_dep_result = diagnostic_dep.run(...)` for each diagnostic it depends on. [[Since `d_dep.run()` caches, this usually should not be expensive, right?]]

class `DiagnosticsManager`
- `diagnostics: List[Diagnostic]`
- `add_diagnostic(self, diagnostic: Diagnostic)` Appends to `self.diagnostics`
- `last_run_state: TrainState`
- `should_run: bool` Master kill-switch for all diagnostics in this manager
- `shared_context: dict`
- `update_state(state: TrainState)` Updates `current_state`
- `update_shared_context(**kwargs)` Receives kwargs and saves them into `shared_context`
- `run_diagnostics(state: TrainState)` Runs all the `diagnostics`
- There will be a DiagnosticsManager that manages diagnostics that run before training step, another for after training, another at validation time (after epoch), etc. Wherever best. Remember that each diagnostic will have a `should_run` method to determine how often it actually runs.


class `DiagnosticsBuilder`
- `all_diagnostics: defaultdict` Dictionary where each key is a diagnostic class name. Corresponding values are lists of `Diagnostic`s of that class. Default value is `[]`.
```python
def fetch_duplicate_diagnostic(self, diagnostic):
    # Note: Raises an exception if type(diagnostic) does not have __eq__ defined
    matches = [x for x in self.all_diagnostics[type(diagnostic)] if x == diagnostic] 
    if not matches:
        return None
    elif len(matches) > 1:
        raise ExceptionOfSomeKind
    else:
        return matches[0]
def create_diagnostic(diagnostic_class, *args, **kwargs):
    new_diagnostic = new__class(*args, **kwargs)
    duplicate = self.fetch_duplicate_diagnostic(new_diagnostic)

    if duplicate:
        return duplicate
    else:
        self.all_diagnostics[diagnostic_class].append(new_diagnostic)
        return new_diagnostic
```


file `create_diagnostics.py`
- Initializes a DiagnosticsBuilder
- Reads diagnostics config file
- Config should also specify how often to run diagnostics by default.
- Keys in diagnostics config file are diagnostics objects to create, values are the parameters of the Diagnostics object to set. Should also be possible to override how often to run a particular diagnostic [[what will go wrong here?]]
- Uses `DiagnosticsBuilder.create_diagnostic()` to create diagnostic objects and their associated managers using the config (if multiple diagnostics share a dependency with the same parameters, there will not be a duplicate because `create_diagnostics` checks for duplicates. [[this will work, right?]])
