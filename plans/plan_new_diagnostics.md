I want a new Diagnostics system in this repository. As it is, in @methods/SelectionMethod.py, `compute_diagnostics` gets called at the end of each batch. That's the hook into the existing diagnostics system, the system that I want to change. The `log_diagnostics` method of the `DiagnosticsLogger` checks if logging should even happen, and then does a ton of logging.

I want to change how it works. I envision the following classes

A `TrainingState` dataclass object that holds the current `epoch`, `batch_idx`, and any other training time data necessary for the diagnostics to decide when to diagnose, like the total number of epochs and batches.

A `Diagnostic` class, which is partially abstract (not all methods are implemented). 
- Each child class should have minimal attributes. For example, a `DiagnoseValAcc` object would only need the current model and the validation set.
- Each `Diagnostic` child object has a `run()` method, which runs its respective diagnostic and returns results as a dict. It logs the current training state as `self.last_diagnosed_state`.
- Each child also has a `conditional_run()` method, which runs `run()` if the current state is not `self.last_diagnosed_state` and `self.should_run()` returns True.
- It should have functionality to determine the frequency at which it runs. A `DiagnoseValAcc` object might only run once per epoch, for example. 
    - Do not assume that `run()` will get called regularly—for example, if a diagnostic is supposed to run every epoch, and it gets called after 10 epochs of getting run, it should run.
- The `conditional_run` method will take in a `TrainingState` object, which `self.should_run()` will process and use to determine if `run` should run.

I want a `DiagnosticsManager` object with the following properties:
- Attribute `self.diagnostics`: a list of `Diagnostic` objects
- An `update_state()` method, which updates the current `TrainingState` as `self.training_state`.
- A `run()` method, which uses the interally stored training states and runs all the diagnostics using `diagnostic.conditional_run()`. It toggles the model to `eval()` before running diagnostics and restores its prior train/eval mode afterward, centralized here rather than in each `Diagnostic`.
- An `append_diagnostic()` method, which appends a new diagnostic or list of diagnostics to `self.diagnostics`
- `DiagnosticsManager` owns all `wandb_log` calls exclusively: `run()` collects the dicts returned by each `diagnostic.conditional_run()`, concatenates them, and issues a single `wandb_log(..., step=...)` call. Individual `Diagnostic`s never call wandb directly — this keeps the `step` argument monotonically increasing, since `DiagnosticsManager.run()` is the single call site invoked in increasing step order.

[[What critiques do you have with this plan so far? What will go catastrophically wrong?]]

{{1. The biggest risk is losing shared computation. Right now `SnapshotManager.build_snapshot` does ONE forward pass over train/val to produce train_loss, train_acc, val_loss, val_acc, and normed-logits norms together (diagnostics.py:222-260). If `DiagnoseValAcc`, `DiagnoseValLoss`, etc. become separate `Diagnostic` objects each holding "the model and the validation set" and each doing its own `run()`, you'll multiply expensive forward passes by however many diagnostics need them (e.g. 4x val-set passes instead of 1x). Same problem applies to NTK and linear-probe diagnostics, which are already separated out and expensive on their own. You need some notion of a shared per-step cache/context that multiple `Diagnostic`s can pull cached forward-pass results from, or this redesign will be slower than what exists today despite being cleaner.}}

{{2. Checkpointing is currently entangled with logging. `SelectionMethod.compute_diagnostics` calls `self.save_model(epoch)` whenever `diagnostics_state['logged']` is True, and `is_best`/`best_acc`/`best_epoch` come out of the same `log_diagnostics` call (SelectionMethod.py:265-288). If diagnostics become independent objects each deciding their own schedule, you need an explicit answer to "who decides when to checkpoint and what counts as best" — that can't just be "whatever diagnostic happened to run this step."}}
[[Each `Diagnostic.run()` method should return what it would log, and the `DiagnosticsManager` would concatenate them together for logging. Would that work?]]

{{2a. That solves critique 3 (logging aggregation / wandb step monotonicity) well. It does not by itself solve critique 2's actual question, which is checkpointing/`is_best`, not logging. `save_model` is called from `SelectionMethod.compute_diagnostics` based on `is_best`/`best_acc`/`best_epoch`, which today come out of `SnapshotManager` as stateful properties, not just logged values. If a generic dict union is the only channel out of `DiagnosticsManager.run()`, you need an explicit convention for how `is_best`/`best_acc`/`best_epoch` surface through that union (e.g. reserved keys the manager looks for, or a dedicated non-`Diagnostic` `BestModelTracker` the manager holds separately and exposes as queryable state after `run()`) — otherwise `SelectionMethod` has no way to know whether to checkpoint.}}
[[Good point. I would like the checkpointer to also be a diagnostics object. Diagnostic objects should be able to have dependencies.  ]]

{{3. wandb requires a monotonically non-decreasing `step` argument across calls. Today all diagnostics for a given step get merged into one `log_data` dict and logged in a single `wandb_log(..., step=total_step)` call (diagnostics.py:277). If each `Diagnostic.run()` logs independently, you risk multiple `wandb.log` calls per step (likely fine) but also risk out-of-order steps if diagnostics with different schedules end up running/logging in an order that isn't monotonic in `total_step`. The `DiagnosticsManager` should probably own log buffering/flushing centrally rather than letting each `Diagnostic` call wandb directly.}}
[[see comments below your note 2]]

{{3a. Same answer as note 26 below — yes, this solves the monotonicity concern, but only if `DiagnosticsManager.run()` is the *single* call site that ever calls `wandb_log`, and it's invoked in strictly increasing `total_step` order (matching the current single call site in `SelectionMethod.after_batch`). Worth stating explicitly in the plan body: `DiagnosticsManager` owns all `wandb_log` calls; individual `Diagnostic`s never call wandb directly, they only return dicts.}}
[[Exactly. Add this to the plan.]]

{{4. Model train/eval mode toggling is centralized today (`model.eval()` at the top of `log_diagnostics`, restored at the bottom). If every `Diagnostic.run()` does this independently, you risk redundant toggling or — worse — a diagnostic that forgets to toggle and silently picks up dropout/BatchNorm train-mode artifacts. This should stay centralized in `DiagnosticsManager.run()`, wrapping the whole batch of `conditional_run()` calls once.}}
[[Facts. Add this to the plan.]]

{{5. The "catch up if skipped" requirement (item 10 in your plan) is easy for simple periodic schedules (every epoch, every N epochs) but harder for the existing logarithmic schedule and for NTK's specific checkpoint list (`ntk_eigenvalue_concentration_checkpoints = [20, 40, 80]`, i.e. specific epochs, not a period). `should_run` needs to support at least three schedule kinds — periodic, logarithmic, and explicit checkpoint lists — and "catch up" logic needs to mean "did I cross a scheduled point since I was last asked," not just "is the current epoch/batch divisible by something."}}
[[correct]]

{{5a. Good — no plan edit needed beyond what's already there (item on "functionality to determine the frequency"). Just flagging for implementation time: `should_run` will need at least three schedule strategies (periodic, logarithmic, explicit checkpoint list), so it's probably its own small strategy object/enum rather than a single frequency int.}}

{{6. Idempotency: the existing `mark_logged`/`_logged_steps` set exists specifically because `compute_diagnostics` gets called both after every batch and after every epoch, and the same `total_step` could otherwise get logged twice (diagnostics.py:178-181). Each `Diagnostic` needs its own "have I already run for this step" guard, or the `DiagnosticsManager` needs to dedupe centrally before dispatching to `conditional_run()`.}}
[[See changes in plan above. Would that work?]]

{{6a. Mostly yes. Comparing the current `TrainingState` to `self.last_diagnosed_state` stops the literal same state from re-triggering `run()` twice, covering the case `mark_logged` exists for today. One subtlety to pin down: this only works if batch-end and epoch-end calls produce *equal* `TrainingState` values for what's conceptually "the same step." E.g. if the epoch-end trigger constructs a `TrainingState` with `batch_idx=-1` (sentinel) while the batch-end call for the last batch of that epoch used `batch_idx=total_batches-1`, the two states would be unequal and an epoch-frequency diagnostic could run twice for what's really one step. Worth specifying in the plan what `TrainingState` looks like at an epoch-end trigger, and whether it's guaranteed equal to the last in-epoch `TrainingState`.}}

{{7. Minor: `should_run`/`conditional_run` taking epoch and batch number separately means each `Diagnostic` (or the manager) needs to know `total_batches` to reconstruct a global step count for logarithmic-style schedules — worth deciding whether the manager passes `total_step` directly instead of epoch+batch, to avoid duplicating that arithmetic in every diagnostic.}}
[[See changes in plan above. Would that work?]]

{{7a. Yes — `TrainingState` carrying `epoch`, `batch_idx`, and the totals (epochs, batches) alongside them means a global step count can be derived in one place rather than duplicated per `Diagnostic`. Suggest making `total_step` a computed property on `TrainingState` itself (e.g. `epoch * total_batches + batch_idx`), so there's exactly one implementation of that arithmetic and every `Diagnostic.should_run()` just reads `state.total_step`.}}