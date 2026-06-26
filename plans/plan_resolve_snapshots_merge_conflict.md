# Plan: Resolve unresolved merge conflict in snapshots.py

## Problem

`methods/method_utils/snapshots.py` has two unresolved git merge conflict
blocks (lines 249-252 and 281-308), left over from merging `main` into
`HEAD`. This causes a `SyntaxError` on import, crashing every run (see
`logs/12408242.err`).

## Analysis

- Conflict 1 (lines 249-252): `HEAD` passes no extra kwarg to
  `_calculate_snapshot_stats`; `main` adds `override_labels=self.true_labels`.
- Conflict 2 (lines 281-308): `HEAD` rebuilds `metrics` using old variable
  names (`f`, `e`, `fv`, `ev`, `yh`, `yvh`) that no longer exist in this
  function's scope, and calls
  `self._calculate_snapshot_stats(..., true_labels=self.true_labels)` — but
  `_calculate_snapshot_stats` (line 142) only accepts an `override_labels`
  kwarg, not `true_labels`. This HEAD code would raise `TypeError` if it ever
  executed. `main`'s side of this conflict is empty — i.e. main already
  deleted this block, because the equivalent functionality (noisy/true-label
  loss and accuracy) is now produced earlier via `override_labels` and the
  `train_losses_noisy`/`train_errors_noisy` outputs folded into `metrics` at
  lines 264-266.

**Resolution: keep `main`'s side for both conflicts, discard `HEAD`'s side.**

## Changes

### `methods/method_utils/snapshots.py`

Conflict 1 — replace:
```python
            train_log_probs, train_logits_l2_norms, train_losses, train_errors, train_eval_labels, train_losses_noisy, train_errors_noisy = self._calculate_snapshot_stats(
                model,
                self.fixed_train_loader,
                device,
<<<<<<< HEAD
=======
                override_labels=self.true_labels,
>>>>>>> main
            )
```
with:
```python
            train_log_probs, train_logits_l2_norms, train_losses, train_errors, train_eval_labels, train_losses_noisy, train_errors_noisy = self._calculate_snapshot_stats(
                model,
                self.fixed_train_loader,
                device,
                override_labels=self.true_labels,
            )
```

Conflict 2 — replace:
```python
<<<<<<< HEAD
        metrics = {
            'train_loss': float(f.mean().item()),
            'train_acc': float(1.0 - e.mean().item()),
            'val_loss': float(fv.mean().item()),
            'val_acc': float(1.0 - ev.mean().item()),
            'train_normed_logits_l2_mean': float(torch.norm(yh, p=2, dim=1).mean().item()),
            'val_normed_logits_l2_mean': float(torch.norm(yvh, p=2, dim=1).mean().item()),
        }
        if self.true_labels is not None:
            yht, ft, et = self._calculate_snapshot_stats(
                model,
                self.fixed_train_loader,
                device,
                true_labels=self.true_labels,
            )
            snapshot.update({
                'yht': yht,
                'ft': ft,
                'et': et,
            })
            metrics.update({
                'train_loss_true_labels': float(ft.mean().item()),
                'train_acc_true_labels': float(1.0 - et.mean().item()),
            })

=======
>>>>>>> main
        return snapshot, metrics
```
with:
```python
        return snapshot, metrics
```
(i.e. delete the entire conflict block, leaving just `return snapshot, metrics`)

### `methods/method_utils/diagnostics.py` — fix stale reference to deleted HEAD-side API

`log_diagnostics` (around line 240-248) still assumes the *old* semantics
from the deleted HEAD block: that `metrics['train_loss']`/`['train_acc']`
are computed against loader (possibly noisy) labels, with a separate
`train_*_true_labels` pair as a bonus when true labels are available. But
in main's refactor (now the only code path, per conflict 1 above),
`build_snapshot` calls `_calculate_snapshot_stats(..., override_labels=self.true_labels)`
for the train loader. Inside `_calculate_snapshot_stats`, when
`override_labels` is set, the *primary* loss/error is evaluated against
`override_labels` (i.e. true labels) and the *secondary* "noisy" loss/error
is evaluated against the loader's own `targets`. So the semantics are now
inverted from what `diagnostics.py` assumes:
- `metrics['train_loss']` / `['train_acc']` → now true-label accuracy (when `self.true_labels` is set)
- `metrics['train_loss_noisy_labels']` / `['train_acc_noisy_labels']` → loader-label accuracy

This is also why `uses_true_labels_for_train_accuracy()` doesn't exist on
`SnapshotManager` (it was HEAD-side API, never carried over) and why
`train_acc_true_labels`/`train_loss_true_labels` keys don't exist in
`snapshot_metrics` (they were only produced by the dead HEAD block deleted
in conflict 2 above). `SnapshotManager.has_label_noise()` (`return self.true_labels is not None`)
is the equivalent boolean check under main's naming.

Replace:
```python
                if self.snapshot_manager.uses_true_labels_for_train_accuracy():
                    log_data['train_acc_true_labels'] = snapshot_metrics['train_acc_true_labels']
                    log_data['train_loss_true_labels'] = snapshot_metrics['train_loss_true_labels']
                    log_data['train_acc_loader_labels'] = snapshot_metrics['train_acc']
                    log_data['train_loss_loader_labels'] = snapshot_metrics['train_loss']
```
with:
```python
                if self.snapshot_manager.has_label_noise():
                    log_data['train_acc_true_labels'] = snapshot_metrics['train_acc']
                    log_data['train_loss_true_labels'] = snapshot_metrics['train_loss']
                    log_data['train_acc_loader_labels'] = snapshot_metrics['train_acc_noisy_labels']
                    log_data['train_loss_loader_labels'] = snapshot_metrics['train_loss_noisy_labels']
```
This preserves the wandb log key names (`train_acc_true_labels`, etc.) so
existing dashboards keep working, while sourcing values from the metric
keys that actually exist under main's refactor.

## Notes

- No other files in the repo contain unresolved conflict markers (checked
  via `grep -rn -E "^(<<<<<<<|=======|>>>>>>>)"` across `.py`/`.yaml`/`.yml`).
- Conflicts 1 and 2 are pure conflict-resolution cleanup with no behavior
  change relative to `main`'s intended code. The `diagnostics.py` change is
  a follow-on fix required because that file (no conflict markers, already
  "resolved" by someone) was left referencing the old HEAD-side API/semantics
  that conflict resolution removes.

## Summary of changes

- [ ] Resolve conflict 1 in `methods/method_utils/snapshots.py` (keep `main`'s `override_labels` kwarg)
- [ ] Resolve conflict 2 in `methods/method_utils/snapshots.py` (delete dead `HEAD` block, keep `main`'s empty side)
- [ ] Fix `methods/method_utils/diagnostics.py` lines ~244-248 to use `has_label_noise()` and the correct (swapped) metric keys
