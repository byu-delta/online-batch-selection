# Plan: Make Snapshot Location Follow `--exp_base` Like the Save Dir Does

## Context

`--exp_base` (default `./exp/`, e.g. `./exp-ablation/` in `slurm_run_blobs_deep_linear.py`) controls where experiment output dirs (`config.yaml`, checkpoints, etc.) land — that part is already configurable per-sweep.

Snapshots are not: `SnapshotManager.__init__` (`methods/method_utils/snapshots.py:65-67`) always writes to

```python
snapshots_dir = os.path.join(self.context.project_root, 'snapshots', self.context.dataset_name)
self.snapshots_path = os.path.join(snapshots_dir, f'{self.context.artifact_stem}.p')
```

`project_root` is hardcoded in `methods/SelectionMethod.py:93` as the repo root (`os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))`) — it has no knowledge of `exp_base` at all. So no matter what `--exp_base` is passed, snapshots always land in the same `snapshots/<dataset_name>/` folder. Two different ablation studies (e.g. `./exp/` vs `./exp-ablation/`) that happen to produce the same `artifact_stem` (same method/seed/model/optim/hyperparams) will silently overwrite each other's snapshot file even though their experiment output dirs are kept separate.

Why `main.py` can't currently derive this itself: `main.py` never receives `--exp_base` — only `get_save_dir.py` (called separately, by hand or from a slurm script) takes `--exp_base` and bakes it into the `--save_dir` string it returns. `main.py` only ever sees the final `--save_dir`, not the original `--exp_base` value, so there's currently no clean way for it to know which "exp dir" a run belongs to.

## Decision

Add `--exp_base` as an explicit argument to `main.py` itself (mirroring `get_save_dir.py`'s existing `--exp_base` arg), thread it through `config['exp_base']` → `DiagnosticsRunContext` → `SnapshotManager`, and use it to build the snapshots path as:

```
snapshots/<exp_base_name>/<dataset_name>/<artifact_stem>.p
```

where `<exp_base_name>` is `os.path.basename(os.path.normpath(exp_base))` (e.g. `./exp-ablation/` → `exp-ablation`, default `./exp/` → `exp`).

This is preferred over trying to parse the exp_base back out of `--save_dir` (the alternative), because `save_dir` is built by string-concatenation of many fields after the first `os.path.join` (see `utils.py:get_save_dir`), and the blobs slurm script's `model_id` regex (`slurm_run_blobs_deep_linear.py`) already accidentally injects an extra `/` into `save_dir` (a separate pre-existing bug, not in scope here) — parsing `save_dir`'s first path segment to recover `exp_base` would be fragile and silently wrong if that regex bug (or similar) changes how many path segments precede the dataset name. An explicit `--exp_base` argument is unambiguous regardless of how `save_dir` itself is structured.

## Steps

~~1. **`main.py`**: add `parser.add_argument('--exp_base', type=str, default='./exp/', ...)` (matching `get_save_dir.py`'s existing flag/default), and set `config['exp_base'] = args.exp_base` alongside the existing `config['save_dir'] = save_dir` (`main.py:204-205`).~~
~~2. **`methods/method_utils/diagnostics_context.py`**: add an `exp_base: str` field to `DiagnosticsRunContext`.~~
~~3. **`methods/SelectionMethod.py`**: pass `exp_base=self.config['exp_base']` into the `DiagnosticsRunContext(...)` construction (`methods/SelectionMethod.py:105-126`).~~
~~4. **`methods/method_utils/snapshots.py`**: change the `snapshots_dir` line to:
   ```python
   exp_base_name = os.path.basename(os.path.normpath(self.context.exp_base))
   snapshots_dir = os.path.join(self.context.project_root, 'snapshots', exp_base_name, self.context.dataset_name)
   ```~~
~~5. **Slurm scripts** (`slurm_run_blobs_deep_linear.py`, `slurm_run_cifar_3_deep_linear.py`, `slurm_run_mnist_deep_linear.py`): add `--exp_base`, EXP_BASE to each generated `python_cmd` (alongside the existing `--save_dir`), since `main.py` will now require/use it directly rather than only `get_save_dir.py` knowing about it.~~
~~6. **No backfill**: existing snapshots under `snapshots/<dataset_name>/` stay where they are; only new runs get the new `snapshots/<exp_base_name>/<dataset_name>/` layout. (Flag if you'd rather move historical snapshots into an `exp`-named subfolder for consistency — not done by default here.)~~

Also fixed in passing: `main.py`'s fallback `get_save_dir(config, args.notes)` call (when `--save_dir` isn't given) wasn't passing `exp_base` through at all before this change, silently always using the default `'./exp/'` regardless of `--exp_base`. Now it passes `exp_base=args.exp_base`.

## Open question

Should `--exp_base` in `main.py` default to `./exp/` (matching `get_save_dir.py`, so a run launched without specifying it lands in `snapshots/exp/<dataset_name>/...`), or should it be required with no default so every run is always explicit about which exp dir it belongs to?

[[Default to exp]]
{{Done — `--exp_base` defaults to `./exp/` in `main.py`, matching `get_save_dir.py`.}}