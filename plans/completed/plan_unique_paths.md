# Plan: Unique Paths for Blobs Sweep Runs

## Problem

Two independent collision sites exist for runs that differ only in `alpha` or `n_samples`:

### 1. Experiment output directory (`exp-ablation/MakeBlobs/…`)

`get_save_dir` (in `utils.py`) builds a path from method, model type, bs, epochs, lr, optimizer, scheduler, seed, and ratio. It does not include `n_samples` or `alpha`. All blobs runs that share those fields — but differ in `n_samples` or `alpha` — land in the same directory and stomp on each other's `config.yaml`, `checkpoint.pth.tar`, `log.txt`, etc.

Current path shape (reconstructed from the running jobs):
```
exp-ablation/MakeBlobs/RhoLoss_deeplinear_bs320_ep1500_lr0.1_SGD_constant_seed1_r0.1_saxe/
  deep_linear_1024_16layer_hidden/
    2026_Jun_16_log.txt   ← multiple jobs writing here simultaneously
```

### 2. Snapshot / selected-points artifact path (`snapshots/MakeBlobs/…`)

`build_artifact_stem` in `main.py` builds a JSON key containing bsel, seed, model type, optim, bs, ratio, lr, wd, layers, hidden_dim — but not `n_samples` or `alpha`. This stem is used by `SnapshotManager` and `NTKLogger` for file names under `snapshots/` and `selected_points/`. Multiple runs overwrite each other's snapshots.

---

## Root cause

Neither `get_save_dir` nor `build_artifact_stem` is aware of blobs-specific sweep dimensions (`n_samples`, `alpha`). The blobs slurm script iterates over both, but never injects them into the paths it asks `main.py` to use.

---

## Proposed fix

**Principle:** keep changes localized. Don't pollute `get_save_dir` or `build_artifact_stem` with blobs-specific logic. Instead, inject the distinguishing suffix at the two call sites.

### Change 1 — `slurm_run_blobs_deep_linear.py`: add `n`/`alpha` to the exp directory

After `save_dir` is computed from `get_save_dir.py` and `.strip()`-ed, append `_n{n}_alpha{alpha}` **before** the `model_id` suffix (so the structure stays readable):

```python
# existing
save_dir = subprocess.check_output([..., "--exp_base", EXP_BASE], text=True).strip()

# NEW — insert before model_id suffix
save_dir += f'_n{n}_alpha{alpha}'

# existing
model_id = re.search(r'deep_linear_(.+)\.yaml', model).group(1)
save_dir += f'_{model_id}_hidden'
```

Result path shape:
```
exp-ablation/MakeBlobs/RhoLoss_deeplinear_bs320_ep1500_lr0.1_SGD_constant_seed1_r0.1_n1024_alpha1.0_saxe/
  deep_linear_1024_16layer_hidden/
```

### Change 2 — `main.py`: add `--artifact_suffix` CLI argument

Add one new optional argument to `main.py`'s argument parser:

```python
parser.add_argument('--artifact_suffix', type=str, default=None,
                    help='Suffix appended to artifact_stem for snapshot/selected-points file names.')
```

Append it (if provided) after `build_artifact_stem` sets `config['artifact_stem']`:

```python
config['artifact_stem'] = build_artifact_stem(args, config)
if args.artifact_suffix:
    config['artifact_stem'] += f'_{args.artifact_suffix}'
```

### Change 3 — `slurm_run_blobs_deep_linear.py`: pass `--artifact_suffix` to `main.py`

In the `python_cmd` list, add:

```python
"--artifact_suffix", f"n{n}_alpha{alpha}",
```

This makes snapshot file names like:
```
snapshots/MakeBlobs/{"bsel":"RhoLoss","seed":1,...,"layers":16,"hidden_dim":1024}_n1024_alpha1.0.p
```

---

## Files changed

| File | Change |
|------|--------|
| `slurm_run_blobs_deep_linear.py` | Add `_n{n}_alpha{alpha}` to `save_dir` (Change 1), pass `--artifact_suffix n{n}_alpha{alpha}` in `python_cmd` (Change 3) |
| `main.py` | Add `--artifact_suffix` arg; append it to `artifact_stem` when provided (Change 2) |

`utils.py`, `get_save_dir.py`, and `build_artifact_stem` are **not modified**.

---

## Checklist

- ~~Change 1: `slurm_run_blobs_deep_linear.py` — append `_n{n}_alpha{alpha}` to save_dir before model_id suffix~~
- ~~Change 2: `main.py` — add `--artifact_suffix` arg and append to `config['artifact_stem']`~~
- ~~Change 3: `slurm_run_blobs_deep_linear.py` — add `--artifact_suffix n{n}_alpha{alpha}` to `python_cmd`~~
