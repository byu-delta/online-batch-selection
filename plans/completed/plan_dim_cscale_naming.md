# Plan: Include `dim` and `center_scale` in MakeBlobs Snapshot/Save-Dir Names

## Context

`slurm_run_blobs_deep_linear.py` sweeps over `DIMS` and `CENTER_SCALES` (along with seeds, methods, models, optims, `N_SAMPLES`, `ALPHAS`). Two artifacts are named per-run:

1. **Snapshot files** (`snapshots/MakeBlobs/<artifact_stem>.p`), built by `build_artifact_stem()` in `main.py:18-32`. The stem currently encodes `bsel, seed, model, opt, bs, ratio, lr, wd, layers, hidden_dim`. The script appends `--artifact_suffix f"n{n}_alpha{alpha}"` (`slurm_run_blobs_deep_linear.py:223`), so `n_samples`/`alpha` are covered too — but **`dim` and `center_scale` are absent everywhere**.
2. **Experiment save dirs** (`exp-ablation/...`), built by `get_save_dir()` in `utils.py:42-57`, which also has no notion of `dim`/`center_scale`. The slurm script manually appends `_n{n}_alpha{alpha}` and `_{model_id}_hidden` (`slurm_run_blobs_deep_linear.py:206-208`) — again no `dim`/`cscale`.

Today `DIMS = [32]` and `CENTER_SCALES = [1.5]` are both single-element lists, so there's no actual collision yet. But if either list grows to more than one value, runs/snapshots for different `dim`/`cscale` will silently overwrite each other since every other identifying field is unchanged.

## Goal

Make every varying parameter in `slurm_run_blobs_deep_linear.py`'s sweep (`SEEDS, DIMS, CENTER_SCALES, N_SAMPLES, ALPHAS, METHODS_*, MODEL_CONFIGS, OPTIMS`) reflected in both the snapshot filename and the save dir, so no two distinct configs can ever map to the same path.

## Steps

~~1. **Snapshot naming**: in `slurm_run_blobs_deep_linear.py`, change the `--artifact_suffix` value (currently `f"n{n}_alpha{alpha}"`, line 223) to also include `dim` and `cscale`, e.g. `f"d{dim}_cscale{cscale}_n{n}_alpha{alpha}"`.~~
~~2. **Save dir naming**: in the same script, extend the manual suffix built at lines 206-208 (`save_dir += f'_n{n}_alpha{alpha}'`) to also include `dim`/`cscale`, matching the format used for the snapshot suffix for consistency.~~
~~3. **Sanity check**: confirm no other currently-swept field is silently missing. Re-derive the full list of fields encoded in `build_artifact_stem()` + the slurm-script-applied suffix + save_dir, and diff against every variable in the `product(...)` calls that build `jobs` (lines 162-184) and `labels_job_ids` (line 136). (Already manually checked as of this plan's writing — only `dim`/`cscale` were missing.)~~
~~4. **No backfill of old data**: existing snapshots/save dirs from the single-dim/single-cscale sweep remain valid as-is (no collision occurred), so this is a forward-looking fix only — no renaming of existing files required.~~

## Notes

- This only touches `slurm_run_blobs_deep_linear.py` (the script that builds the `--artifact_suffix` and `save_dir` strings) — `main.py`'s `build_artifact_stem()` and `utils.py`'s `get_save_dir()` stay generic/dataset-agnostic, as they are shared across CIFAR3/MNIST/MakeBlobs sweep scripts.
- If `slurm_run_blobs_deep_linear.py` ever adds another swept parameter in the future, remember to extend both suffixes again.
