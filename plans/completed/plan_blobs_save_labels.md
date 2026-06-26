# Plan: Run save_labels.py from slurm_run_blobs_deep_linear.py

## Context

`save_labels.py` saves `{"train": y_train, "val": y_val}` to a `.p` file for use in analysis notebooks. It needs to be run once per unique data config. The default output path `labels/MakeBlobs.p` would collide across different generated configs (different dims and center scales), so we must use `--output` to give each a unique path.

The makeblobs split seed comes from `dataset.random_state` in the data config (set to `42` in the template), not from `--seed` passed to `save_labels.py`, so `--seed` is irrelevant for the split here.

## Change

In `slurm_run_blobs_deep_linear.py`, after the call to `write_generated_configs(DIMS, CENTER_SCALES)` (line 89), add a loop that calls `save_labels.py` for each generated data config:

```python
for dim, cscale in product(DIMS, CENTER_SCALES):
    data_cfg = str(GEN_DIR / "data" / f"makeblobs_d{dim}_cscale{cscale}.yaml")
    out_path = f"labels/makeblobs_d{dim}_cscale{cscale}.p"
    subprocess.run(
        [
            "python", "save_labels.py",
            "--data", data_cfg,
            "--output", out_path,
            "--overwrite",
        ],
        check=True,
    )
```

`--overwrite` is passed so re-running the script doesn't error if the labels file already exists.
