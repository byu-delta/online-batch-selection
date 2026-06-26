# Plan: Submit save_labels.py as SLURM job

## Problem

`save_labels.py` needs a GPU but is currently run directly (lines 91–102 of
`slurm_run_blobs_deep_linear.py`). When `USE_SLURM = True` it should be
submitted via `sbatch`, just like the main experiment jobs. The main jobs for a
given `(dim, cscale)` must not start until that pair's labels file exists, so
they need a SLURM dependency on the corresponding labels job.

## Changes to `slurm_run_blobs_deep_linear.py`

### 1. Extract a shared `make_sbatch` helper

Replace the inline `sbatch_script = dedent(...)` block with a function so both
the labels job and the experiment jobs use identical SLURM resource declarations:

```python
def make_sbatch(cmd: list[str], job_name: str) -> str:
    return dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --output=logs/%j.out
        #SBATCH --error=logs/%j.err
        #SBATCH --gres=gpu:1
        #SBATCH --cpus-per-task=4
        #SBATCH --mem=32GB
        #SBATCH --time=1:00:00
        #SBATCH -C pascal

        {shlex.join(cmd)}
        """
    )
```
[[Add an optional arg (str) for time, defaulting to '1:00:00'. The save_labels call will only need 15 minutes]]


### 2. Submit save_labels.py via sbatch (when USE_SLURM)

Replace the current `save_labels.py` loop with one that branches on `USE_SLURM`.
When true, submit each `(dim, cscale)` pair as an sbatch job and record the
returned job ID.  When false, keep the existing direct `subprocess.run`.
[[good]]

```python
labels_job_ids: dict[tuple, str] = {}   # (dim, cscale) -> slurm job id

for dim, cscale in product(DIMS, CENTER_SCALES):
    data_cfg = str(GEN_DIR / "data" / f"makeblobs_d{dim}_cscale{cscale}.yaml")
    out_path = f"labels/makeblobs_d{dim}_cscale{cscale}.p"
    cmd = [
        "python", "save_labels.py",
        "--data", data_cfg,
        "--output", out_path,
        "--overwrite",
    ]
    if USE_SLURM:
        result = subprocess.run(
            ["sbatch"],
            input=make_sbatch(cmd, f"labels_d{dim}_cs{cscale}"),
            text=True,
            check=True,
            capture_output=True,
        )
        # sbatch prints "Submitted batch job <id>"
        job_id = result.stdout.strip().split()[-1]
        labels_job_ids[(dim, cscale)] = job_id
    else:
        subprocess.run(cmd, check=True)
```

### 3. Add dependency to experiment job submissions

When building each experiment's sbatch script, look up the labels job ID for
that experiment's `(dim, cscale)` and add `--dependency=afterok:<id>`.

In the job-submission loop, parse `(dim, cscale)` from the `data` path (already
available in scope as loop variables), then add to `make_sbatch`:

```python
def make_sbatch(cmd: list[str], job_name: str, dependency: str | None = None) -> str:
    dep_line = f"#SBATCH --dependency=afterok:{dependency}\n" if dependency else ""
    return dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --output=logs/%j.out
        #SBATCH --error=logs/%j.err
        {dep_line}#SBATCH --gres=gpu:1
        #SBATCH --cpus-per-task=4
        #SBATCH --mem=32GB
        #SBATCH --time=1:00:00
        #SBATCH -C pascal

        echo "save_dir: {save_dir}"

        {shlex.join(cmd)}
        """
    )
```

And in the submission loop, pass the dependency:

```python
dep = labels_job_ids.get((dim, cscale)) if USE_SLURM else None
sbatch_script = make_sbatch(
    python_cmd + ['--wandb_not_upload'],
    job_name=f"blobs_s{seed}",
    dependency=dep,
)
```

The experiment-job loop currently iterates over a flat `jobs` list whose tuples
don't carry `dim`/`cscale`. We need to thread them through. Change the jobs list
to include `dim` and `cscale`:

```python
jobs = (
    [
        (seed, dim, cscale,
         str(GEN_DIR / "data"   / f"makeblobs_d{dim}_cscale{cscale}.yaml"),
         model, optim,
         str(GEN_DIR / "method" / f"{method_name}_d{dim}_cscale{cscale}.yaml"),
        )
        for seed, dim, cscale, optim, method_name, model
        in product(SEEDS, DIMS, CENTER_SCALES, OPTIMS, METHODS_HYPERPLANE, MODEL_CONFIGS)
    ] + [
        (seed, dim, cscale,
         str(GEN_DIR / "data" / f"makeblobs_d{dim}_cscale{cscale}.yaml"),
         model, optim, method,
        )
        for seed, dim, cscale, optim, method, model
        in product(SEEDS, DIMS, CENTER_SCALES, OPTIMS, METHODS_FIXED, MODEL_CONFIGS)
    ]
)
```

And unpack accordingly in the submission loop:

```python
for seed, dim, cscale, data, model, optim, method in tqdm(jobs, ...):
```

[[good]]
## Summary of all changes

- ~~Add `make_sbatch(cmd, job_name, dependency=None)` function~~
- ~~Replace inline `dedent(...)` in experiment loop with `make_sbatch(...)` call~~
- ~~Replace direct `subprocess.run` of `save_labels.py` with sbatch submission (when `USE_SLURM`), collecting job IDs~~
- ~~Thread `dim`/`cscale` through the `jobs` list and submission loop~~
- ~~Pass `dependency=labels_job_ids.get((dim, cscale))` when submitting experiment jobs~~
