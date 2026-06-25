# Plan: Lazy-Import `wandb` in `utils.py`

## Context

`slurm_run_blobs_deep_linear.py` shells out to `get_save_dir.py` once per job (216 times in the last sweep) just to compute a save-dir path string. Each call takes ~5.5s, which is slow enough that by job #62 the corresponding label-generation job's record had already been purged from Slurm's job table (`MinJobAge=300s` on this cluster), causing `sbatch --dependency=afterok:<purged_job_id>` to fail with "Job dependency problem" and crash the whole submission loop.

Root cause, confirmed by timing: `utils.py:6` does `import wandb` at module level. `get_save_dir.py:2` does `from utils import get_save_dir, get_configs`, which executes the entire `utils.py` module — including that `import wandb` — even though neither `get_save_dir()` nor `get_configs()` touch wandb at all.

```
$ time python -c "import wandb"   # inside online-bs-p100 env
real    0m5.157s
```

That ~5.15s accounts for essentially all of the observed ~5.5s/job overhead. The only real uses of `wandb` in `utils.py` are three methods on the logger class (`wandb_init`, `wandb_log`, `wandb_finish`, lines 103-114).

## Decision

Move `import wandb` from module scope into the three methods that actually use it (or at minimum into `wandb_init`, since `wandb_log`/`wandb_finish` are only ever called after `wandb_init` has already run in the same process — but importing in all three is more obviously correct and just as cheap, since repeated imports of an already-loaded module are free). This makes `get_save_dir.py` (and anything else importing `utils` without ever calling these methods) pay zero `wandb` import cost, while `main.py`'s actual training runs — which always call `wandb_init` — pay the cost exactly once, same as today.

## Steps

~~1. **`utils.py`**: remove `import wandb` from the top of the file (line 6).~~
~~2. **`utils.py`**: add `import wandb` as the first line inside `wandb_init`, `wandb_log`, and `wandb_finish` (lines 103-114), each preceded by the comment `# Imported here to keep utils' module-level imports light, because importing wandb takes a long time`.~~ [[How about just in wandb_init? would that still work?]]
{{No — a plain `import wandb` inside `wandb_init` binds `wandb` as a local name in `wandb_init`'s own scope only; it does not add `wandb` to the module's global namespace, so `wandb_log`/`wandb_finish` (separate function scopes that reference the bare name `wandb`) would raise `NameError` once `wandb` is no longer imported at module level. (You *could* work around this with `global wandb; import wandb` inside `wandb_init` to force it into module globals, but that's a hacky one-method-mutates-global-state pattern and fragile if `wandb_log`/`wandb_finish` are ever called before `wandb_init` in some future code path.) Importing in all three is simpler and effectively free after the first call in a process — once `wandb` is in `sys.modules`, every subsequent `import wandb` anywhere is just a dict lookup, not a re-execution.}}
[[Definitely add a comment: # Imported here to keep utils' module-level imports light, because importing wandb takes a long time]]
{{Done — added above.}}
~~3. **Verify**: re-run the timing check (`time python -c "from utils import get_save_dir"`) and confirm it no longer pays the ~5s cost; also do a quick smoke-test run of `main.py` (which calls `wandb_init`/`wandb_log`/`wandb_finish` via the logger) to confirm wandb logging still works unchanged.~~

Verification results: `time python -c "from utils import get_save_dir"` dropped from ~5s to 0.288s; `get_save_dir.py` end-to-end dropped from ~5.5s to 0.332s. A standalone smoke test of `custom_logger.wandb_init`/`wandb_log`/`wandb_finish` (with `WANDB_MODE=disabled`) confirmed all three still work correctly with the lazy import.

~~4. **No change needed to the Slurm dependency/label-job logic** — with `get_save_dir.py` now fast, the 216-job submission loop should comfortably finish well within the 300s `MinJobAge` window, making the dependency-aging failure mode moot without needing `sbatch --wait` or any other workaround.~~
