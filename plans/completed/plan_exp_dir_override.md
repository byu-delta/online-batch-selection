# Plan: Make base experiment directory configurable

## Problem

`utils.get_save_dir` hardcodes `./exp/` as the base. To redirect a run to
`./exp-ablation/MakeBlobs/...` without touching dataset name or post-processing
paths in the slurm script, we need to make the base configurable.

## Changes

### 1. `utils.py` — add `exp_base` parameter to `get_save_dir`

```python
def get_save_dir(config, notes=None, exp_base='./exp/'):
    save_dir = exp_base
    save_dir = os.path.join(save_dir, config['dataset']['name'])
    ...  # rest unchanged
```

### 2. `get_save_dir.py` — add `--exp_base` argument

```python
parser.add_argument('--exp_base', type=str, default='./exp/',
                    help='Base directory for experiment outputs.')
```

Pass it through:

```python
print(get_save_dir(config, args.notes, exp_base=args.exp_base))
```

### 3. `slurm_run_blobs_deep_linear.py` — add `EXP_BASE` constant and pass it

```python
EXP_BASE = "./exp/"  # change to e.g. "./exp-ablation/" to redirect output
```

Pass it when calling `get_save_dir.py`:

```python
get_save_dir_cmd = [
    "python", "get_save_dir.py",
    "--method", method,
    "--data", data,
    "--model", model,
    "--optim", optim,
    "--seed", str(seed),
    "--exp_base", EXP_BASE,
]
save_dir = subprocess.check_output(get_save_dir_cmd, text=True).strip()
```

## Notes

- Default `./exp/` preserves all existing behavior.
- `main.py` receives `--save_dir` explicitly so needs no changes.

## Summary of changes

- ~~Add `exp_base` param to `get_save_dir` in `utils.py`~~
- ~~Add `--exp_base` arg to `get_save_dir.py`~~
- ~~Add `EXP_BASE` constant to `slurm_run_blobs_deep_linear.py` and pass it to `get_save_dir.py`~~ {{Also adding to `slurm_run_cifar_3_deep_linear.py` and `slurm_run_mnist_deep_linear.py`}}
- ~~Add `EXP_BASE` to `slurm_run_cifar_3_deep_linear.py` and `slurm_run_mnist_deep_linear.py`~~
