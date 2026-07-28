from pathlib import Path
import os
import shlex
from tqdm import tqdm
import subprocess
import yaml
from enum import Enum

from .generate_configs import CONFIGS_TEMP_DIR, _set_dotted

class RunType(Enum):
    DRY = 0 # Does not run at all
    NORMAL = 1 # For use on systems without slurm
    SBATCH = 2 # To run jobs in the background
    SRUN = 3   # To run jobs with slurm in series, not in the background

def run_job(
        config_path, 
        run_type: RunType = RunType.SBATCH,
        *,
        cpus: str = '4',
        mem: str = '32GB', # gb
        time: str = '1:00:00',
        name: str = 'online-bs',
        preemptible=True,
        download=True,
        wandb_upload=False,
        hide_slurm_id=False, # Allows one to run jobs on a slurm allocation with RunType.NORMAL without causing jobs to resume
        experiments_dir="."
    ):
    if download:
        download_cmd = ["python", "perform_downloads.py", "--method", config_path]
        if run_type == RunType.DRY:
            tqdm.write(f'Dry run. Would have run `{download_cmd}`')    
        else:
            # Perform necessary downloads
            subprocess.run(download_cmd, check=True)

    python_cmd = ["python", "main.py", "--config", config_path, "--experiments_dir", experiments_dir]

    if not wandb_upload:
        python_cmd.append("--wandb_not_upload")

    if run_type == RunType.DRY:
        tqdm.write(f'Dry run. Would have run `{python_cmd}`')

    if run_type == RunType.NORMAL:
        env = None
        if hide_slurm_id:
            env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
        subprocess.run(python_cmd, check=True, env=env)
        return

    slurm_flags = [
        "--gres=gpu:1",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--time={time}",
        f"--job-name={name}",
    ]
    if exclude_nodes is not None:
        slurm_flags.append(f"--exclude={','.join(exclude_nodes)}")

    if run_type == RunType.SRUN:
        subprocess.run(["srun"] + slurm_flags + python_cmd, check=True)
        return
    
    # Prepare slurm log dir
    Path("logs/slurm").mkdir(parents=True, exist_ok=True)
    slurm_flags += [
        "--output=logs/slurm/%j.out",
        "--error=logs/slurm/%j.err",
    ]

    if preemptible:
        slurm_flags += [
            "--requeue",
            "--qos=standby"
        ]


    if run_type == RunType.SBATCH:
        subprocess.run(
            ["sbatch"] + slurm_flags + [f"--wrap={shlex.join(python_cmd)}"],
            check=True,
        )
        return


def _read_run_info_value(experiment_dir, key, default=None):
    """Read a single key out of a run's `run_info.yaml` (its mutated config
    snapshot), falling back to `default` if the file or key is absent (e.g. a
    run that predates `run_info.yaml`)."""
    run_info_path = os.path.join(experiment_dir, "run_info.yaml")
    if not os.path.isfile(run_info_path):
        return default
    with open(run_info_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data.get(key, default)


def extend_job(experiment_dir, *args, additional_epochs, overrides=None, **kwargs):
    """Continue a previous run for `additional_epochs` more epochs.

    Loads `<experiment_dir>/input_config.yaml`, applies `overrides` (a
    dotted-path -> value dict, for changing method/lr/diagnostics/etc. on the
    continued run), and wires up `resume.from`/`resume.additional_epochs`
    before delegating to `run_job`. All other args/kwargs pass straight
    through to `run_job`, except `experiments_dir`, which defaults to the
    parent run's own logged `experiments_dir` (from its `run_info.yaml`) so
    the extended run lands in the same subdirectory unless overridden.
    """
    input_config_path = os.path.join(experiment_dir, "input_config.yaml")
    with open(input_config_path, "r") as f:
        config = yaml.safe_load(f)

    for dotted, value in (overrides or {}).items():
        _set_dotted(config, dotted, value)

    resume = config.setdefault("resume", {})
    resume["from"] = experiment_dir
    resume["additional_epochs"] = additional_epochs

    os.makedirs(CONFIGS_TEMP_DIR, exist_ok=True)
    out_path = os.path.join(
        CONFIGS_TEMP_DIR, f"resume_{os.path.basename(os.path.normpath(experiment_dir))}.yaml"
    )
    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    kwargs.setdefault(
        "experiments_dir",
        _read_run_info_value(experiment_dir, "experiments_dir", default="."),
    )

    return run_job(out_path, *args, **kwargs)