"""Submit the basic single-dataset baseline runs.

Each entry is a concrete merged config under ./config_templates/ run via
`main.py --config <config>` (the seed is a top-level key in the config). Run
output dirs are claimed at runtime under ./experiments/; SLURM stdout/stderr
go to logs/slurm/%j.{out,err}. Jobs request --requeue so preemption restarts land
back in the same run dir. Set USE_SLURM=False to run locally instead.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import run_job, RunType

CONFIGS = [] # Add configs here

RUN_TYPE = RunType.SBATCH

for config_path in CONFIGS:
    run_job(config_path, RUN_TYPE)

print("Run script completed.")
