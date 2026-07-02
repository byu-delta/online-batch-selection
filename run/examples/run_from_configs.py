# Copy this script into `run/` (as opposed to `run/examples/`, where it is
# currently) to use it!

from run_utils import run_job, RunType

RUN_TYPE = RunType.DRY # Change to NORMAL, SBATCH, or SRUN to actually run

CONFIGS = ["configs/examples/mnist_noise.yaml"]

for config_path in CONFIGS:
    run_job(config_path, RUN_TYPE)

print("Completed run submission script.")
