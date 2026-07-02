# Copy this script into `run/` (as opposed to `run/examples/`, where it is
# currently) to use it!

from run_utils import run_job, RunType, generate_configs
from tqdm import tqdm

RUN_TYPE = RunType.DRY # Change to NORMAL, SBATCH, or SRUN to actually run.

TEMPLATE = "configs/examples/cifar3_template.yaml"

PARAMS_TO_VARY = {
    "seed": [1, 2, 3],
    "training_opt.optimizer": ["SGD", "AdamW"],
    "training_opt.optim_params.lr": [0.001, 0.01, 0.1]
}

config_paths = generate_configs(TEMPLATE, PARAMS_TO_VARY)

for config_path in tqdm(config_paths, desc='Submitting runs'):
    run_job(config_path, RUN_TYPE)

print("Completed run submission script.")
