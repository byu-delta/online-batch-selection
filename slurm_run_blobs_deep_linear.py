from itertools import product
from pathlib import Path
from textwrap import dedent
from datetime import datetime
from tqdm import tqdm
import subprocess
import re

WANDB_PROJECT = "Matthew—Deep Linear Networks (Blobs)"

SEEDS = [1]
DIAGNOSTICS = "configs/diagnostics/snapshots_log_interval.yaml"
CONFIG_DIR = "configs/makeblobs"

METHODS = [
    f"{CONFIG_DIR}/method/uniform-0.1.yaml",
]

MODELS = [
    f"{CONFIG_DIR}/model/deep_linear_saxe/deep_linear_1024_3layer.yaml",
]

OPTIMS = [f"{CONFIG_DIR}/optim/adamw-320-0.001-0.01.yaml"]
DATAS = [f"{CONFIG_DIR}/data/makeblobs_1024d_2class.yaml"]

Path("logs").mkdir(exist_ok=True)

CENTERS_NPY = "models/teacher/makeblobs_1024d_centers_seed42.npy"
if not Path(CENTERS_NPY).exists():
    print("Teacher model not found — generating geometry and teacher...")
    subprocess.run(
        [
            "python", "data/make_blobs_teacher.py",
            "--n_features", "1024",
            "--center_scale", "1.0",
            "--center_seed", "42",
            "--alpha", "0.5",
            "--noise_seed", "0",
            "--out_dir", "models/teacher",
        ],
        check=True,
    )

save_dirs_file = (
    Path("logs")
    / f"save_dirs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

jobs = list(product(SEEDS, DATAS, MODELS, OPTIMS, METHODS))

with open(save_dirs_file, "w") as f:
    for seed, data, model, optim, method in tqdm(
        jobs,
        desc="Submitting jobs",
        total=len(jobs),
    ):
        save_dir = subprocess.check_output(
            [
                "python",
                "get_save_dir.py",
                "--method", method,
                "--data", data,
                "--model", model,
                "--optim", optim,
                "--seed", str(seed),
            ],
            text=True,
        ).strip()

        model_id = re.search(r'deep_linear_(.+)\.yaml', model).group(1)
        save_dir += f'_{model_id}_hidden'

        f.write(save_dir + "\n")
        f.flush()

        sbatch_script = dedent(
            f"""\
            #!/bin/bash
            #SBATCH --job-name=blobs_s{seed}
            #SBATCH --output=logs/%j.out
            #SBATCH --error=logs/%j.err
            #SBATCH --gres=gpu:1
            #SBATCH --cpus-per-task=4
            #SBATCH --mem=8GB
            #SBATCH --time=0:30:00

            echo "save_dir: {save_dir}"

            python main.py \\
                --method "{method}" \\
                --data "{data}" \\
                --model "{model}" \\
                --optim "{optim}" \\
                --diagnostics "{DIAGNOSTICS}" \\
                --seed "{seed}" \\
                --save_dir "{save_dir}" \\
                --wandb_not_upload \\
                --wandb_project "{WANDB_PROJECT}"
            """
        )

        subprocess.run(
            ["sbatch"],
            input=sbatch_script,
            text=True,
            check=True,
        )

print(
    "All jobs submitted. Running Weights & Biases sync daemon. Ctrl+C to stop syncing"
)

subprocess.run(
    [
        "python",
        "wandb-sync-daemon.py",
        "--save_dirs",
        str(save_dirs_file),
    ],
    check=True,
)
