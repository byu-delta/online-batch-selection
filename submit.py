import subprocess
import shlex

SEEDS = [1, 2, 3, 4, 5]

CONFIG_DIR = "configs/mnist"

DIAGNOSTICS = f"{CONFIG_DIR}/diagnostics/feature_learning_log_interval.yaml"

METHODS = [
    f"{CONFIG_DIR}/method/rholoss-0.1.yaml",
    f"{CONFIG_DIR}/method/uniform-0.1.yaml",
    f"{CONFIG_DIR}/method/DivBS-0.1.yaml",
]

MODELS = [f"{CONFIG_DIR}/model/lenet.yaml"]
OPTIMS = [f"{CONFIG_DIR}/optim/adamw-320-0.001-0.01.yaml"]
DATAS = [f"{CONFIG_DIR}/data/mnist.yaml"]

for seed in SEEDS:
    for data in DATAS:
        for model in MODELS:
            for optim in OPTIMS:
                for method in METHODS:

                    command = (
                        f"python main.py "
                        f"--method {shlex.quote(method)} "
                        f"--data {shlex.quote(data)} "
                        f"--model {shlex.quote(model)} "
                        f"--optim {shlex.quote(optim)} "
                        f"--diagnostics {shlex.quote(DIAGNOSTICS)} "
                        f"--seed {seed} "
                        f"--wandb_not_upload"
                    )

                    sbatch_cmd = [
                        "sbatch",
                        "--job-name=mnist",
                        "--output=logs/%x_%j.out",
                        "--error=logs/%x_%j.err",
                        "--nodes=1",
                        "--ntasks=1",
                        "--cpus-per-task=8",
                        "--mem=16G",
                        "--gres=gpu:1",
                        "--qos=standby",
                        "--time=3-00:00:00",
                        "--wrap",
                        command,
                    ]

                    result = subprocess.run(
                        sbatch_cmd,
                        capture_output=True,
                        text=True,
                        check=True,
                    )

                    print(result.stdout.strip())