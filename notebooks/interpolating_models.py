import sys
from pathlib import Path
# sys.path.append(str(Path.cwd().parent))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
import argparse

import models

parser = argparse.ArgumentParser()

parser.add_argument("--thr1", type=float, default=None)
parser.add_argument("--thr2", type=float, default=None)
parser.add_argument("--model1", type=str, required=True)
parser.add_argument("--model2", type=str, required=True)
parser.add_argument("--epoch1", type=int, required=True)
parser.add_argument("--epoch2", type=int, required=True)
parser.add_argument("--num_points", type=int, default=25)
parser.add_argument("--progress", type=float)
parser.add_argument("--progress_diff", type=float)
parser.add_argument("--method", choices=["DivBS_RhoLoss", "RhoLoss_DivBS"], default="DivBS_RhoLoss")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=ROOT / "interpolation/cifar10interpolation_10-07-26/data",
)
parser.add_argument("--output-name", type=str, default=None)
parser.add_argument("--dataset-root", type=Path, default=ROOT / "_CIFAR")
parser.add_argument("--batch-size", type=int, default=2048)
parser.add_argument("--device", type=str, default="cuda")

args = parser.parse_args()

# model1_path = "/home/todd30ap/online-batch-selection/experiments/cifar10_interpolation/9-16-26/DivBS-RhoLoss_log-schedule/20260915_213949_CIFAR10_DivBS_RhoLoss_progThreshold-1_Seed-1/snapshots/epochs/epoch064.pth.tar"
# model2_path = "/home/todd30ap/online-batch-selection/experiments/cifar10_interpolation/9-16-26/DivBS-RhoLoss_log-schedule/20260915_213949_CIFAR10_DivBS_RhoLoss_progThreshold-1_Seed-1/snapshots/epochs/epoch075.pth.tar"

# epoch = 64
# thr1 = 0.0
# thr2 = 0.0
# num_points = 25
# chart_title = f"DivBS-RhoLoss Interpolation between Epoch: 64 and Epoch: 75 with Thr: 1.0"
# data_name = f"DivBS-RhoLos_thr-1_epoch-64-75"

model1_path = args.model1
model2_path = args.model2

epoch1 = args.epoch1
epoch2 = args.epoch2
progress = args.progress
thr1 = args.thr1
thr2 = args.thr2
num_points = args.num_points

method_label = "RhoLoss-DivBS" if args.method == "RhoLoss_DivBS" else "DivBS-RhoLos"
progress_label = "unknown" if progress is None else str(progress)
chart_title = f"{args.method} interpolation: threshold {thr1}, epoch {epoch1} to threshold {thr2}, epoch {epoch2}"
data_name = args.output_name or (
    f"Train_{method_label}_thr-{thr1}-{thr2}"
    f"_progress-{progress_label}_step-{epoch1 * 157}-{epoch2 * 157}"
)
print(f"Mean progress: {progress}; absolute progress difference: {args.progress_diff}", flush=True)


def get_model(model_path):
    snapshot = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = snapshot['state_dict']

    # These checkpoints were trained with BatchNorm disabled. Keep this
    # explicit so the loader cannot construct the wrong architecture.
    model = models.ResNet(m_type='resnet18', num_classes=10, BatchNorm=False)
    model.load_state_dict(state_dict)
    return model

def interpolate_models(model1, model2):
    model = copy.deepcopy(model1)
    model.eval()

    params = list(model.parameters())
    params1 = list(model1.parameters())
    params2 = list(model2.parameters())

    # buffers = list(model.buffers())
    # buffers1 = list(model1.buffers())
    # buffers2 = list(model2.buffers())

    def model_at_t(t):
        with torch.inference_mode():
            for p, p1, p2 in zip(params, params1, params2):
                p.copy_(t * p1 + (1 - t) * p2)

            # These runs have no BatchNorm buffers. Keep buffer interpolation
            # disabled; enabling it would change this experiment's model.
            # for b, b1, b2 in zip(buffers, buffers1, buffers2):
            #     if b.dtype.is_floating_point:
            #         b.copy_(t * b1 + (1 - t) * b2)
            #     else:
            #         b.copy_(b1)
        return model
    return model_at_t

def plot_data(model1, model2, data_loader, loss_func, device, chart_title, data_name, num_points):
    print("Starting Interpolation", flush=True)

    with torch.no_grad():
        model_func = interpolate_models(model1, model2)

        print("Interpolating Models Function Created", flush=True)
        inputs = np.linspace(0, 1, num_points)
        losses = []
        for t in inputs:
            print(f"Interpolating Model {t}", flush=True)
            model = model_func(t)
            model.train()
            loss_sum = 0.0
            num_samples = 0
            for images, targets in data_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss_sum += loss_func(outputs, targets).item()
                num_samples += targets.size(0)
            losses.append(loss_sum / num_samples)

        print("Saving Data", flush=True)
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f"{data_name}.npy", np.array(losses))

        # print("Generating Plot", flush=True)
        # plt.plot(inputs, losses)
        # plt.xlabel('t')
        # plt.ylabel('Loss')
        # plt.title(chart_title)
        # plt.savefig(f"./interpolation/plots/{data_name}.png")
        # print("Plot Saved", flush=True)

print("Starting", flush=True)

device = torch.device(args.device)

print("Loading Model 1", flush=True)
model1 = get_model(model1_path).to(device)
print("Loading Model 2", flush=True)
model2 = get_model(model2_path).to(device)
print("Finished Loading Models", flush=True)

print("setting vars", flush=True)
im_size = (32, 32)
num_classes = 10
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

print("setting transforms", flush=True)
transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 32 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

print("Loading Dataset", flush=True)
dataset = datasets.CIFAR10(str(args.dataset_root), train=True, download=False, transform=transform)
print("Creating DataLoader", flush=True)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
print("DataLoader Created", flush=True)

plot_data(model1, model2, data_loader, nn.CrossEntropyLoss(reduction="sum"), device, chart_title, data_name, num_points)
