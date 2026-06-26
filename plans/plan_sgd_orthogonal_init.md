# Plan: Orthogonal Init for DeepLinear to Fix SGD Training Failure

## Context

This started as an investigation into why SGD fails to train the 16-layer `DeepLinear`/`DeepLinearReLU` models while AdamW trains them fine. The findings below are unchanged from that investigation; the **Steps** section at the end is the actual plan to implement.

## Symptom

Comparing two SLURM runs with identical method/data/model (`Uniform`, `makeblobs_d32_cscale1.0_n16384_alpha1.5`, `deep_linear_1024_16layer` / `deep_linear_relu_1024_16layer`), differing only in optimizer:

- **AdamW** (`logs/12345295.out`, lr=0.001): loss drops from 0.693 → 0.35 and train acc climbs to ~0.85+ within 3-5 epochs.
- **SGD** (`logs/12345296.out`, `logs/12345297.out`, lr=0.01, no momentum): loss stays pinned at **0.6931-0.6934** (= ln 2, i.e. exactly what a model outputting all-zero logits would give on a balanced binary task) and train acc stays at exactly **0.5000** for 500+ epochs straight. Both the identity and ReLU variant show this.

This isn't a tuning issue — `slurm_run_blobs_deep_linear.py`'s `OPTIMS` list already sweeps SGD lr over five orders of magnitude (`1, 0.1, 0.01, 0.0001, 0.00001`). A real "wrong learning rate" would show *some* lr in that range making progress (even if slow/unstable at the extremes). Getting near-machine-precision-identical loss across the whole sweep points to a structural problem, not a hyperparameter problem.

## Root cause: no orthogonal/variance-preserving init, vanishing gradients with depth

`models/DeepLinear.py` builds the network purely from `nn.Linear` layers and never touches their initialization — it relies on PyTorch's default `nn.Linear.reset_parameters()` (`kaiming_uniform_(weight, a=sqrt(5))`, which is `mode='fan_in'` and not tuned for identity/ReLU here). I grepped for any custom init logic (`orthogonal`, `saxe`, `init_weights`, `kaiming`, `xavier`, `reset_parameters`) across the repo — the only hits are in `ResNet.py`. **Nothing initializes `DeepLinear`/`DeepLinearReLU` with an orthogonal or otherwise depth-aware scheme**, despite the model configs living under `configs/makeblobs/model/deep_linear_saxe/` — a name that strongly implies Saxe-style orthogonal initialization (Saxe et al. 2013, "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks") was intended but was never actually implemented.

Why this matters for depth specifically: with default `nn.Linear` init, each layer's output variance is roughly `fan_in * Var(weight) * input_var`. For Kaiming-uniform with `a=sqrt(5)`, `Var(weight) ≈ 1/(3·fan_in)`, so each layer scales variance by roughly `1/3`. Stacking 16 of these (hidden_dim=1024, identity activation) shrinks the forward signal by roughly `(1/3)^16 ≈ 1.3e-8` — the logits are essentially all driven to ~0 before training even starts. That matches the observed loss of exactly `ln 2` (a model with all-zero logits on a balanced 2-class problem). The backward gradients shrink by the same compounding factor, since backprop through a linear chain multiplies by the same weight matrices (transposed).

This explains the AdamW-vs-SGD asymmetry directly:
- **AdamW normalizes each parameter's update by its own running RMS gradient magnitude** (Adam's second-moment estimate). Even if the raw gradient at an early layer is `1e-8` instead of `1e-2`, Adam's update step is still roughly unit-scale (governed by the configured `lr`), so it doesn't matter that the raw gradient vanished — Adam adaptively rescales it back up.
- **Plain SGD has no such per-parameter rescaling.** The update is just `lr * grad`. If `grad ≈ 1e-8` for the early layers, no value of `lr` in a reasonable range (`1e-5` to `1`) produces a meaningful update — it's either still negligible (small lr) or, if it were large enough to move the vanished-gradient layers, would massively overshoot any layer whose gradient *isn't* vanished (e.g. the final classifier layer, which sees the largest gradient since it's closest to the loss). There may be no single global lr that works for all 16 layers simultaneously, which is exactly the symptom seen — total stagnation across the whole lr sweep.

The ReLU variant (`DeepLinearReLU`) shows the identical failure (`logs/12345297.out`), which makes sense — `nn.Linear`'s default init isn't ReLU-aware either (it doesn't apply the `gain=sqrt(2)` Kaiming would use for ReLU), so the same vanishing-signal problem applies there too, compounded by ReLU zeroing out roughly half the units on top of that.

## Why this isn't visible from the optimizer code itself

`methods/method_utils/optimizer.py::create_optimizer` correctly builds `torch.optim.SGD(params=model.parameters(), lr=..., weight_decay=...)` — there's no bug there; SGD is constructed correctly with the lr/weight_decay from config, no momentum is specified (defaults to 0), which is consistent with how `sgd-step*.yaml` configs are written (only `lr` and `weight_decay` keys; no `momentum` key). The optimizer logic itself is fine; the issue is entirely upstream, in model initialization.

## Decision

Fix the model's initialization rather than the optimizer config. Orthogonal init keeps every layer's singular values at 1, which directly stops the `(1/3)^depth` variance collapse described above, without adding any normalization layers that would compromise the "deep linear" architecture being studied. It's also a one-line, standard, well-understood init (not exotic), consistent with what the `deep_linear_saxe` config folder name already implies was intended.

## Steps

~~1. **Add orthogonal init to `DeepLinear.__init__`** in `models/DeepLinear.py`: after building `self.hidden` and `self.classifier`, loop over all `nn.Linear` submodules and apply `nn.init.orthogonal_(layer.weight)` (keep biases at their default zero init). Since `hidden_dim` (1024) differs from `input_dim`/`num_classes` for the first/last layers, `orthogonal_` still works on non-square matrices (it orthogonalizes the rows/columns of whichever dimension is smaller), so no special-casing is needed.~~
2. **Verify on one SGD config before relaunching the full sweep**: run a single short job (`deep_linear_1024_16layer`, `sgd-step0.01.yaml`, a few epochs) and confirm train loss moves off `ln 2` within the first several epochs, the way the AdamW run does.
3. **Re-run the SGD sweep** (`slurm_run_blobs_deep_linear.py` with `OPTIMS` as currently configured) once step 2 confirms the fix, for both `deep_linear_1024_16layer` and `deep_linear_relu_1024_16layer`.
4. **Treat pre-fix results as stale**: any existing snapshots/`exp-ablation` runs (AdamW included) used the old default init, so for apples-to-apples comparison either re-run AdamW too with the new init, or clearly separate old vs. new results (e.g. by archiving old `exp-ablation`/`snapshots` data, similar to the `temp-old-labels` move done earlier) rather than mixing init schemes within one analysis.
