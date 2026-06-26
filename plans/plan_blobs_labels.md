# Plan: Generate Labels Files for MakeBlobs Dataset

## Context

`labels/` holds pre-saved train/val label arrays (e.g., `CIFAR3.p`, `MNIST.p`) produced by `save_labels.py`. These are currently used by noise-variant dataset configs (e.g., `makeblobs_noise.yaml` → `labels/MakeBlobs_noise.p`) so that the true clean labels are available at training time for diagnostics even after label corruption.

The `slurm_run_blobs_deep_linear.py` experiment does **not** use noise variants, so it does not currently require any labels files. However, if you want to use the noise pipeline for blobs, or run diagnostics that compare against true labels, you will need per-config label files.

## Key design issue: dataset seed

For CIFAR3/MNIST the dataset is a fixed canonical split, so one labels file covers all experiment seeds. **MakeBlobs is different**: the train/test split is generated via `sklearn.train_test_split` using `random_state=config['seed']`. The generated data configs (`configs/makeblobs/generated/data/`) do not set `dataset.random_state`, so the split seed comes from the top-level experiment seed (1, 2, or 3 in the current sweep).

This means a labels file saved with one seed will not match the data seen by a run with a different seed.

**Options:**

1. **Fix `dataset.random_state` in the generated data configs.** Add `random_state: <N>` under `dataset:` in `write_generated_configs()` in `slurm_run_blobs_deep_linear.py` (and in any future sweep scripts). Pick any fixed value (e.g., 42). Then `MakeBlobs` always generates the same split regardless of experiment seed, and one labels file per (dim, cscale) pair is sufficient.

2. **Save one labels file per (seed, dim, cscale) triple.** Run `save_labels.py` nine times (3 seeds × 3 dims × 2 center scales = 18 invocations for the current sweep). Pass matching `--seed` and `--output labels/MakeBlobs_d{dim}_cscale{cscale}_seed{s}.p`. The noise config would then need to be seed-specific, complicating the pipeline.

**Recommendation: Option 1.** It matches how all other datasets work (single labels file) and is the least invasive change.

## Steps (assuming Option 1)

~~1. **Update `write_generated_configs`** in `slurm_run_blobs_deep_linear.py` to inject `dataset.random_state: 42` into each generated data config. This makes the blob geometry fixed across experiment seeds.~~

~~2. **Regenerate the data configs** by re-running the config generation portion of `slurm_run_blobs_deep_linear.py` (or calling `write_generated_configs` directly). This overwrites the existing files in `configs/makeblobs/generated/data/`.~~
[[But the datasets would still be exactly the same as before, right? Since the random seed is the same?]]
{{1. No — the dataset would change for existing runs. The seed governs both `make_blobs` (cluster geometry and sample positions) and `train_test_split`. Currently there is no `random_state` in the data configs, so the loader falls back to `config['seed']` (1, 2, or 3 in the sweep). After this change it would always use 42. For any run whose experiment seed was not 42, both the blob geometry and the train/test split assignment will differ. That's expected and fine — the point is to decouple dataset identity from experiment seed — but it does mean any previously-saved results used a different dataset than the one produced by the updated configs.}}

3. **Run `save_labels.py` for each (dim, cscale) pair** — run with `online-bs-p100` conda env active (needs torchvision). All 11 existing generated configs were updated in step 2:

   ```bash
   for f in configs/makeblobs/generated/data/makeblobs_d*.yaml; do
     base=$(basename $f .yaml)
     suffix=${base#makeblobs_}
     python save_labels.py \
       --data "$f" \
       --optim configs/makeblobs/optim/adamw-320-0.001-0.01.yaml \
       --seed 42 \
       --dataset "MakeBlobs_${suffix}" \
       --output "labels/MakeBlobs_${suffix}.p"
   done
   ```

   Notes:
   - `--optim` is required to populate `training_opt.batch_size` (needed by `MakeBlobs` for `test_batch_size` fallback).
   - `--seed 42` matches the `random_state` baked into the configs in step 1.
   - Teacher files (`models/teacher/makeblobs_*`) already exist for all dims.

~~4. **Update noise data configs** (e.g., `configs/makeblobs/data/makeblobs_noise.yaml`) or create per-(dim,cscale) noise configs pointing to the correct `labels/MakeBlobs_d{dim}_cscale{cscale}.p`.~~

~~Steps 1–3 are prerequisites; step 4 is only needed when you actually want to run a noise-blobs experiment.~~ [[No need for step 4. I don't plan to use noise with MakeBlobs.]]

[[Would you summarize in the chat for me how the existing data configs (step 2) work? The ones for mnist and cifar?]]
{{2. CIFAR3 and MNIST data configs only contain static metadata: dataset name, root path, num_classes, image size, and input_dim. There is no `random_state` because those datasets come from canonical fixed splits downloaded from the source — the train/test partition is baked into the dataset itself, not computed at runtime. MakeBlobs is fundamentally different: it synthesizes both the data and the split on every loader call, which is why seeding matters there.}}