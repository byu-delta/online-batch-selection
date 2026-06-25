# Plan: Sweep over teacher noise alpha

## Background

`alpha` is the dimensionless noise level passed to `data/make_blobs_teacher.py`
(`noise_std = alpha / sqrt(n_features)`). It appears in the filenames of two
generated teacher artefacts:

- `models/teacher/makeblobs_{d}d_cscale{cs}_hyperplane_alpha{alpha}_nseed0.pth`
- `models/teacher/makeblobs_{d}d_cscale{cs}_wnoised_alpha{alpha}_nseed0.npy`

Both paths are currently hardcoded as `alpha1.0` in three places inside
`write_generated_configs` and the teacher-generation loop in
`slurm_run_blobs_deep_linear.py`. To sweep alpha we need to:

1. Add `ALPHAS` as a top-level sweep list. {{Done — adding inline comment: `# noise_std = alpha / sqrt(n_features)`}}
2. Thread `alpha` through every place that constructs or references those paths.
3. Include `alpha` in generated config and label filenames so configs don't collide.

No changes are needed outside `slurm_run_blobs_deep_linear.py` — the method
templates already hold `teacher_model_path: null` and the data template already
holds `wnoised_file: null`; the generator fills them in at runtime.

---

## Changes to `slurm_run_blobs_deep_linear.py`

### 1. Add `ALPHAS` sweep constant

```python
ALPHAS = [1.0]
```

Place it alongside `DIMS`, `CENTER_SCALES`, `N_SAMPLES`.

### 2. Thread `alpha` into the teacher-generation loop

```python
for dim, cscale, alpha in product(DIMS, CENTER_SCALES, ALPHAS):
    print(f"Generating geometry and teacher for dim={dim}, center_scale={cscale}, alpha={alpha}...")
    subprocess.run(
        [
            "python", "data/make_blobs_teacher.py",
            "--n_features", str(dim),
            "--center_scale", str(cscale),
            "--center_seed", "42",
            "--alpha", str(alpha),
            "--noise_seed", "0",
            "--out_dir", "models/teacher",
        ],
        check=True,
    )
```

### 3. Thread `alpha` into `write_generated_configs`

Change signature to `write_generated_configs(dims, center_scales, n_samples_list, alphas)`.

Inside the `for d, cs in product(dims, center_scales)` loop, add an outer loop
over `alphas`. Method configs only depend on `(d, cs, alpha)` (not `n`), so
generate them inside this loop:

```python
for d, cs, alpha in product(dims, center_scales, alphas):
    teacher_path = f"models/teacher/makeblobs_{d}d_cscale{cs}_hyperplane_alpha{alpha}_nseed0.pth"

    for name, tmpl in [
        (f"rholoss-0.1-hyperplane_d{d}_cscale{cs}_alpha{alpha}",  rholoss_tmpl),
        (f"bayesian-0.1-hyperplane_d{d}_cscale{cs}_alpha{alpha}", bayesian_tmpl),
    ]:
        cfg = copy.deepcopy(tmpl)
        cfg['teacher_model_path'] = teacher_path
        p = GEN_DIR / "method" / f"{name}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.dump(cfg))

    for n in n_samples_list:
        cfg = copy.deepcopy(data_tmpl)
        cfg['dataset']['n_samples']   = n
        cfg['dataset']['n_features']  = d
        cfg['dataset']['input_dim']   = [1, d]
        cfg['dataset']['random_state'] = 42
        cfg['dataset']['center_file'] = f"models/teacher/makeblobs_{d}d_cscale{cs}_centers_seed42.npy"
        cfg['dataset']['wstar_file']  = f"models/teacher/makeblobs_{d}d_cscale{cs}_wstar_seed42.npy"
        cfg['dataset']['wnoised_file'] = f"models/teacher/makeblobs_{d}d_cscale{cs}_wnoised_alpha{alpha}_nseed0.npy"
        cfg['bayes_accuracy'] = round(float(ndtr(cs)), 3)
        p = GEN_DIR / "data" / f"makeblobs_d{d}_cscale{cs}_n{n}_alpha{alpha}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.dump(cfg))
```

Call site becomes:
```python
write_generated_configs(DIMS, CENTER_SCALES, N_SAMPLES, ALPHAS)
```

### 4. Thread `alpha` into the labels-job loop

```python
labels_job_ids: dict[tuple, str] = {}  # (dim, cscale, n, alpha) -> slurm job id

for dim, cscale, n, alpha in product(DIMS, CENTER_SCALES, N_SAMPLES, ALPHAS):
    data_cfg = str(GEN_DIR / "data" / f"makeblobs_d{dim}_cscale{cscale}_n{n}_alpha{alpha}.yaml")
    out_path = f"labels/makeblobs_d{dim}_cscale{cscale}_n{n}_alpha{alpha}.p"
    cmd = [
        "python", "save_labels.py",
        "--data", data_cfg,
        "--output", out_path,
        "--overwrite",
    ]
    if USE_SLURM:
        result = subprocess.run(
            ["sbatch"],
            input=make_sbatch(cmd, f"labels_d{dim}_cs{cscale}_n{n}_a{alpha}", time="0:15:00"),
            text=True, check=True, capture_output=True,
        )
        labels_job_ids[(dim, cscale, n, alpha)] = result.stdout.strip().split()[-1]
    else:
        subprocess.run(cmd, check=True)
```

### 5. Thread `alpha` into the jobs list

The `METHODS_HYPERPLANE` method name now includes alpha in its filename, so the
jobs list must carry `alpha` and construct the method path accordingly:

```python
jobs = (
    [
        (
            seed, dim, cscale, n, alpha,
            str(GEN_DIR / "data"   / f"makeblobs_d{dim}_cscale{cscale}_n{n}_alpha{alpha}.yaml"),
            model, optim,
            str(GEN_DIR / "method" / f"{method_name}_d{dim}_cscale{cscale}_alpha{alpha}.yaml"),
        )
        for seed, dim, cscale, n, alpha, optim, method_name, model
        in product(SEEDS, DIMS, CENTER_SCALES, N_SAMPLES, ALPHAS, OPTIMS, METHODS_HYPERPLANE, MODEL_CONFIGS)
    ] + [
        (
            seed, dim, cscale, n, alpha,
            str(GEN_DIR / "data" / f"makeblobs_d{dim}_cscale{cscale}_n{n}_alpha{alpha}.yaml"),
            model, optim, method,
        )
        for seed, dim, cscale, n, alpha, optim, method, model
        in product(SEEDS, DIMS, CENTER_SCALES, N_SAMPLES, ALPHAS, OPTIMS, METHODS_FIXED, MODEL_CONFIGS)
    ]
)
```

### 6. Update the submission loop

Unpack `alpha` and pass it to the dependency lookup:

```python
for seed, dim, cscale, n, alpha, data, model, optim, method in tqdm(jobs, ...):
    ...
    dep = labels_job_ids.get((dim, cscale, n, alpha))
    ...
```

---

## Summary of changes

- ~~Add `ALPHAS = [1.0]` constant~~
- ~~Thread `alpha` into teacher-generation loop (`--alpha str(alpha)`)~~
- ~~Add `alphas` parameter to `write_generated_configs`; loop over `(d, cs, alpha)`~~
- ~~Use `alpha{alpha}` in method config names and `wnoised_file`/`teacher_path` strings~~
- ~~Use `alpha{alpha}` in generated data config filenames~~
- ~~Thread `alpha` into labels loop; update labels filename and job ID key~~
- ~~Thread `alpha` into jobs list and submission loop~~
- ~~Update dependency lookup key to `(dim, cscale, n, alpha)`~~
