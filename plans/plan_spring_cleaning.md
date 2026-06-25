# Plan: Spring Cleaning

A merged refactor combining two efforts:
1. **New diagnostics system** (was `plan_new_diagnostics_3.md`) — a dependency-aware, cached, self-registering `Diagnostic` framework.
2. **New nomenclature** (was `plan_new_nomenclature.md`) — unique run identities, a single output tree under `./experiments/`, a readable label cache, and single-file config templates.

This document is the implementation-ready spec. The two source plans remain for the design discussion (the `[[ ]]`/`{{ }}` notes); this plan is what we build from.

---

## 0. Guiding Principles

- **Every run is self-contained.** All outputs of a run live under one directory: `./experiments/run_<timestamp>_<hash>/`.
- **Collisions are bugs, and bugs are loud.** Any attempt to overwrite an existing output file raises an exception. The *only* exception is the cache (see §3), where a verified hit is a deliberate, silent reuse.
- **One config per run.** A run is driven by a single merged config file, not four mix-and-match files. Sweeps are generated from templates.
- **No new heavyweight dependencies.** Config templating is hand-rolled on `pyyaml`; we do not adopt OmegaConf/Hydra.

---

## 1. Directory Layout

```
./experiments/                         # all run outputs (git-ignored)
    run_20260625_131500_qrzff9e/
        config.yaml                    # the exact resolved config this run used
        wandb/                         # W&B run files (wandb.init(dir=...))
        logs/                          # file_log outputs from diagnostics
        snapshots/                     # checkpoints, spectral snapshots, NTK, etc.
        labels -> ../../cache/labels/cifar3_train_seed42_labels.pt   # symlink (see §3)
./cache/                               # shared, cross-run cache (git-ignored)
    labels/
        cifar3_train_seed42_labels.pt
./configs-temp/                        # generated sweep configs (git-ignored)
./configs/                             # hand-written single-file configs / templates
```

Add `experiments/`, `cache/`, and `configs-temp/` to `.gitignore`.

---

## 2. Run Identity & Collision Rules

- Each run gets an id: `run_<timestamp>_<hash>`, e.g. `run_20260625_131500_qrzff9e`.
    - `timestamp`: `YYYYMMDD_HHMMSS`.
    - `hash`: short **random** token (not content-derived). Rationale: parallel SLURM array jobs and intentional reruns must never collide; a random token guarantees uniqueness, and reruns of an identical config are allowed. (Content-hashing would block legitimate reruns under the collision rule.)
- A central helper creates the run directory and all its subdirs **once**, at startup, and returns the paths. If the directory already exists → raise (effectively impossible with a random hash, but the rule is enforced).
- A small write-guard utility is used for all run outputs: *if the target path exists, raise.* This enforces "collisions are bugs" uniformly across logs, snapshots, and config snapshots.
- The resolved config actually used by the run is written to `experiments/run_.../config.yaml` at startup. This is the durable record (W&B also logs it).

---

## 3. Label Cache

Decision: **readable names, no hash, loud on overwrite** (option A from the source notes — label inputs are all short scalars, so the filename can losslessly encode them).

- Cache lives in `./cache/labels/`, shared across runs.
- Filename **losslessly encodes every determining input**: dataset, split, seed, and any transform that changes the labels. E.g. `cifar3_train_seed42_labels.pt`. The determining inputs are an explicit, auditable list in code (e.g. `LABEL_CACHE_KEYS`), not "whatever is in the config dict."
- **Read path:** if the file exists → reuse it silently (this is the cache hit; it is *not* a collision). If absent → compute and write.
- **Write path:** writing uses the §2 write-guard semantics with one twist — `save_labels.py` first checks existence; on a hit it skips the write. It must **never** silently overwrite an existing cache file. If code paths ever attempt to write *different* content to an existing name, that's a naming bug → raise loudly.
- In each run dir, create a **symlink** `experiments/run_.../labels -> ../../cache/labels/<name>.pt` so the run is browsable as self-contained while storage stays shared and `save_labels.py` stops overwriting.
- Caveat (documented, not blocking): symlinks don't survive `tar`/`scp` cleanly. If a run dir is ever archived off-cluster, the link dangles. Mitigation: the cache filename is fully determined by inputs recorded in `config.yaml`, so the link is reconstructible; or hardlink if same-filesystem archival is needed.
- Future note: if a cached artifact ever has inputs that *aren't* short scalars (a transform pipeline, an index list), switch that cache to option B (readable abbreviated name + sidecar `.meta` input-hash, reuse-on-match / raise-on-mismatch). Not needed for labels.

---

## 4. Config System

### 4.1 Single merged config

- A run is driven by **one** YAML file: the concatenation of what used to be the `method`, `data`, `model`, `optim`, and `diagnostics` configs, nested under top-level sections:

```yaml
data:    { ... }
model:   { ... }
optim:   { ... }
method:  { ... }
diagnostics: { ... }   # consumed by create_diagnostics.py (see §5.7)
```

- `main.py` loads this one file. (The old four-`--`flag interface is replaced; see §7 for the migration of `create_diagnostics.py` to read the `diagnostics:` subtree instead of its own file.)

### 4.2 Config templates & generation

A template is a single config with some leaf values set to the sentinel `__REQUIRED__`, marking values that must be supplied at generation time.

`generate_configs(template_path, params_to_vary)`:

- `params_to_vary`: `dict[str, list]` keyed by **dotted paths** into the config, e.g.
  ```python
  params_to_vary = dict(
      optim__lr=[1e-3, 1e-2, 1e-1],   # written as "optim.lr"
      model__name=['LeNet', 'ResNet'],
  )
  generate_configs("configs/cifar3_base.yaml", {"optim.lr": [1e-3,1e-2,1e-1],
                                                 "model.name": ["LeNet","ResNet"]})
  ```
- **Validation rules (all raise on violation):**
    1. Every dotted key in `params_to_vary` **must exist** in the template (typo protection).
    2. Every key in `params_to_vary` **must currently be `__REQUIRED__`** in the template. Passing a key that already holds a concrete value → raise. (Original rule: "if a key is not null and is passed, raise.")
    3. Every `__REQUIRED__` leaf in the template **must be covered** by `params_to_vary`. An unfilled `__REQUIRED__` after generation → raise. (Original rule: "if a key is null, the generator must be passed it.")
- **Expansion:** full **Cartesian product** over the value lists.
- **Output naming:** each generated file is named `<template_stem>_<k1><v1>_<k2><v2>...yaml`, i.e. the template's name followed by an underscore-separated list of the changing parameters and their values (dotted keys sanitized for filesystem safety). E.g. `cifar3_base_lr0.01_nameResNet.yaml`.
- **Output location:** `./configs-temp/` (git-ignored), for use by the SLURM submission scripts and tests.
- **Collision check:** generated filenames go through the §2 write-guard — if a target already exists, raise.
- The config a run ultimately uses is **also** snapshotted into its `experiments/run_.../config.yaml` (§2) so W&B and the run dir record exactly what ran.

---

## 5. Diagnostics System

(Carried over from `plan_new_diagnostics_3.md`, integrated with §1–§4.)

### 5.1 `TrainState` (dataclass)
- `epoch: int`, `batch_idx: int`, `total_epochs: int`, `total_batches: int`, `total_steps: int`

### 5.2 `DiagnosticInfo` (dataclass)
- `name: str`, `info: Any`

### 5.3 `Diagnostic` (partially abstract)

**Attributes**
- `manager: DiagnosticsManager` — given in constructor; auto-registers self with manager.
- `log_path: str | None` — given in constructor; defaults to `None`; `file_log` is a no-op if unset. **Composed from the run dir** (§7): top-level diagnostics receive `run_dir/logs/...`.
- `should_run: Callable -> bool` — given in constructor; defaults to `lambda: True`.
- `last_run_state: TrainState | None` — initially `None`.
- `last_run_diagnostic: DiagnosticInfo | None` — initially `None`.

**Methods**
- `get_state() -> TrainState` — returns `manager.current_state`.
- `get_context() -> dict` — returns `manager.shared_context`.
- `_run() -> DiagnosticInfo` — **abstract.** Takes no args. Child calls `dep.run()` for each dependency first, then computes and returns a `DiagnosticInfo`. `dep.run()` caches by state, so calling it here is cheap if the dep already ran this step.
- `run() -> DiagnosticInfo` — if `get_state() == last_run_state`, returns cached `last_run_diagnostic`; otherwise calls `_run()`, updates `last_run_state`/`last_run_diagnostic`, returns result. Cache is keyed on state only — diagnostics must not depend on per-call args that vary independently of state; pass such values through `shared_context`.
- `conditional_run()` — calls `should_run(get_state())`; if `True`, calls `run()`. Used by the manager.
- `wandb_log(infos: List[DiagnosticInfo])` — logs `last_run_diagnostic` with necessary elements of `last_run_state` to W&B.
- `file_log(infos: List[DiagnosticInfo])` — logs `last_run_diagnostic` with necessary elements of `last_run_state` to `log_path`.
- `log()` — calls `wandb_log([last_run_diagnostic])` and `file_log([last_run_diagnostic])`.
- `__eq__` — raises `NotImplementedError` by default; children must override if they may be deduplicated in `create_diagnostics.py`.

### 5.4 `DiagnosticsManager`

**Attributes**
- `diagnostics: List[Diagnostic]` — top-level diagnostics only; dependency diagnostics are not registered here.
- `current_state: TrainState`
- `should_run: bool` — master kill-switch for all diagnostics in this manager.
- `shared_context: dict`

**Methods**
- `_update_state(state)` — sets `current_state`.
- `_update_shared_context(**kwargs)` — merges kwargs into `shared_context`.
- `run_diagnostics(state, *, **kwargs)` — runs `_update_state`, `_update_shared_context`, then `conditional_run()` on each registered diagnostic, then `_log_diagnostics()`.
- `_log_diagnostics()` — calls `log()` on each registered diagnostic.

*Multiple managers exist for different training phases (pre-batch, post-batch, validation, etc.).*

### 5.5 `DiagnosticsBuilder`

**Attributes**
- `all_diagnostics: defaultdict(list)` — keyed by diagnostic class; values are instance lists.

**Methods**
```python
def fetch_duplicate_diagnostic(self, diagnostic) -> Diagnostic | None:
    matches = [x for x in self.all_diagnostics[type(diagnostic)] if x == diagnostic]
    if not matches:
        return None
    elif len(matches) > 1:
        raise ValueError(f"Multiple identical diagnostics of type {type(diagnostic).__name__}")
    else:
        return matches[0]

def create_diagnostic(self, diagnostic_class, *args, **kwargs) -> Diagnostic:
    new_diagnostic = diagnostic_class(*args, **kwargs)
    duplicate = self.fetch_duplicate_diagnostic(new_diagnostic)
    if duplicate:
        return duplicate
    self.all_diagnostics[diagnostic_class].append(new_diagnostic)
    return new_diagnostic
```

### 5.6 `create_diagnostics.py`
- Initializes a `DiagnosticsBuilder` and the appropriate `DiagnosticsManager`s.
- **Reads the `diagnostics:` subtree of the single merged config** (§4.1), *not* a standalone diagnostics file. Keys are `Diagnostic` class names in scope.
- Calls `DiagnosticsBuilder.create_diagnostic()` per top-level diagnostic; shared dependencies built internally are deduplicated via `__eq__`.
- Registers top-level diagnostics with the appropriate manager; dependency diagnostics are not added to any manager's list.
- Receives the run dir (§7) and composes each top-level diagnostic's `log_path` under `run_dir/logs/` (and snapshot paths under `run_dir/snapshots/`).

### 5.7 Diagnostics config (the `diagnostics:` section)
```yaml
diagnostics:
    logging_defaults:
        log_interval: logarithmic
        save_init: 5
        save_freq: 4
    diagnostics:
        GradNorm:                  # example
            logging:               # optional per-diagnostic override of defaults
                log_interval: ...
            params:
                norm: 2            # kwargs passed to GradNorm's constructor
```

---

## 6. Reconciliation Points (where the two plans touch)

These are the seams that must be built deliberately or the merge silently fails:

1. **Cache vs. collision rule.** The cache (§3) is explicitly exempt from "collisions raise": a verified hit is silent reuse. Only unkeyed run outputs get the write-guard. Both rules coexist only because they're scoped this way.
2. **Config layer ownership.** `create_diagnostics.py` must read the `diagnostics:` subtree (§5.6), not its own file. Build the single-config format (§4) **before** wiring diagnostics, or diagnostics breaks when the merge lands.
3. **Run-dir injection.** The run dir (§2) must be threaded into the diagnostics builder/manager so `log_path` and snapshot paths land under `experiments/run_.../`. Without this, diagnostics scatter output outside the run tree — defeating the whole nomenclature goal.
4. **W&B dir.** `wandb.init(dir=run_dir/wandb)` must be set at startup so W&B files live under the run dir.

---

## 7. Implementation Phases

Ordered so each phase leaves the repo runnable.

**Phase 1 — Output tree & run identity (§1, §2)**
- [ ] Add `experiments/`, `cache/`, `configs-temp/` to `.gitignore`.
- [ ] Run-id + run-dir creation helper (timestamp + random hash, subdirs, raise-on-exists).
- [ ] Write-guard utility (raise if target exists) used by all run-output writes.
- [ ] Point W&B (`wandb.init(dir=...)`) and run-config snapshot at the run dir.

**Phase 2 — Label cache (§3)**
- [ ] `LABEL_CACHE_KEYS` explicit determining-input list + readable filename builder.
- [ ] `save_labels.py`: read-hit reuses silently; write never overwrites (raise on different content for same name).
- [ ] Symlink the cache file into the run dir.

**Phase 3 — Single merged config (§4.1)**
- [ ] `main.py` loads one merged YAML with `data/model/optim/method/diagnostics` sections.
- [ ] Convert existing four-file configs into merged single files.

**Phase 4 — Config templates (§4.2)**
- [ ] `__REQUIRED__` sentinel + the three validation rules (raise on each).
- [ ] `generate_configs` with dotted keys, Cartesian product, `<template>_<k><v>...` naming, output to `configs-temp/`, write-guard.
- [ ] Update SLURM submission scripts to generate + consume templated configs.

**Phase 5 — Diagnostics framework (§5)**
- [ ] `TrainState`, `DiagnosticInfo`, `Diagnostic`, `DiagnosticsManager`, `DiagnosticsBuilder`.
- [ ] `create_diagnostics.py` reading the `diagnostics:` subtree and receiving the run dir.
- [ ] Port existing diagnostics (spectral snapshots, NTK, linear probes, checkpointing) onto the new `Diagnostic` base with `__eq__` where dedup is possible.

**Phase 6 — Wiring & cleanup (§6)**
- [ ] Thread run dir into the diagnostics builder/manager; compose `log_path`/snapshot paths under it.
- [ ] Remove the old `./exp/` path scheme and the standalone diagnostics-config plumbing.

---

## 8. Open Questions

- {{Naming collision between template params: if two different dotted keys share a leaf name (e.g. `optim.lr` and `sched.lr`), the `<k><v>` filename fragment should use the sanitized *full dotted path*, not just the leaf, to stay unique. Assuming yes unless you object.}}
- {{Should Phase 3 (single config) land before or after Phase 5 (diagnostics)? Spec'd as before, since §6.2 requires it. If you want diagnostics built first against the old config, say so and I'll reorder — but that incurs rework.}}
