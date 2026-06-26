# Plan: spring cleaning 2 — slim `create_diagnostics` call + drop `--wandb_project`

Two small follow-ups to the spring-cleaning refactor. Nothing implemented yet.
Branch `spring-cleaning`, no remote push. Decisions in `[[ ]]`, my answers in `{{ }}`.

---

## Item 1 — eliminate the `--wandb_project` CLI flag

> main.py:178-179 — the --wandb_project CLI flag overrides wandb.project if passed.
> [[Eliminate this flag. The config should determine the project]]

{{Agreed. The project belongs in the config's `wandb:` section. Concretely:}}

- Delete the arg definition (`main.py:143-144`).
- Delete the override (`main.py:178-179`), leaving:
  ```python
  # W&B init kwargs come from the config's wandb section (§4.1).
  wandb_kwargs = dict(config.get('wandb', {}))
  ```
- Update the comment on `main.py:176` to drop "CLI flags override".
- {{Grep confirms `--wandb_project` is not referenced by any `run_*.py`/slurm
  script or template, so no caller needs migrating. If a run wants a non-default
  project it sets `wandb.project` in its merged config (or the template). OK?}}

### Checklist (Item 1)
- ~~Remove `--wandb_project` argparse entry (`main.py:143-144`).~~
- ~~Remove the override branch (`main.py:178-179`) + fix comment.~~

---

## Item 2 — move the diagnostics-resources unpacking into `create_diagnostics`

> Read line 124 of SelectionMethod.py. I want this whole big dictionary unpacking
> to happen in the diagnostics code, not in selection method.
> ```python
> self.diagnostics = create_diagnostics(self, project_root=project_root **other_diagnostics_resources)
> ```
> where `other_diagnostics_resources` is currently empty `{}` but able to be received.
> The config is an attribute of `self` … Project root is all that isn't in the config.

{{Yes — your read is right. Every entry in today's `diagnostics_resources`
(SelectionMethod.py:99-122) is derivable from `self` *except* `project_root`,
which is computed from `__file__`. So we pass the method instance + `project_root`
and let `create_diagnostics` do the extraction. New signature:}}

```python
def create_diagnostics(method, *, project_root, **other_resources):
```

**`create_diagnostics` builds `resources` internally** (verbatim the dict that
lives in SelectionMethod today, sourced off `method`):

```python
diagnostics_config = method.config.get('diagnostics', {})
resources = {
    'save_dir':         method.config['save_dir'],
    'project_root':     project_root,
    'artifact_stem':    method.config['artifact_stem'],
    'dataset_name':     method.config['dataset']['name'],
    'model_name':       method.config['networks']['params'].get(
                            'm_type', method.config['networks']['type']),
    'seed':             method.config['seed'],
    'fixed_train_loader': method.fixed_train_loader,
    'test_loader':      method.test_loader,
    'total_batches':    len(method.train_loader),
    'num_train_samples': method.num_train_samples,
    'num_epochs':       method.epochs,
    'num_steps':        method.num_steps,
    'initial_best_acc': method.best_acc,
    'initial_best_epoch': method.best_epoch,
    'noisy_indices':    method.data_info.get('noisy_indices'),
    'true_labels':      method.data_info.get('true_labels'),
    'wstar_test_acc':   method.data_info.get('wstar_test_acc'),
    'what_test_acc':    method.data_info.get('what_test_acc'),
    'bayes_accuracy':   method.config.get('bayes_accuracy'),
    'num_classes':      method.num_classes,
    'config':           method.config,
    'logger':           method.logger,
    **other_resources,
} 
``` 
[[Yes, but for conciseness you can go ahead and extract the config as config = method.config or something]]

The rest of `create_diagnostics` (schedules, managers, `set_static_context`,
the leaf loop) is unchanged — it already consumes `resources` and
`diagnostics_config`.

**SelectionMethod (replaces lines 95-124)** becomes:

```python
# Diagnostics own their resource extraction; SelectionMethod just hands over
# itself + the one thing not in the config (the repo root).
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
other_diagnostics_resources = {}
from create_diagnostics import create_diagnostics
self.diagnostics = create_diagnostics(
    self,
    project_root=project_root,
    **other_diagnostics_resources,
)
```

This restores the two commented-out lines' values (`model_name`/`dataset_name`)
by computing them inside `create_diagnostics`, so the current uncommitted
`NameError` state is resolved by the move itself (lines 92-93 get deleted, not
un-commented).

### Open questions
- {{**project_root**: `create_diagnostics.py` lives at the repo root, so it could
  compute `os.path.dirname(os.path.abspath(__file__))` itself and we'd pass *nothing*
  but `method`. Cleaner call site, but then the "project root" definition lives in
  two places if anything else needs it. I lean toward **keeping your explicit
  `project_root=` kwarg** (one source of truth, matches your proposal). Keep it
  explicit, or compute it inside? [[ ]]}} [[Explicit]]
- {{**coupling direction**: this makes `create_diagnostics` depend on
  `method`'s attribute names (`method.epochs`, `method.fixed_train_loader`, …)
  instead of `method` depending on the resource keys. That's the trade you're
  asking for and it's fine — diagnostics-specific knowledge moves into the
  diagnostics module. Just flagging it's a swap, not a removal, of coupling. OK?}} [[Ok]]
- {{Keep `other_diagnostics_resources = {}` explicit in SelectionMethod (per your
  proposal) so subclasses have an obvious extension point, even though it's empty
  today? I'd keep it. [[keep]]}}

### Checklist (Item 2)
- ~~Change `create_diagnostics` signature to `(method, *, project_root, **other_resources)`
      and build `resources` from `method` inside it.~~ (uses `config = method.config` local per note)
- ~~Replace SelectionMethod.py:92-124 with the slim call (delete the dict +
      the two commented lines).~~
- ~~GPU smoke test on `configs-temp/makeblobs_smoke.yaml` (all leaves) + commit.~~

---

## Item 3 — make `_LossErrorLeaf`'s required class attrs abstract

The base `_LossErrorLeaf` (diagnostics.py:101-119) currently ships concrete
placeholder defaults (`loader_key="train"`, `label_source="loader"`,
`metric="loss"`, `log_key="train_loss"`) that every subclass overrides anyway.
A subclass that forgot one would fail silently/murkily (e.g. `metric=None`-style
mistakes pick the wrong branch). Make them `None` and fail loudly.

**Change (diagnostics.py:101-111):**
```python
class _LossErrorLeaf(Diagnostic):
    """Shared base for the mean-loss / mean-acc leaves. Subclasses MUST set all
    four class attrs below; ``log_key`` may also be overridden per-run via the
    config's diagnostics ``params`` (see Item 4)."""
    loader_key = None      # 'train' (fixed_train_loader) | 'val' (test_loader)
    label_source = None    # 'loader' (loader's own labels) | 'true' (clean labels)
    metric = None          # 'loss' (mean NLL) | 'acc' (1 - mean 0/1 error)
    log_key = None         # W&B metric name (the info dict key) for this leaf

    _REQUIRED = ("loader_key", "label_source", "metric", "log_key")

    def __init__(self, manager, builder, should_run=None, **params):
        # Item 4: config may override the displayed metric name.
        if params.get("log_key") is not None:
            self.log_key = params["log_key"]
        for attr in self._REQUIRED:
            if getattr(self, attr) is None:
                raise TypeError(
                    f"{type(self).__name__} must set class attr '{attr}' "
                    f"(it is still None on the abstract _LossErrorLeaf base)."
                )
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        self.dep = builder.build(PerSampleLossError, manager, builder,
                                 self.loader_key, self.label_source)
```
The six concrete subclasses (lines 122-138) are unchanged — they already set all
four. `_LossErrorLeaf` is never instantiated directly, so the guard only ever
fires on a genuinely incomplete new subclass.

### Checklist (Item 3)
- ~~Set the four base attrs to `None` + add the `_REQUIRED` guard in `__init__`.~~
- ~~Covered by the same makeblobs smoke test (exercises TrainLoss/TrainAcc/
      ValLoss/ValAcc/TrueLabel* leaves) — confirms no false positive.~~

---

## Item 4 — let the config override a leaf's `log_key` (W&B metric name)

Verified safe: `log_key` is **not** a cache key (cache = `TrainState`, base.py:76)
nor a dedup key (dedup = `__eq__`, base.py:184). For `_LossErrorLeaf` it's only
the dict key in the logged payload (= the W&B metric name) and the file-log key.
Nothing outside `diagnostics.py` references it. The one constraint: two
**simultaneously enabled** leaves must not resolve to the same `log_key` (their
`wandb.log` calls would clobber each other at the same step).

**Mechanism (no separate plumbing needed):** `create_diagnostics` already forwards
each diagnostic's `params` to the constructor as `**params`. The Item-3 snippet
above already consumes `params["log_key"]`. So enabling the override is *just* the
Item-3 change — a config can now say:
```yaml
diagnostics:
  diagnostics:
    TrainLoss: { params: { log_key: noisy_train_loss } }
    TrainAcc:  { params: { log_key: noisy_train_acc } }
```
and the train-loss/acc metrics log under those names instead.

### Scope decisions
Implemented with my stated leans (the `[[ ]]` were left blank and the user said
"implement all"):
- Override scoped to **`log_key` only**; `loader_key`/`label_source`/`metric` stay
  class-level (they change *what is computed*).
- **Collision guard added** in `create_diagnostics`: raises if two enabled
  `_LossErrorLeaf`s resolve to the same `log_key`.
- Scoped to the **`_LossErrorLeaf` family** (TrainLoss/TrainAcc/ValLoss/ValAcc/
  TrueLabel*); other leaves hardcode multiple keys and are out of scope.

### Checklist (Item 4)
- ~~(Covered by Item 3's `params["log_key"]` consumption — no extra code there.)~~
- ~~Collision guard in `create_diagnostics` for duplicate resolved
      `log_key`s among enabled `_LossErrorLeaf`s.~~
- ~~Add a `log_key` override to a noisy example/smoke config and confirm the
      renamed metric appears in the W&B dry-run.~~ (`makeblobs_smoke.yaml`:
      `TrainLoss`/`TrainAcc` → `noisy_train_loss`/`noisy_train_acc`; verified in
      `logs/TrainLoss.log`.)

---

## Notes
- All items are mechanical. Item 1 removes an override path. Item 2 has no behavior
  change (same `resources`, assembled elsewhere). Item 3 adds a guard with no effect
  on the existing complete subclasses. Item 4 is opt-in and a no-op for configs that
  don't set `log_key`.
