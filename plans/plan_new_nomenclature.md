Big refactor. All runs should have a unique identifier—perhaps a timestamp and a hash. Any filename collisions should cause an exception to be raised. Currently dealing with filepath collisions because all runs go to `./exp/`, and because `snapshots` names are nonsense. Also when we save points with `save_labels.py` there are overwrites. Again, everything should have a hash or something unique. Also, all outputs of a run should lie in a particular directory. I propose that all outputs of this code lie in `./experiments`, and within `experiments/`, each run gets its own timestamp-and-hash directory, like `./experiments/run_20260625_131500_qrzff9e/`, and inside that there is a directory for W&B, another for logs, another for snapshots, etc. Everything in there.

[[I know that there is caching functionality for labels and such. Is this plan totally incompatible with that? Maybe labels could be logged in both places or something? Maybe with symlinks? Put the actual file in another folder _with a unique identifier of some kind_ and a symlink in the run directory. Suggest other ideas if you have them!]]

{{Not incompatible, but the cache fundamentally pulls against this plan's "collision = exception" rule and must be exempted from it. A cache exists precisely so a value is computed once and *shared across runs* — i.e. deliberate filename reuse. If every run owned its own copy under `experiments/run_.../`, you'd lose the sharing that makes the cache worth having, and a literal collision rule would throw on the first cache hit.

Resolution — content-addressed shared store + symlink, as you suggested:
- Put the real artifact in a shared, git-ignored store keyed by a hash of its *inputs*, not a per-run id: `./cache/labels/<dataset-content-hash>.pt`. The hash is over the things that determine the labels (dataset config, split, seed, transform), so identical inputs deterministically map to the same file.
- A cache **hit** (file already exists for that hash) is a no-op *reuse*, never a "collision." Only *unkeyed* run outputs (the run dir, logs, snapshots) get collision-detection. State the exemption explicitly in the plan, or rules #1 (collisions raise) and the cache will contradict each other.
- In the run dir, drop a symlink: `experiments/run_.../labels -> ../../cache/labels/<hash>.pt`. The run is self-contained for browsing, but storage is shared and `save_labels.py` stops overwriting (it writes `<hash>.pt` and is idempotent).

[[Honestly, the names don't need to be hashed in the cache, I'd like them to be readable. But, if there is an attempt to overwrite a file, raise an exception loudly.]]

{{Agreed, and readable names work — but note what the "raise loudly on overwrite" rule forces, because it's subtle. With a hash, *file exists ⇒ cache hit ⇒ reuse* is unambiguous. With a readable name, existence is ambiguous: it could be a legitimate hit (same inputs) OR a genuine collision (different inputs that happen to produce the same readable name, because the name didn't capture some input). So you cannot just "raise on every existing file" — that would crash every legitimate rerun and destroy the cache. And you can't just "reuse every existing file" — that silently serves stale/wrong labels when the name is incomplete.

The clean rule that satisfies both your asks (readable + loud-on-overwrite):
- The readable name must encode **every input that determines the content** (e.g. `cifar3_train_seed42_labels.pt` — dataset, split, seed, and any transform that affects labels). This is the load-bearing requirement; if the name omits a determining input, two different artifacts map to one name.
- Store a tiny **sidecar** next to it recording a hash of the full inputs: `cifar3_train_seed42_labels.pt` + `cifar3_train_seed42_labels.meta` (the input hash + a human-readable dump of the inputs). [[Like, _all_ the parameters in that context, or just those that determined the .pt file?]]

{{Only the ones that *determine* the .pt content — emphatically not all the parameters in context. This is the whole game:
- Hash too *few* inputs → false cache hits: you change something that actually affects the labels but the hash doesn't see it, so you silently reuse stale/wrong data. (The danger you're guarding against.)
- Hash too *many* inputs → false misses: the labels don't depend on `lr` or model architecture, but if those are in the hash, every sweep over them recomputes identical labels and the cache never shares. (Defeats the point of caching.)

So the determining set for label caching is narrow: dataset identity, split, seed, and any transform/preprocessing that changes the labels themselves. Not the optimizer, not the model, not the method.

Two refinements:
1. Make the determining set an **explicit, auditable whitelist** per cache type, not "auto-hash whatever dict is lying around." Hidden auto-hashing of the ambient config is exactly how you drift into false misses when someone adds an unrelated field. A named list (`LABEL_CACHE_KEYS = [...]`) is greppable and reviewable.
2. The **hash** covers only the determining subset; the **human-readable dump** in the sidecar can be richer (dump more context for debugging). Keep those two roles separate — integrity check vs. debugging breadcrumb.}}
- On write: if the file doesn't exist → write both. If it exists → read the sidecar hash. **Match → reuse, skip the write** (the normal cache-hit path, no exception). **Mismatch → raise loudly** — this is exactly the "attempt to overwrite" you want to catch, and it means a name failed to capture an input (a bug), not a benign rerun.

So: readable primary names for browsability, sidecar hash for the integrity check. You get human-readable cache files, idempotent reruns, AND a loud failure the instant two different things claim the same name. Revises caveat 2 below — the cache no longer *needs* hashed filenames, only a hashed integrity check.}}

Two caveats:
{{You're right to push — if the filename *losslessly* encodes every determining parameter, the hash is redundant: a name collision already implies identical inputs, so existence = hit and a same-name-different-content collision is impossible by construction. The hash only earns its place because "readable filename" and "lossless encoding of all determining params" are in tension:

- Some determining inputs don't fit in a readable name. A seed and a dataset name do. But a transform pipeline, a list of noisy indices, or a float like `0.30000000004` either bloats the name into something unreadable or gets *abbreviated/rounded* — and the moment you abbreviate for readability, the name is no longer injective. Two different inputs can now land on the same readable name.
- The hash is the price of readability. It's the lossless backstop that lets the *name* stay short and human while still detecting the "same name, different inputs" case — which is exactly the loud-overwrite failure you asked for.

So it's a genuine fork, pick one:
- **(A) Name encodes everything, no hash.** Works only if every determining param is a short scalar. Names get long/ugly as soon as one isn't, and you must be disciplined to include *all* of them. No sidecar needed.
- **(B) Readable (abbreviated) name + sidecar hash.** Names stay legible; the hash catches abbreviation/rounding collisions and powers the loud-on-overwrite guarantee.

What you *can't* have is "fully readable names" + "no hash" + "loud on genuine collision" at the same time — unless every determining input is a short scalar. If label inputs really are all short scalars (dataset, split, seed), go with (A) and drop the hash. If any aren't, (B). My bet is labels are simple enough for (A), and the hash is overkill *there* — but the run-dir/config side likely isn't, so keep the mechanism in mind for those.}}

1. Symlinks don't survive `tar`/`rsync`/`scp` cleanly by default and don't exist on some filesystems — if you ever archive a single run dir to move it off the supercomputer, the label link dangles. If that matters, record the cache *hash* in the run's `config.yaml` (so the link is reconstructible) or hardlink instead of symlink (same filesystem only).
2. Decide the hash basis now — it's the same decision the run-dir hash needs (see my chat notes): content-hash (reproducible, dedups reruns) vs random (always unique). Caches *must* be content-hash to work; run dirs can be either.}}

I also want to have "config templates". Instead of having tons of configs that you mix and match together, I want a run to use a single config file. The single config file should essetially be the concatenation of all the existing configs ("data", "method", "diagnostics", etc.) To facilitate varying hyperparameters in the slurm submission scripts, I want functionality for config templates as follows:
- If a key in the config is null, the template generator should expect it to be passed. Otherwise raise Exception
- If a key in the config is _not_ null, and the template generator is passed that key, raise Exception.
Example:
```python
config = config.yaml
...
params_to_vary = dict(
    lr=[1e-3, 1e-2, 1e-1],
    model=['LeNet', 'ResNet']
)
generate_configs(config, params_to_vary)
```
These configs should be saved to a temp dir that is git-ignored (`./configs-temp`) for use in the tests.
