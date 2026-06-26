# Dataset / Label Consistency Note

`save_labels.py` saves `{"train": y_train, "val": y_val}` to `labels/<dataset>.p` for use in analysis notebooks. The labels are collected with `shuffle=False`, so they are in dataset index order (0, 1, ..., N-1).

## Why the labels stay valid during training

Training uses `shuffle=True`, but that only reorders batches each epoch — it never changes the underlying `dataset[i]` mapping. So the labels file is a stable index → label lookup that remains correct throughout training.

## Conditions for consistency

The dataset is fully determined by (data config YAML + seed). Both `save_labels.py` and `main.py` must use:

1. The **same data config YAML** (`--data` argument)
2. The **same seed** — `save_labels.py` defaults to `--seed 16`; training also defaults to seed 16

For datasets with a random train/val split at construction time (e.g., `makeblobs`), the split is seeded via `config['seed']`, so the same seed guarantees the same split.

## Usage

Run `save_labels.py` once per dataset, then point analysis notebooks at the output file:

```bash
python save_labels.py --data configs/cifar3/data/cifar3.yaml
# -> labels/CIFAR3.p
```

Notebooks load the file and use `labels["train"][i]` to look up the true label for training sample `i`.
