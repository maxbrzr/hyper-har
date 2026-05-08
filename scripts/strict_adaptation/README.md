# Strict Adaptation Experiments

This directory keeps the stricter subject-disjoint adaptation experiment isolated
from the original LOSO artifacts.

## Split Semantics

For each held-out test subject, the remaining subjects are split into disjoint
roles:

- `base_pretrain_subject_ids`: subjects used to train the frozen TinierHAR base
- `meta_train_subject_ids`: base-unseen subjects used to train the hypernetwork
- `meta_val_subject_ids`: base-unseen subjects used for checkpoint selection
- `test_subject_id`: final base-unseen and meta-unseen subject

The key difference from the original meta-LOSO setup is that the hypernetwork no
longer trains on subjects that were also used to pretrain the base model.

The base model's own validation split is window-level within
`base_pretrain_subject_ids` by default. That means `pretrain_train_subject_ids`
and `pretrain_val_subject_ids` can overlap, but both remain disjoint from
`meta_train_subject_ids`, `meta_val_subject_ids`, and `test_subject_id`.

## Run Order

First train strict base checkpoints:

```bash
uv run python scripts/strict_adaptation/pretrain.py
```

Then train the direct SetEncoder + HyperNet LoRA adapter predictor:

```bash
uv run python scripts/strict_adaptation/meta_train_direct.py
```

Artifacts are written under:

```text
artifacts/strict_adaptation/pretrain/
artifacts/strict_adaptation/meta_direct/
```
