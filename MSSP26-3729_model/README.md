# PG-MARL

This package contains the model architecture and executable protocol guards
corresponding to `Revised Manuscript (unmarkered) - 16改.docx`.

Included components:

- MMPS motor-side residual, friction and RV order-domain descriptors;
- the source-trained MMPS fusion network and peak-preserving residual encoder;
- the three-layer, four-head GATv2 TASN;
- the 200-step GRU-form recurrent update with post-update LayerNorm;
- TAMD support/query losses, reliability-weighted probability distillation and
  conditional multi-kernel MMD;
- source-only A/B/C training, validation and checkpoint selection guards;
- frozen D/E evaluation with five predeclared seeds and fixed equal-weight
  pre-softmax logit aggregation;
- immutable-window manifest construction and source/target isolation audit.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Audit the fixed split

```bash
python scripts/build_fixed_manifest.py \
  --data-root /path/to/Borunte_1510A_Dataset \
  --output artifacts/window_manifest.csv

python scripts/verify_source_only_protocol.py \
  --manifest artifacts/window_manifest.csv
```

## Train one declared seed

```bash
python train.py \
  --config configs/manuscript_11_16.json \
  --manifest artifacts/window_manifest.csv \
  --records-root /path/to/prepared-record-root \
  --seed 42
```

## Frozen target evaluation

```bash
python evaluate_frozen.py \
  --config configs/manuscript_11_16.json \
  --manifest artifacts/window_manifest.csv \
  --records-root /path/to/prepared-record-root \
  --checkpoints checkpoints/pgmarl_manuscript_seed42.pt \
                checkpoints/pgmarl_manuscript_seed52.pt \
                checkpoints/pgmarl_manuscript_seed62.pt \
                checkpoints/pgmarl_manuscript_seed72.pt \
                checkpoints/pgmarl_manuscript_seed82.pt \
  --targets D E \
  --acknowledge-final-target-evaluation
```

The target evaluator has no interface for target calibration, seed screening,
checkpoint reselection or learned ensemble weighting.
