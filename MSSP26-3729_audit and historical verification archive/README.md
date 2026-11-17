# PG-MARL manuscript-aligned reference implementation

This repository is the code-facing companion to the 16 November 2026 revised
manuscript.  It implements the stated source-only A/B/C development protocol,
the offline Multi-Mechanism Physics Surrogate (MMPS), the deployed
Topology-Aware Spatiotemporal Network (TASN), and Task-Aligned
Meta-Distillation (TAMD).

## What this package matches

- fixed, non-overlapping 2.0 s windows (300 windows per complete 600 s trial);
- 100 Hz internal/IMU sequences with 200 recurrent states per window;
- effective 1 kHz cleaned vibration and 128 samples per motor revolution for
  order-domain features;
- source-only fitting, normalization, validation, checkpoint selection and
  hyperparameter selection on trajectories A/B/C;
- trajectories D and E are registered only by the final evaluator;
- three stacked four-head GATv2 layers on the six-joint mechanical chain;
- a standard GRU-form recurrent unit followed by recurrent-output LayerNorm;
- global mean pooling into a 64-dimensional student embedding;
- MMPS reference embeddings/probabilities, feature MSE, reliability-weighted
  teacher-to-student KL on support only, conditional multi-kernel MMD, and a
  three-step source-only inner/outer episode;
- five declared seeds: 42, 52, 62, 72 and 82.

The deployed TASN does **not** contain the experimental v6-v14 residual,
RV-band, order-spectrum, or safe-residual branches found in the historical
archive.

## Important provenance limitation

The local workstation snapshot used to assemble this repository contains the
historical A-D dataset and early 152-window D evaluations.  It does not contain
the final trajectory-E records, the five final 11.16 checkpoints, or the full
per-window A-E prediction exports supporting the revised numerical tables.
Consequently, the values transcribed into `evidence/manuscript_claims/` are
manuscript claims, not regenerated outputs of this package.  See
`evidence/PROVENANCE_GAPS.md` before public release or reviewer delivery.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

PyTorch and PyG wheels are platform specific.  If the generic install fails,
install the matching CUDA/CPU wheels from their official instructions first.

## Data layout

The raw root is expected to contain directories named
`Domain_A_Trajectory` through `Domain_E_Trajectory`, followed by the existing
health-state and `trial_###` folders.  Raw records are intentionally excluded
from this public-code package.  Generate and audit a fixed manifest first:

```bash
python scripts/build_fixed_manifest.py \
  --data-root /path/to/Borunte_1510A_Dataset \
  --output artifacts/window_manifest.csv

python scripts/verify_source_only_protocol.py \
  --manifest artifacts/window_manifest.csv
```

The manifest builder only records fixed time boundaries.  `train.py` expects
preprocessed `.npz` window records containing the arrays documented in
`docs/DATA_SCHEMA.md`.  The adapter in `scripts/prepare_fixed_windows.py`
shows how the retained BORUNTE CSV reader is used to generate those records.

## Source-only training

```bash
python train.py --config configs/manuscript_11_16.json \
  --manifest artifacts/window_manifest.csv \
  --records-root Borunte_1510A_Dataset \
  --seed 42
```

Run all five seeds before any target evaluation.  The training executable
rejects D/E manifest rows and records source-validation checkpoint metadata.

## Final frozen evaluation

```bash
python evaluate_frozen.py \
  --config configs/manuscript_11_16.json \
  --manifest artifacts/window_manifest.csv \
  --records-root Borunte_1510A_Dataset \
  --checkpoints checkpoints/pgmarl_manuscript_seed42.pt \
                checkpoints/pgmarl_manuscript_seed52.pt \
                checkpoints/pgmarl_manuscript_seed62.pt \
                checkpoints/pgmarl_manuscript_seed72.pt \
                checkpoints/pgmarl_manuscript_seed82.pt \
  --targets D E --acknowledge-final-target-evaluation
```

This evaluator exports per-window predictions, per-run confusion matrices,
run metrics, fixed equal-weight logit-ensemble metrics, and SHA-256 identities.
It does not tune, calibrate, or rank checkpoints on D/E.

## Repository map

- `pgmarl/`: models, physics descriptors, TAMD losses, protocol and metrics;
- `configs/`: the manuscript-aligned constants and declared seeds;
- `scripts/`: manifest, preprocessing, audit and artifact-export utilities;
- `evidence/`: manuscript claims, reviewer artifact checklist and provenance
  gaps;
- `tests/`: synthetic shape/protocol tests that do not use target data.

## Public release checklist

Before publishing, add the actual license selected by the authors, populate the
final A-E artifacts listed in `evidence/REVIEWER_ARTIFACT_CHECKLIST.md`, run
`python scripts/verify_release.py`, and replace all `MISSING` entries in the
provenance ledger with immutable file hashes and timestamps.
