# Reviewer-facing code scope

| Reviewer requirement | Code location |
|---|---|
| Fixed 2.0-s non-overlapping windows and trial-disjoint splits | `pgmarl/protocol.py`, `scripts/build_fixed_manifest.py` |
| A/B/C-only training, validation and checkpoint selection | `train.py`, `scripts/verify_source_only_protocol.py` |
| MMPS mechanics, residual, friction and order descriptors | `pgmarl/physics.py`, `pgmarl/models.py` |
| Three-layer four-head GATv2, GRU recurrence and LayerNorm | `pgmarl/models.py` |
| TAMD feature/probability transfer and conditional MMD | `pgmarl/losses.py`, `train.py` |
| Five fixed seeds and source-validation selection metadata | `configs/manuscript_11_16.json`, `train.py` |
| D/E registered only after freezing | `evaluate_frozen.py` |
| Per-window logits, predictions, metrics and checkpoint hashes | `evaluate_frozen.py` |

This package contains code only.  Numerical results, raw data, processed data,
checkpoints and prediction exports are separate release artifacts.
