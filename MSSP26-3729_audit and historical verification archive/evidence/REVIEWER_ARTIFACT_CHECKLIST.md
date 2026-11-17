# Reviewer artifact checklist

| Artifact | Required content | Local status |
|---|---|---|
| Fixed manifest | domain/class/trial/split/window boundaries and SHA-256 | A-D 300-window manifest generated and audited; E rows missing |
| Source selection | six candidate screens, validation ranking, selected epoch | MISSING for final 11.16 runs |
| Config identity | canonical JSON and SHA-256 | staged reference config present |
| Checkpoints | teacher/student weights for all declared seeds and methods | MISSING locally |
| Predictions | label, logits, prediction, class probabilities and immutable window identity | MISSING for final A-E runs |
| Per-run metrics | A/P/R/F1, per-class values, confusion matrix and supports | MISSING for final A-E runs |
| Ensemble metrics | fixed equal-weight pre-softmax logit average | MISSING for final A-E runs |
| Bootstrap | 2,000 replicate trial/run draws and interval outputs | MISSING locally |
| Chronology | config freeze, checkpoint freeze, E acquisition and first access | MISSING locally |
| Environment | package lock, Python/PyTorch/CUDA/GPU identity | partly available; final run export required |
