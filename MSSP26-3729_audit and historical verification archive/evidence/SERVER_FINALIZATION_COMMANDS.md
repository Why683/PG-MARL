# Server-side finalization commands

Run these commands from the unpacked manuscript-aligned repository in the
existing `py38` environment. They do not authorize target-domain tuning.

```bash
pip install -r requirements.txt
pip install -e .

python scripts/build_fixed_manifest.py \
  --data-root Borunte_1510A_Dataset \
  --output artifacts/window_manifest.csv

python scripts/verify_source_only_protocol.py \
  --manifest artifacts/window_manifest.csv

python -m unittest discover -s tests -v

for seed in 42 52 62 72 82; do
  python -u train.py \
    --config configs/manuscript_11_16.json \
    --manifest artifacts/window_manifest.csv \
    --records-root Borunte_1510A_Dataset \
    --seed "$seed" \
    --output checkpoints
done

python -u evaluate_frozen.py \
  --config configs/manuscript_11_16.json \
  --manifest artifacts/window_manifest.csv \
  --records-root Borunte_1510A_Dataset \
  --checkpoints checkpoints/pgmarl_manuscript_seed42.pt \
                checkpoints/pgmarl_manuscript_seed52.pt \
                checkpoints/pgmarl_manuscript_seed62.pt \
                checkpoints/pgmarl_manuscript_seed72.pt \
                checkpoints/pgmarl_manuscript_seed82.pt \
  --targets D E \
  --acknowledge-final-target-evaluation

python scripts/verify_release.py
```

Before training, the private BORUNTE raw reader must be connected to
`scripts/prepare_fixed_windows.py` and must write exactly the NPZ arrays in
`docs/DATA_SCHEMA.md`. This connection cannot be guessed safely because the
final trajectory-E files are absent from the workstation snapshot.

After evaluation, add the resulting manifest, audit JSON, checkpoints,
per-window logits/predictions, per-run metrics, confusion matrices, bootstrap
draws, source-selection histories and chronology records to `evidence/`, then
rerun the release verifier.
