from __future__ import annotations

import hashlib
import json
from pathlib import Path


REQUIRED = [
    "README.md", "requirements.txt", "configs/manuscript_11_16.json",
    "pgmarl/models.py", "pgmarl/physics.py", "pgmarl/losses.py",
    "pgmarl/protocol.py", "train.py", "evaluate_frozen.py",
    "reviewer/REVIEWER_CODE_SCOPE.md",
]
FORBIDDEN_PARTS = {"evidence", "legacy_adapter", "manuscript_claims",
                   "historical_local_verification", "Borunte_1510A_Dataset"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    missing = [item for item in REQUIRED if not (root / item).exists()]
    if missing:
        raise SystemExit(f"missing required model-release files: {missing}")
    forbidden = [str(path.relative_to(root)) for path in root.rglob("*")
                 if set(path.relative_to(root).parts) & FORBIDDEN_PARTS]
    if forbidden:
        raise SystemExit(f"forbidden data/evidence paths present: {forbidden}")
    if (root / "scripts" / "prepare_fixed_windows.py").exists():
        raise SystemExit("heterogeneous-rate preprocessing script must be separate")
    ledger = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS.json" or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        ledger[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "SHA256SUMS.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    print(f"PASS: model-only release contains {len(ledger)} hashed files")


if __name__ == "__main__":
    main()
