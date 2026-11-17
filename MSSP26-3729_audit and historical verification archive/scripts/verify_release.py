from __future__ import annotations

import hashlib
import json
from pathlib import Path


REQUIRED = [
    "README.md", "requirements.txt", "configs/manuscript_11_16.json",
    "pgmarl/models.py", "pgmarl/losses.py", "pgmarl/protocol.py",
    "train.py", "evaluate_frozen.py", "evidence/PROVENANCE_GAPS.md",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    missing = [item for item in REQUIRED if not (root / item).exists()]
    if missing:
        raise SystemExit(f"missing required release files: {missing}")
    ledger = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if ".git" in path.parts or path.name == "SHA256SUMS.json":
            continue
        ledger[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(
            path.read_bytes()).hexdigest()
    (root / "SHA256SUMS.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    print(f"PASS: {len(ledger)} release files hashed")


if __name__ == "__main__":
    main()
