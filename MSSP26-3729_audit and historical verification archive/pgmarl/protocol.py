from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


SOURCE_DOMAINS = ("Domain_A_Trajectory", "Domain_B_Trajectory", "Domain_C_Trajectory")
TARGET_DOMAINS = ("Domain_D_Trajectory", "Domain_E_Trajectory")


@dataclass(frozen=True)
class WindowRecord:
    domain: str
    class_id: int
    raw_class_id: int
    class_name: str
    trial_id: str
    split: str
    window_index: int
    start_s: float
    end_s: float
    npz_path: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_trials(trial_ids: Sequence[str], target_domain: bool) -> Dict[str, List[str]]:
    ordered = sorted(trial_ids)
    if target_domain:
        return {"target_test": ordered}
    if len(ordered) < 3:
        raise ValueError("each source class requires at least three independent trials")
    # The retained acquisition uses trial_002/003 for val/test when exactly
    # three trials exist and trial_003/002 otherwise.  This reproduces the
    # original fixed trial identity rule while keeping splits disjoint.
    if len(ordered) == 3:
        val_id, test_id = "trial_002", "trial_003"
    else:
        val_id, test_id = "trial_003", "trial_002"
    missing = [item for item in (val_id, test_id) if item not in ordered]
    if missing:
        raise ValueError(f"missing declared source split trials: {missing}")
    return {
        "train": [item for item in ordered if item not in {val_id, test_id}],
        "validation": [val_id],
        "test": [test_id],
    }


def build_manifest(data_root: Path, output_csv: Path,
                   window_seconds: float = 2.0,
                   complete_trial_seconds: float = 600.0) -> pd.DataFrame:
    records: List[dict] = []
    expected_windows = int(round(complete_trial_seconds / window_seconds))
    allowed = set(SOURCE_DOMAINS + TARGET_DOMAINS)
    for domain_dir in sorted(path for path in data_root.iterdir()
                             if path.is_dir() and path.name in allowed):
        for class_dir in sorted(path for path in domain_dir.iterdir() if path.is_dir()):
            prefix, _, class_name = class_dir.name.partition("_")
            raw_class_id = int(prefix)
            canonical = {
                "wear": 0, "crack": 1, "broken": 2,
                "normal": 3, "pitting": 4,
            }
            class_key = class_name.lower().replace(" tooth", "").strip()
            if class_key not in canonical:
                raise ValueError(f"unsupported health-state folder: {class_dir.name}")
            class_id = canonical[class_key]
            trials = sorted(path for path in class_dir.iterdir()
                            if path.is_dir() and path.name.startswith("trial_"))
            partitions = split_trials(
                [trial.name for trial in trials], domain_dir.name in TARGET_DOMAINS)
            trial_lookup = {trial.name: trial for trial in trials}
            for split, trial_ids in partitions.items():
                for trial_id in trial_ids:
                    trial_dir = trial_lookup[trial_id]
                    for window_index in range(expected_windows):
                        start = window_index * window_seconds
                        records.append(WindowRecord(
                            domain_dir.name, class_id, raw_class_id, class_name, trial_id,
                            split, window_index, start, start + window_seconds,
                            str((trial_dir.relative_to(data_root) /
                                 "processed_windows" /
                                 f"window_{window_index:03d}.npz").as_posix()),
                        ).__dict__)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError(f"no supported domain folders found below {data_root}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def verify_source_only_manifest(frame: pd.DataFrame) -> dict:
    required = {"domain", "class_id", "trial_id", "split", "window_index",
                "start_s", "end_s"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")
    source = frame[frame.domain.isin(SOURCE_DOMAINS)]
    target = frame[frame.domain.isin(TARGET_DOMAINS)]
    illegal_target = target[target.split != "target_test"]
    if not illegal_target.empty:
        raise ValueError("D/E contain non-target-test records")
    split_sets = {
        split: set(zip(rows.domain, rows.class_id, rows.trial_id))
        for split, rows in source.groupby("split")
    }
    for left in split_sets:
        for right in split_sets:
            if left < right and split_sets[left] & split_sets[right]:
                raise ValueError(f"trial overlap between {left} and {right}")
    duplicate = frame.duplicated(
        ["domain", "class_id", "trial_id", "window_index"]).any()
    if duplicate:
        raise ValueError("duplicate fixed window identity")
    return {
        "source_rows": int(len(source)),
        "target_rows": int(len(target)),
        "rows_by_domain_split": {
            f"{domain}/{split}": int(len(rows))
            for (domain, split), rows in frame.groupby(["domain", "split"])
        },
        "pass": True,
    }


def write_audit(frame: pd.DataFrame, path: Path) -> dict:
    audit = verify_source_only_manifest(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit
