from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FixedWindowDataset(Dataset):
    """Load immutable window records selected by the manifest."""

    def __init__(self, manifest: Path, domains: Sequence[str], split: str,
                 records_root: Optional[Path] = None) -> None:
        frame = pd.read_csv(manifest)
        self.manifest_parent = Path(manifest).resolve().parent
        self.records_root = (Path(records_root).resolve()
                             if records_root is not None else None)
        self.frame = frame[
            frame.domain.isin(domains) & frame.split.eq(split)
        ].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"no manifest rows for domains={domains}, split={split}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        listed = Path(row.npz_path)
        candidates = ([listed] if listed.is_absolute() else [])
        if self.records_root is not None and not listed.is_absolute():
            candidates.append(self.records_root / listed)
        if not listed.is_absolute():
            candidates.append(self.manifest_parent / listed)
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"window record not found; checked: {[str(x) for x in candidates]}")
        with np.load(path, allow_pickle=False) as payload:
            item = {
                "sequence": torch.from_numpy(payload["sequence"].astype(np.float32)),
                "mechanism_features": torch.from_numpy(
                    payload["mechanism_features"].astype(np.float32)),
                "residual_sequence": torch.from_numpy(
                    payload["residual_sequence"].astype(np.float32)),
                "label": torch.tensor(int(payload["label"]), dtype=torch.long),
                "domain": str(row.domain),
                "trial_id": str(row.trial_id),
                "window_index": int(row.window_index),
            }
        return item


def collate_fixed(batch: Iterable[dict]) -> dict:
    batch = list(batch)
    return {
        "sequence": torch.stack([item["sequence"] for item in batch]),
        "mechanism_features": torch.stack([
            item["mechanism_features"] for item in batch]),
        "residual_sequence": torch.stack([
            item["residual_sequence"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
        "domains": [item["domain"] for item in batch],
        "trial_ids": [item["trial_id"] for item in batch],
        "window_indices": [item["window_index"] for item in batch],
    }
