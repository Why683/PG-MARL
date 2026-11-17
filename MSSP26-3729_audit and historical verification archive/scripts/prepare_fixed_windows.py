"""Adapter contract for preparing fixed records from the BORUNTE CSV tree.

The historical raw reader is included in the companion archive.  Copy its
validated CSV/channel parser into a private deployment module and call
``write_record`` for every manifest row.  This public package intentionally
does not guess institution-specific raw filenames or redistribute raw data.
"""

import argparse
from pathlib import Path

import numpy as np


def write_record(path: Path, sequence: np.ndarray, theta: np.ndarray,
                 theta_dot: np.ndarray, theta_ddot: np.ndarray,
                 tau_true: np.ndarray, vibration: np.ndarray,
                 mechanism_features: np.ndarray, residual_sequence: np.ndarray,
                 label: int) -> None:
    if sequence.shape[0] != 200 or sequence.shape[1] != 6:
        raise ValueError("sequence must have shape [200,6,C]")
    if vibration.shape[0] != 2000:
        raise ValueError("effective 1 kHz vibration must contain 2000 samples")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, sequence=sequence.astype(np.float32), theta=theta.astype(np.float32),
        theta_dot=theta_dot.astype(np.float32), theta_ddot=theta_ddot.astype(np.float32),
        tau_true=tau_true.astype(np.float32), vibration=vibration.astype(np.float32),
        mechanism_features=mechanism_features.astype(np.float32),
        residual_sequence=residual_sequence.astype(np.float32),
        label=np.asarray(label, dtype=np.int64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.parse_args()
    raise SystemExit(
        "Raw-file naming is institution-specific. Connect the retained validated "
        "BORUNTE reader from the companion archive to write_record(); do not "
        "silently infer channels or trial timestamps.")


if __name__ == "__main__":
    main()
