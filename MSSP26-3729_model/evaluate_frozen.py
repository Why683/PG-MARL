from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pgmarl.dataset import FixedWindowDataset, collate_fixed
from pgmarl.metrics import classification_metrics
from pgmarl.models import TASN
from pgmarl.protocol import SOURCE_DOMAINS, TARGET_DOMAINS, sha256_file, verify_source_only_manifest


def load_student(checkpoint_path: Path, device: torch.device) -> Tuple[TASN, Dict]:
    payload = torch.load(checkpoint_path, map_location=device)
    config = payload["config"]
    if payload.get("target_domains_registered_during_training") not in ([], None):
        raise RuntimeError("checkpoint reports target registration during training")
    model = TASN(
        config["node_input_dim"], config["hidden_dim"], config["embedding_dim"],
        config["num_classes"], config["num_joints"], config["gat_layers"],
        config["gat_heads"], config["internal_sequence_length"]).to(device)
    model.load_state_dict(payload["student_state_dict"], strict=True)
    model.eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path,
                        default=Path("Borunte_1510A_Dataset"))
    parser.add_argument("--checkpoints", type=Path, nargs=5, required=True)
    parser.add_argument("--targets", nargs="+", choices=["D", "E"], required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/final_evaluation"))
    parser.add_argument("--acknowledge-final-target-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_final_target_evaluation:
        raise SystemExit("final target evaluation requires explicit acknowledgement")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_seeds = config["seeds"]
    frame = pd.read_csv(args.manifest)
    verify_source_only_manifest(frame)
    target_domains = [f"Domain_{name}_Trajectory" for name in args.targets]
    if not set(target_domains).issubset(TARGET_DOMAINS):
        raise RuntimeError("only D/E may be registered by the final evaluator")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, payloads = zip(*(load_student(path, device) for path in args.checkpoints))
    if [payload["seed"] for payload in payloads] != expected_seeds:
        raise RuntimeError("checkpoint order/seeds do not match the declared five seeds")
    manifest_hash = sha256_file(args.manifest)
    for payload in payloads:
        if payload["manifest_sha256"] != manifest_hash:
            raise RuntimeError("checkpoint and evaluator manifest hashes differ")
    args.output.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "protocol": "frozen five-seed A/B/C-to-D/E final evaluation",
        "manifest_sha256": manifest_hash,
        "config_sha256": sha256_file(args.config),
        "checkpoints": [{"path": str(path), "sha256": sha256_file(path),
                         "seed": payload["seed"]}
                        for path, payload in zip(args.checkpoints, payloads)],
        "target_metrics": {},
    }
    for domain in target_domains:
        dataset = FixedWindowDataset(
            args.manifest, [domain], "target_test", args.records_root)
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                            collate_fn=collate_fixed)
        labels, identities = [], []
        logits_by_seed = [[] for _ in models]
        with torch.no_grad():
            for raw in loader:
                sequence = raw["sequence"].to(device)
                labels.extend(raw["labels"].tolist())
                identities.extend(zip(raw["trial_ids"], raw["window_indices"]))
                for index, model in enumerate(models):
                    _, logits = model(sequence)
                    logits_by_seed[index].append(logits.cpu().numpy())
        labels_np = np.asarray(labels)
        logits_by_seed = [np.concatenate(parts) for parts in logits_by_seed]
        per_seed = []
        rows = []
        for seed, logits in zip(expected_seeds, logits_by_seed):
            predictions = logits.argmax(axis=1)
            per_seed.append(classification_metrics(labels_np, predictions))
            for identity, label, prediction, vector in zip(
                    identities, labels_np, predictions, logits):
                rows.append([seed, domain, identity[0], identity[1], int(label),
                             int(prediction), *[float(value) for value in vector]])
        ensemble_logits = np.mean(np.stack(logits_by_seed), axis=0)
        ensemble = classification_metrics(labels_np, ensemble_logits.argmax(axis=1))
        domain_dir = args.output / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        with (domain_dir / "per_window_predictions.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["seed", "domain", "trial_id", "window_index",
                             "label", "prediction", "logit_0", "logit_1",
                             "logit_2", "logit_3", "logit_4"])
            writer.writerows(rows)
        (domain_dir / "metrics.json").write_text(json.dumps({
            "per_seed": dict(zip(map(str, expected_seeds), per_seed)),
            "equal_weight_logit_ensemble": ensemble,
            "n_physical_windows": int(labels_np.size),
        }, indent=2), encoding="utf-8")
        run_summary["target_metrics"][domain] = {
            "macro_f1_mean": float(np.mean([item["macro_f1"] for item in per_seed])),
            "macro_f1_sample_sd": float(np.std(
                [item["macro_f1"] for item in per_seed], ddof=1)),
            "ensemble_macro_f1": ensemble["macro_f1"],
            "n": int(labels_np.size),
        }
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))
    print("FINALITY: do not use D/E outputs for model, seed or checkpoint selection.")


if __name__ == "__main__":
    main()
