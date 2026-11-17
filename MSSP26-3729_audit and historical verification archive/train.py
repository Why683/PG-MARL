from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import higher
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from pgmarl.dataset import FixedWindowDataset, collate_fixed
from pgmarl.losses import (
    calibrate_mmd_bandwidth,
    conditional_multi_kernel_mmd,
    cosine_beta,
    outer_query_loss,
    support_task_loss,
)
from pgmarl.models import MMPS, TASN
from pgmarl.protocol import SOURCE_DOMAINS, TARGET_DOMAINS, sha256_file, verify_source_only_manifest


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move(batch: dict, device: torch.device) -> dict:
    return {
        **batch,
        "sequence": batch["sequence"].to(device),
        "mechanism_features": batch["mechanism_features"].to(device),
        "residual_sequence": batch["residual_sequence"].to(device),
        "labels": batch["labels"].to(device),
    }


def make_loader(manifest: Path, domains: List[str], split: str,
                batch_size: int, shuffle: bool, seed: int,
                records_root: Path) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        FixedWindowDataset(manifest, domains, split, records_root), batch_size=batch_size,
        shuffle=shuffle, generator=generator, collate_fn=collate_fixed,
        num_workers=0, drop_last=shuffle)


def repeat_loader(loader: DataLoader):
    """Re-iterate a loader indefinitely without caching complete batches."""
    while True:
        yield from loader


@torch.no_grad()
def evaluate(student: TASN, manifest: Path, domains: List[str], split: str,
             batch_size: int, device: torch.device,
             records_root: Path) -> Tuple[float, float, Dict]:
    student.eval()
    all_labels, all_predictions, all_domains = [], [], []
    loader = make_loader(
        manifest, domains, split, batch_size, False, 0, records_root)
    for raw in loader:
        batch = move(raw, device)
        _, logits = student(batch["sequence"])
        all_labels.extend(batch["labels"].cpu().tolist())
        all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
        all_domains.extend(raw["domains"])
    labels = np.asarray(all_labels)
    predictions = np.asarray(all_predictions)
    pooled = float(f1_score(labels, predictions, average="macro", zero_division=0))
    by_domain = {}
    for domain in domains:
        mask = np.asarray(all_domains) == domain
        by_domain[domain] = {
            "accuracy": float(accuracy_score(labels[mask], predictions[mask])),
            "macro_f1": float(f1_score(
                labels[mask], predictions[mask], average="macro", zero_division=0)),
            "n": int(mask.sum()),
        }
    worst = min(item["macro_f1"] for item in by_domain.values())
    return worst, pooled, by_domain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/window_manifest.csv"))
    parser.add_argument("--records-root", type=Path,
                        default=Path("Borunte_1510A_Dataset"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed is not one of the five predeclared seeds")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_frame = pd.read_csv(args.manifest)
    verify_source_only_manifest(manifest_frame)
    source_frame = manifest_frame[manifest_frame.domain.isin(SOURCE_DOMAINS)]
    if source_frame.empty or source_frame.domain.isin(TARGET_DOMAINS).any():
        raise RuntimeError("training manifest registration is not source-only")

    source_domains = list(config["source_domains"])
    batch_size = int(config["batch_size"])
    train_loader = make_loader(
        args.manifest, source_domains, "train", batch_size, True, args.seed,
        args.records_root)
    first = next(iter(train_loader))
    feature_dim = int(first["mechanism_features"].shape[1])
    teacher = MMPS(feature_dim, config["embedding_dim"], config["num_classes"]).to(device)
    student = TASN(
        config["node_input_dim"], config["hidden_dim"], config["embedding_dim"],
        config["num_classes"], config["num_joints"], config["gat_layers"],
        config["gat_heads"], config["internal_sequence_length"]).to(device)

    teacher_optimizer = torch.optim.Adam(
        teacher.parameters(), lr=1e-3, weight_decay=1e-5)
    teacher_best = None
    teacher_best_score = (-float("inf"), -float("inf"))
    teacher_iterator = repeat_loader(train_loader)
    for epoch in range(int(config["teacher_epochs"])):
        teacher.train()
        for _ in range(int(config["updates_per_epoch"])):
            batch = move(next(teacher_iterator), device)
            _, logits = teacher(
                batch["mechanism_features"], batch["residual_sequence"])
            loss = nn.functional.cross_entropy(logits, batch["labels"])
            teacher_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            teacher_optimizer.step()
        # Teacher selection uses only A/B/C validation.
        if (epoch + 1) % int(config["validate_every_epochs"]) == 0:
            teacher.eval()
            labels, predictions, domains = [], [], []
            val_loader = make_loader(
                args.manifest, source_domains, "validation", batch_size, False, 0,
                args.records_root)
            with torch.no_grad():
                for raw in val_loader:
                    batch = move(raw, device)
                    _, logits = teacher(
                        batch["mechanism_features"], batch["residual_sequence"])
                    labels.extend(batch["labels"].cpu().tolist())
                    predictions.extend(logits.argmax(1).cpu().tolist())
                    domains.extend(raw["domains"])
            labels_np, predictions_np = np.asarray(labels), np.asarray(predictions)
            per_domain = []
            for domain in source_domains:
                mask = np.asarray(domains) == domain
                per_domain.append(f1_score(
                    labels_np[mask], predictions_np[mask], average="macro", zero_division=0))
            score = (min(per_domain), f1_score(
                labels_np, predictions_np, average="macro", zero_division=0))
            if score > teacher_best_score:
                teacher_best_score = score
                teacher_best = copy.deepcopy(teacher.state_dict())
    if teacher_best is None:
        raise RuntimeError("no teacher checkpoint selected on source validation")
    teacher.load_state_dict(teacher_best)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    domain_loaders = {
        domain: repeat_loader(make_loader(
            args.manifest, [domain], "train", batch_size, True,
            args.seed + index + 1, args.records_root))
        for index, domain in enumerate(source_domains)
    }
    outer_optimizer = torch.optim.Adam(
        student.parameters(), lr=config["outer_lr"],
        weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        outer_optimizer, T_max=config["meta_epochs"], eta_min=config["outer_lr_min"])

    # Calibrate one frozen source-only embedding scale before target registration.
    calibration = move(next(iter(train_loader)), device)
    with torch.no_grad():
        calibration_embedding, _ = student(calibration["sequence"])
    bandwidth = calibrate_mmd_bandwidth(
        calibration_embedding, config["mmd_calibration_max_windows"],
        config["mmd_calibration_seed"])

    best_student = None
    best_score = (-float("inf"), -float("inf"))
    best_epoch = None
    selection_history = []
    start = time.perf_counter()
    for epoch in range(int(config["meta_epochs"])):
        student.train()
        beta = cosine_beta(config["beta_init"], epoch, config["meta_epochs"])
        for update_index in range(int(config["updates_per_epoch"])):
            query_index = (epoch * int(config["updates_per_epoch"]) + update_index) % 3
            query_domain = source_domains[query_index]
            support_domain = source_domains[(query_index + 1) % 3]
            support = move(next(domain_loaders[support_domain]), device)
            query = move(next(domain_loaders[query_domain]), device)
            with torch.no_grad():
                support_teacher_embedding, support_teacher_logits = teacher(
                    support["mechanism_features"], support["residual_sequence"])
                query_teacher_embedding, query_teacher_logits = teacher(
                    query["mechanism_features"], query["residual_sequence"])
            inner_optimizer = torch.optim.SGD(student.parameters(), lr=config["inner_lr"])
            outer_optimizer.zero_grad(set_to_none=True)
            with higher.innerloop_ctx(
                    student, inner_optimizer, copy_initial_weights=False,
                    track_higher_grads=True) as (adapted, differentiable_optimizer):
                for _ in range(int(config["inner_steps"])):
                    support_student_embedding, support_student_logits = adapted(
                        support["sequence"])
                    inner_loss = support_task_loss(
                        support_student_embedding, support_student_logits,
                        support_teacher_embedding, support_teacher_logits,
                        support["labels"], config["alpha_feature"], beta,
                        config["gamma_cross_entropy"], config["temperature"])
                    differentiable_optimizer.step(inner_loss)
                support_student_embedding, _ = adapted(support["sequence"])
                query_student_embedding, query_student_logits = adapted(query["sequence"])
                cmmd = conditional_multi_kernel_mmd(
                    support_student_embedding, query_student_embedding,
                    support_teacher_logits, query_teacher_logits, bandwidth,
                    config["mmd_scales"], config["class_weights"], config["epsilon"])
                loss = outer_query_loss(
                    query_student_embedding, query_student_logits,
                    query_teacher_embedding, query["labels"], cmmd,
                    config["alpha_feature"], config["gamma_cross_entropy"],
                    config["lambda_conditional_mmd"])
                loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            outer_optimizer.step()
        scheduler.step()
        if (epoch + 1) % int(config["validate_every_epochs"]) == 0:
            worst, pooled, by_domain = evaluate(
                student, args.manifest, source_domains, "validation", batch_size,
                device, args.records_root)
            record = {"epoch": epoch + 1, "worst_macro_f1": worst,
                      "pooled_macro_f1": pooled, "by_domain": by_domain}
            selection_history.append(record)
            if (worst, pooled) > best_score:
                best_score = (worst, pooled)
                best_epoch = epoch + 1
                best_student = copy.deepcopy(student.state_dict())
    if best_student is None:
        raise RuntimeError("no student checkpoint selected on A/B/C validation")

    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"pgmarl_manuscript_seed{args.seed}.pt"
    torch.save({
        "protocol_version": config["protocol_version"],
        "seed": args.seed,
        "config": config,
        "config_sha256": sha256_file(args.config),
        "manifest_sha256": sha256_file(args.manifest),
        "source_domains": source_domains,
        "target_domains_registered_during_training": [],
        "teacher_state_dict": teacher_best,
        "student_state_dict": best_student,
        "mechanism_feature_dim": feature_dim,
        "mmd_base_bandwidth": bandwidth,
        "best_source_validation_epoch": best_epoch,
        "source_validation_selection_history": selection_history,
        "training_seconds": time.perf_counter() - start,
    }, output)
    print(f"Checkpoint: {output}")
    print("PASS: D/E were not loaded, evaluated, tuned or selected during training.")


if __name__ == "__main__":
    main()
