from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch.nn import functional as F


def reliability_weights(teacher_logits: torch.Tensor, labels: torch.Tensor,
                        num_classes: int = 5) -> torch.Tensor:
    probabilities = F.softmax(teacher_logits.detach(), dim=1)
    confidence = probabilities.gather(1, labels[:, None]).squeeze(1)
    correct = probabilities.argmax(dim=1).eq(labels).to(probabilities.dtype)
    chance = 1.0 / float(num_classes)
    return correct * ((confidence - chance) / (1.0 - chance)).clamp_min(0.0)


def feature_distillation(student_embedding: torch.Tensor,
                         teacher_embedding: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_embedding, teacher_embedding.detach())


def probability_distillation(student_logits: torch.Tensor,
                             teacher_logits: torch.Tensor,
                             labels: torch.Tensor,
                             temperature: float = 4.0,
                             epsilon: float = 1e-8) -> torch.Tensor:
    weights = reliability_weights(teacher_logits, labels, student_logits.shape[1])
    if not bool(torch.any(weights > 0)):
        return student_logits.new_zeros(())
    teacher_probability = F.softmax(teacher_logits.detach() / temperature, dim=1)
    student_log_probability = F.log_softmax(student_logits / temperature, dim=1)
    per_sample = F.kl_div(
        student_log_probability, teacher_probability, reduction="none").sum(dim=1)
    return (temperature ** 2) * torch.sum(weights * per_sample) / torch.clamp(
        weights.sum(), min=epsilon)


def scaled_cross_entropy(logits: torch.Tensor, labels: torch.Tensor,
                         gamma: float = 2.0) -> torch.Tensor:
    """The manuscript's gamma is a CE coefficient, not focal gamma."""
    return float(gamma) * F.cross_entropy(logits, labels)


def support_task_loss(student_embedding: torch.Tensor, student_logits: torch.Tensor,
                      teacher_embedding: torch.Tensor, teacher_logits: torch.Tensor,
                      labels: torch.Tensor, alpha: float, beta: float,
                      gamma: float, temperature: float) -> torch.Tensor:
    return (
        scaled_cross_entropy(student_logits, labels, gamma)
        + float(alpha) * feature_distillation(student_embedding, teacher_embedding)
        + float(beta) * probability_distillation(
            student_logits, teacher_logits, labels, temperature)
    )


def cosine_beta(beta_init: float, epoch_index: int, total_epochs: int) -> float:
    if total_epochs <= 1:
        return 0.0
    progress = min(max(epoch_index / float(total_epochs - 1), 0.0), 1.0)
    return float(beta_init) * 0.5 * (1.0 + torch.cos(
        torch.tensor(progress * torch.pi)).item())


def calibrate_mmd_bandwidth(embeddings: torch.Tensor, max_points: int = 1024,
                            seed: int = 42, epsilon: float = 1e-8) -> float:
    detached = embeddings.detach()
    if detached.shape[0] > max_points:
        generator = torch.Generator(device=detached.device).manual_seed(seed)
        index = torch.randperm(detached.shape[0], generator=generator,
                               device=detached.device)[:max_points]
        detached = detached[index]
    distances = torch.pdist(detached, p=2)
    positive = distances[distances > epsilon]
    if positive.numel() == 0:
        return 1.0
    return float(torch.median(positive).item())


def _kernel(x: torch.Tensor, y: torch.Tensor, bandwidth: float,
            scales: Sequence[float]) -> torch.Tensor:
    squared = torch.cdist(x, y, p=2).pow(2)
    kernels = [torch.exp(-squared / (2.0 * (bandwidth * scale) ** 2))
               for scale in scales]
    return torch.stack(kernels).mean(dim=0)


def _weighted_mmd(x: torch.Tensor, y: torch.Tensor,
                  weight_x: torch.Tensor, weight_y: torch.Tensor,
                  bandwidth: float, scales: Sequence[float],
                  epsilon: float) -> torch.Tensor:
    wx = weight_x / weight_x.sum().clamp_min(epsilon)
    wy = weight_y / weight_y.sum().clamp_min(epsilon)
    kxx = _kernel(x, x, bandwidth, scales)
    kyy = _kernel(y, y, bandwidth, scales)
    kxy = _kernel(x, y, bandwidth, scales)
    return wx @ kxx @ wx + wy @ kyy @ wy - 2.0 * (wx @ kxy @ wy)


def conditional_multi_kernel_mmd(
        support_embedding: torch.Tensor, query_embedding: torch.Tensor,
        support_teacher_logits: torch.Tensor, query_teacher_logits: torch.Tensor,
        bandwidth: float, scales: Iterable[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
        class_weights: Iterable[float] = (0.2, 0.2, 0.2, 0.2, 0.2),
        epsilon: float = 1e-8) -> torch.Tensor:
    support_probability = F.softmax(support_teacher_logits.detach(), dim=1)
    query_probability = F.softmax(query_teacher_logits.detach(), dim=1)
    terms = []
    for class_index, aggregate_weight in enumerate(class_weights):
        terms.append(float(aggregate_weight) * _weighted_mmd(
            support_embedding, query_embedding,
            support_probability[:, class_index], query_probability[:, class_index],
            bandwidth, tuple(scales), epsilon))
    return torch.stack(terms).sum()


def outer_query_loss(query_student_embedding: torch.Tensor,
                     query_student_logits: torch.Tensor,
                     query_teacher_embedding: torch.Tensor,
                     query_labels: torch.Tensor,
                     conditional_mmd: torch.Tensor,
                     alpha: float = 1.0, gamma: float = 2.0,
                     mmd_weight: float = 0.1) -> torch.Tensor:
    """Equation 16 objective; probability distillation is intentionally absent."""
    return (
        scaled_cross_entropy(query_student_logits, query_labels, gamma)
        + float(alpha) * feature_distillation(
            query_student_embedding, query_teacher_embedding)
        + float(mmd_weight) * conditional_mmd
    )

