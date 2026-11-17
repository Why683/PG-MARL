import torch

from pgmarl.losses import probability_distillation
from pgmarl.models import MMPS, TASN


def test_model_shapes():
    student = TASN(node_input_dim=12, hidden_dim=24, embedding_dim=64)
    embedding, logits = student(torch.randn(2, 200, 6, 12))
    assert embedding.shape == (2, 64)
    assert logits.shape == (2, 5)
    teacher = MMPS(feature_dim=37)
    teacher_embedding, teacher_logits = teacher(
        torch.randn(2, 37), torch.randn(2, 200))
    assert teacher_embedding.shape == (2, 64)
    assert teacher_logits.shape == (2, 5)


def test_all_wrong_teacher_has_zero_probability_loss():
    student = torch.randn(2, 5)
    teacher = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0],
                            [5.0, 0.0, 0.0, 0.0, 0.0]])
    labels = torch.tensor([0, 1])
    assert probability_distillation(student, teacher, labels).item() == 0.0
