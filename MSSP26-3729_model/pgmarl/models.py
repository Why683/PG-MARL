from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Sign-preserving robust compression used by MMPS and TASN."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


class PeakPreservingResidualCNN(nn.Module):
    def __init__(self, output_dim: int = 32) -> None:
        super().__init__()
        c1 = output_dim // 3
        c2 = output_dim // 3
        c3 = output_dim - c1 - c2
        self.branches = nn.ModuleList([
            nn.Conv1d(1, c1, kernel_size=3, padding=1),
            nn.Conv1d(1, c2, kernel_size=7, padding=3),
            nn.Conv1d(1, c3, kernel_size=15, padding=7),
        ])
        self.norm = nn.InstanceNorm1d(output_dim, affine=True)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, residual_sequence: torch.Tensor) -> torch.Tensor:
        if residual_sequence.ndim == 2:
            residual_sequence = residual_sequence[:, None, :]
        features = torch.cat([
            F.gelu(branch(residual_sequence)) for branch in self.branches
        ], dim=1)
        return self.pool(self.norm(features)).squeeze(-1)


class MMPS(nn.Module):
    """Source-trained fusion network over fixed mechanism descriptors.

    Mechanical descriptors are calculated outside this module by
    :mod:`pgmarl.physics`.  The mapping and classification head are learned on
    labelled A/B/C source-training windows only.  The resulting embedding and
    probabilities are reference information, not analytical class labels.
    """

    def __init__(self, feature_dim: int, embedding_dim: int = 64,
                 num_classes: int = 5) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        if self.feature_dim != 37:
            raise ValueError(
                "the manuscript reference expects 15 residual, 4 friction and "
                "18 order-domain scalar descriptors")
        self.group_norms = nn.ModuleList([
            nn.BatchNorm1d(15), nn.BatchNorm1d(4), nn.BatchNorm1d(18)
        ])
        self.residual_encoder = PeakPreservingResidualCNN(32)
        fused_dim = self.feature_dim + 32
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, mechanism_features: torch.Tensor,
                residual_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if mechanism_features.ndim != 2 or mechanism_features.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected [B,{self.feature_dim}] mechanism features, got "
                f"{tuple(mechanism_features.shape)}")
        compressed = symlog(torch.nan_to_num(mechanism_features))
        groups = torch.split(compressed, (15, 4, 18), dim=1)
        normalized = torch.cat([
            norm(group) for norm, group in zip(self.group_norms, groups)
        ], dim=1)
        residual_features = self.residual_encoder(
            symlog(torch.nan_to_num(residual_sequence)))
        embedding = self.fusion(torch.cat([normalized, residual_features], dim=1))
        return embedding, self.classifier(embedding)


class TASN(nn.Module):
    """Deployed graph-temporal student described in Section 3.2.

    The module deliberately has no direct MMPS residual/order branch.  MMPS
    information reaches this student only through TAMD losses during training.
    """

    def __init__(self, node_input_dim: int = 12, hidden_dim: int = 96,
                 embedding_dim: int = 64, num_classes: int = 5,
                 num_joints: int = 6, gat_layers: int = 3,
                 gat_heads: int = 4, expected_steps: int = 200) -> None:
        super().__init__()
        if gat_layers != 3:
            raise ValueError("the 11.16 manuscript reference uses three GATv2 layers")
        self.num_joints = int(num_joints)
        self.expected_steps = int(expected_steps)
        self.input_projection = nn.Linear(node_input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.gat_layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, heads=gat_heads, concat=False,
                      add_self_loops=True)
            for _ in range(gat_layers)
        ])
        self.gat_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(gat_layers)
        ])
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.recurrent_output_norm = nn.LayerNorm(hidden_dim)
        self.embedding = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    @staticmethod
    def chain_edges(num_joints: int, batch_size: int, device: torch.device) -> torch.Tensor:
        src, dst = [], []
        for batch_index in range(batch_size):
            offset = batch_index * num_joints
            for joint in range(num_joints - 1):
                src.extend([offset + joint, offset + joint + 1])
                dst.extend([offset + joint + 1, offset + joint])
        return torch.tensor([src, dst], dtype=torch.long, device=device)

    def forward(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the 64-D student embedding and five-class logits.

        ``sequence`` has shape ``[batch, 200, 6, channels]``.  SymLog and the
        input projection are applied before graph message passing at every
        native time state.  The normalized recurrent state is the state passed
        to the next step and the representation used by graph readout.
        """
        if sequence.ndim != 4:
            raise ValueError("TASN input must have shape [B,T,N,C]")
        batch_size, steps, joints, _ = sequence.shape
        if steps != self.expected_steps:
            raise ValueError(f"expected {self.expected_steps} recurrent states, got {steps}")
        if joints != self.num_joints:
            raise ValueError(f"expected {self.num_joints} joints, got {joints}")
        edges = self.chain_edges(joints, batch_size, sequence.device)
        state = torch.zeros(
            batch_size * joints,
            self.gru.hidden_size,
            dtype=sequence.dtype,
            device=sequence.device,
        )
        projected = self.input_norm(self.input_projection(symlog(sequence)))
        for time_index in range(steps):
            x = projected[:, time_index].reshape(batch_size * joints, -1)
            for layer, norm in zip(self.gat_layers, self.gat_norms):
                update = F.gelu(layer(x, edges))
                x = norm(x + update)
            state = self.gru(x, state)
            state = self.recurrent_output_norm(state)
        node_states = state.view(batch_size, joints, -1)
        graph_state = node_states.mean(dim=1)
        embedding = self.embedding(graph_state)
        return embedding, self.classifier(embedding)
