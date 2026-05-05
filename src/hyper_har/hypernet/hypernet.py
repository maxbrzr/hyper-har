from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class MLPResidualBlock(nn.Module):
    """Residual MLP block used in the hypernetwork backbone."""

    def __init__(
        self, dim: int = 128, hidden_dim: int = 512, dropout: float = 0.05
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        x_norm = self.norm(x)
        # x_norm: (..., dim)
        x_mlp = self.net(x_norm)
        # x_mlp: (..., dim)
        x = x + x_mlp
        # x: (..., dim)
        return x


class HyperNetBackbone(nn.Module):
    """Shared feature extraction backbone processing all target modules simultaneously."""

    def __init__(
        self, c_subject_dim: int, num_target_modules: int, dropout: float = 0.05
    ) -> None:
        super().__init__()
        self.num_target_modules = num_target_modules

        self.subject_encoder = nn.Sequential(
            nn.Linear(c_subject_dim, 96), nn.LayerNorm(96, eps=1e-5)
        )

        self.module_encoder = nn.Sequential(
            nn.Embedding(num_target_modules, 32), nn.LayerNorm(32, eps=1e-5)
        )

        self.mixer = nn.Sequential(
            nn.Linear(128, 512),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Dropout(p=dropout),
        )

        self.mlp1 = MLPResidualBlock(dim=128, hidden_dim=512, dropout=dropout)

    def forward(self, c_subject: torch.Tensor) -> torch.Tensor:
        # Input c_subject: (batch, c_subject_dim)
        batch_size = c_subject.size(0)
        device = c_subject.device

        # 1) Encode subject context and broadcast across module axis
        e_subj = self.subject_encoder(c_subject)
        # e_subj: (batch, 96)
        e_subj = e_subj.unsqueeze(1).expand(-1, self.num_target_modules, -1)
        # e_subj: (batch, num_target_modules, 96)

        # 2) Encode module ids and broadcast across batch axis
        all_module_idxs = torch.arange(self.num_target_modules, device=device)
        # all_module_idxs: (num_target_modules,)
        e_mod = self.module_encoder(all_module_idxs)
        # e_mod: (num_target_modules, 32)
        e_mod = e_mod.unsqueeze(0).expand(batch_size, -1, -1)
        # e_mod: (batch, num_target_modules, 32)

        # 3) Concatenate and mix
        x = torch.cat([e_subj, e_mod], dim=-1)
        # x: (batch, num_target_modules, 128)

        x = self.mixer(x)
        # x: (batch, num_target_modules, 128)
        x = self.mlp1(x)
        # x: (batch, num_target_modules, 128)

        return x


class HyperNet(nn.Module):
    def __init__(
        self,
        num_channels: int,  # number of sensors
        num_classes: int,
        set_encoder_hidden_dim: int = 64,  # From PrototypicalSetEncoder
        nb_filters: int = 4,  # From TinierHAR
        nb_units_gru: int = 16,  # From TinierHAR
        lora_rank: int = 8,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.lora_rank = lora_rank

        # Calculate exactly what the set encoder outputs
        c_subject_dim = num_classes * set_encoder_hidden_dim

        # Calculate exactly what TinierHAR shapes are at each point
        in_c_conv1 = 1
        out_c_conv1 = nb_filters
        in_c_conv_last = 2 * nb_filters
        out_c_conv_last = 2 * nb_filters

        # TinierHAR dynamically collapses the feature dimension before the GRU:
        # gru_input = conv_out_channels (2 * nb_filters) * sensor_channels (num_channels)
        gru_in_dim = (2 * nb_filters) * num_channels

        # 1. Define the 5 target modules and their expected PyTorch parameter shapes
        # Map: name -> (out_features, in_features)
        self.target_shapes: Dict[str, Tuple[int, int]] = {
            "conv1_pointwise": (out_c_conv1, in_c_conv1),
            "conv_last_pointwise": (out_c_conv_last, in_c_conv_last),
            "gru_ih_fwd": (
                3 * nb_units_gru,
                gru_in_dim,
            ),  # 3x for Reset, Update, New gates
            "gru_ih_rev": (3 * nb_units_gru, gru_in_dim),
            "classifier": (num_classes, 2 * nb_units_gru),  # 2x for Bidirectional
        }
        self.module_names = list(self.target_shapes.keys())

        # 2. Shared Backbone
        self.backbone = HyperNetBackbone(c_subject_dim, len(self.module_names), dropout)

        self.mlp2 = MLPResidualBlock(dim=128, hidden_dim=512, dropout=dropout)
        self.mlp3 = nn.Sequential(
            nn.LayerNorm(128, eps=1e-5), nn.Linear(128, 512), nn.SiLU()
        )

        # 3. Dedicated linear output heads per module
        self.heads = nn.ModuleDict()
        for name, (out_dim, in_dim) in self.target_shapes.items():
            size_A = lora_rank * in_dim
            size_B = out_dim * lora_rank

            head = nn.Linear(512, size_A + size_B)

            # Initialization (A = tiny random, B = zero)
            nn.init.zeros_(head.weight)
            nn.init.uniform_(head.bias[:size_A], -1e-3, 1e-3)
            nn.init.zeros_(head.bias[size_A:])

            self.heads[name] = head

    def forward(
        self, c_subject: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        # Input c_subject: (batch, c_subject_dim)

        x_batched = self.backbone(c_subject)
        # x_batched: (batch, num_target_modules, 128)

        x_batched = self.mlp2(x_batched)
        # x_batched: (batch, num_target_modules, 128)
        x_batched = self.mlp3(x_batched)
        # x_batched: (batch, num_target_modules, 512)

        lora_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Route each module embedding to its dedicated output head
        for i, module_name in enumerate(self.module_names):
            x_module = x_batched[:, i, :]
            # x_module: (batch, 512)

            out_flat = self.heads[module_name](x_module)
            # out_flat: (batch, rank * in_dim + out_dim * rank)

            out_dim, in_dim = self.target_shapes[module_name]
            size_A = self.lora_rank * in_dim

            A_flat = out_flat[:, :size_A]
            # A_flat: (batch, rank * in_dim)
            B_flat = out_flat[:, size_A:]
            # B_flat: (batch, out_dim * rank)

            if "conv" in module_name:
                A = A_flat.view(-1, self.lora_rank, in_dim, 1, 1)
                # A: (batch, rank, in_dim, 1, 1)

                B = B_flat.view(-1, out_dim, self.lora_rank, 1, 1)
                # B: (batch, out_dim, rank, 1, 1)
            else:
                A = A_flat.view(-1, self.lora_rank, in_dim)
                # A: (batch, rank, in_dim)

                B = B_flat.view(-1, out_dim, self.lora_rank)
                # B: (batch, out_dim, rank)

            lora_weights[module_name] = (A, B)

        return lora_weights
