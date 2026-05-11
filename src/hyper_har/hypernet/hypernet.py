from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from hyper_har.config import BackboneConfig, HyperNetConfig, SetEncoderConfig


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
        self,
        c_subject_dim: int,
        num_target_modules: int,
        dropout: float = 0.05,
        module_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_target_modules = num_target_modules
        self.input_norm = nn.LayerNorm(c_subject_dim, eps=1e-5)

        # 1. Dynamic Bottleneck Calculation
        # Halve the input, but keep it between 128 (floor) and 512 (ceiling)
        subj_proj_dim = max(128, min(512, c_subject_dim // 2))
        self.subject_encoder = nn.Linear(c_subject_dim, subj_proj_dim)

        self.module_encoder = nn.Sequential(
            nn.Embedding(num_target_modules, module_embed_dim),
            nn.LayerNorm(module_embed_dim, eps=1e-5),
        )

        # 2. Dynamic Mixer Input Size
        mixer_in_dim = subj_proj_dim + module_embed_dim

        self.mixer = nn.Sequential(
            nn.Linear(mixer_in_dim, 512),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            # Project back down to 128 so it matches MLPResidualBlock
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Dropout(p=dropout),
        )

        self.mlp1 = MLPResidualBlock(dim=128, hidden_dim=512, dropout=dropout)

    def forward(self, c_subject: torch.Tensor) -> torch.Tensor:
        # Input c_subject: (batch, c_subject_dim)
        c_subject = self.input_norm(c_subject)
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
        set_encoder_hidden_dim: int | None = None,  # From PrototypicalSetEncoder
        set_encoder_output_dim: int | None = None,
        nb_filters: int | None = None,  # From TinierHAR
        nb_units_gru: int | None = None,  # From TinierHAR
        lora_rank: int | None = None,
        lora_alpha: float | None = None,
        enable_conv1_adapter: bool | None = None,
        enable_conv_last_adapter: bool | None = None,
        dropout: float | None = None,
        backbone_config: BackboneConfig | None = None,
        set_encoder_config: SetEncoderConfig | None = None,
        hypernet_config: HyperNetConfig | None = None,
    ) -> None:
        super().__init__()
        backbone_cfg = backbone_config or BackboneConfig()
        set_encoder_cfg = set_encoder_config or SetEncoderConfig()
        hypernet_cfg = hypernet_config or HyperNetConfig()

        set_encoder_hidden_dim = (
            set_encoder_hidden_dim
            if set_encoder_hidden_dim is not None
            else set_encoder_cfg.hidden_dim
        )
        nb_filters = nb_filters if nb_filters is not None else backbone_cfg.nb_filters
        nb_units_gru = (
            nb_units_gru if nb_units_gru is not None else backbone_cfg.nb_units_gru
        )
        lora_rank = lora_rank if lora_rank is not None else hypernet_cfg.lora_rank
        lora_alpha = lora_alpha if lora_alpha is not None else hypernet_cfg.lora_alpha
        enable_conv1_adapter = (
            enable_conv1_adapter
            if enable_conv1_adapter is not None
            else hypernet_cfg.enable_conv1_adapter
        )
        enable_conv_last_adapter = (
            enable_conv_last_adapter
            if enable_conv_last_adapter is not None
            else hypernet_cfg.enable_conv_last_adapter
        )
        dropout = dropout if dropout is not None else hypernet_cfg.dropout
        self.lora_rank = lora_rank
        self.lora_alpha = float(lora_alpha)
        self.lora_scale_multiplier = float(hypernet_cfg.lora_scale_multiplier)
        self.lora_scale = float(
            (self.lora_alpha / max(1, self.lora_rank)) * self.lora_scale_multiplier
        )
        self.enable_conv1_adapter = bool(enable_conv1_adapter)
        self.enable_conv_last_adapter = bool(enable_conv_last_adapter)

        # Calculate exactly what the set encoder outputs. Newer set encoders may
        # append a global subject context in addition to class-wise prototypes.
        c_subject_dim = (
            int(set_encoder_output_dim)
            if set_encoder_output_dim is not None
            else (num_classes + int(set_encoder_cfg.include_global_context))
            * set_encoder_hidden_dim
        )
        self.c_subject_dim = c_subject_dim

        # Calculate exactly what TinierHAR shapes are at each point
        in_c_conv1 = 1
        out_c_conv1 = nb_filters
        in_c_conv_last = 2 * nb_filters
        out_c_conv_last = 2 * nb_filters

        # TinierHAR dynamically collapses the feature dimension before the GRU:
        # gru_input = conv_out_channels (2 * nb_filters) * sensor_channels (num_channels)
        gru_in_dim = (2 * nb_filters) * num_channels

        # 1. Define target modules and expected parameter shapes.
        # Map: name -> (out_features, in_features)
        self.target_shapes: Dict[str, Tuple[int, int]] = {}
        if self.enable_conv1_adapter:
            self.target_shapes["conv1_pointwise"] = (out_c_conv1, in_c_conv1)
        if self.enable_conv_last_adapter:
            self.target_shapes["conv_last_pointwise"] = (
                out_c_conv_last,
                in_c_conv_last,
            )
        self.target_shapes["gru_ih_fwd"] = (
            3 * nb_units_gru,
            gru_in_dim,
        )  # 3x for Reset, Update, New gates
        self.target_shapes["gru_ih_rev"] = (3 * nb_units_gru, gru_in_dim)
        self.target_shapes["classifier"] = (
            num_classes,
            2 * nb_units_gru,
        )  # 2x for Bidirectional
        if not self.target_shapes:
            raise ValueError("HyperNet must target at least one adapter module.")
        self.module_names = list(self.target_shapes.keys())

        # 2. Shared Backbone
        self.backbone = HyperNetBackbone(
            c_subject_dim,
            len(self.module_names),
            dropout=dropout,
            module_embed_dim=int(hypernet_cfg.module_embed_dim),
        )

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

            # Initialization:
            # Use microscopic non-zero weights to avoid zero-gradient trap into
            # the set encoder on the first optimization steps.
            # Keep heads near-zero to preserve base-model behavior at startup,
            # but avoid exact zeros so gradients can flow immediately.
            nn.init.normal_(head.weight, std=1e-4)
            nn.init.normal_(head.bias[:size_A], std=float(hypernet_cfg.output_bias_std))
            nn.init.normal_(head.bias[size_A:], std=float(hypernet_cfg.output_bias_std))

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
