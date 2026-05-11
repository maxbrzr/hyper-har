from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn as nn

from hyper_har.backbone.tinierhar import ConvBlock, TinierHAR
from hyper_har.config import BackboneConfig

FusionMode = Literal["temporal_tiling", "late"]


class ConditionedTinierHAR(nn.Module):
    """TinierHAR variant that conditions temporal features on a subject embedding.

    The model is intentionally close to :class:`TinierHAR` so an old pretrained
    checkpoint can seed the conv stack, GRU, attention layer, and classifier.
    For temporal fusion, the old GRU input weights are copied into the raw
    feature slice while the new subject-context slice starts at zero. This makes
    the initial conditioned model behave like the pretrained base model.
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        window_size: int,
        subject_embedding_dim: int,
        condition_dim: int = 64,
        fusion_mode: FusionMode = "temporal_tiling",
        nb_conv_blocks: int | None = None,
        nb_filters: int | None = None,
        nb_units_gru: int | None = None,
        drop_prob: float | None = None,
        backbone_config: BackboneConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = backbone_config or BackboneConfig()
        if fusion_mode not in {"temporal_tiling", "late"}:
            raise ValueError(
                "fusion_mode must be one of ['temporal_tiling', 'late'], "
                f"got {fusion_mode!r}."
            )
        if subject_embedding_dim <= 0:
            raise ValueError("subject_embedding_dim must be positive.")
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive.")

        self.input_channels = num_channels
        self.seq_length = window_size
        self.nb_classes = num_classes
        self.subject_embedding_dim = int(subject_embedding_dim)
        self.condition_dim = int(condition_dim)
        self.fusion_mode = fusion_mode

        self.nb_conv_blocks = (
            nb_conv_blocks if nb_conv_blocks is not None else cfg.nb_conv_blocks
        )
        self.nb_units_gru = (
            nb_units_gru if nb_units_gru is not None else cfg.nb_units_gru
        )
        self.nb_filters = nb_filters if nb_filters is not None else cfg.nb_filters
        self.drop_prob = drop_prob if drop_prob is not None else cfg.drop_prob

        conv_blocks: list[nn.Module] = [
            ConvBlock(
                1,
                self.nb_filters,
                kernel_size=5,
                dilation=1,
                use_maxpool=True,
                shortcut=True,
            ),
            ConvBlock(
                self.nb_filters,
                2 * self.nb_filters,
                kernel_size=5,
                dilation=1,
                use_maxpool=True,
                shortcut=True,
            ),
        ]
        for _ in range(self.nb_conv_blocks):
            conv_blocks.append(
                ConvBlock(
                    2 * self.nb_filters,
                    2 * self.nb_filters,
                    kernel_size=5,
                    dilation=1,
                    use_maxpool=False,
                    shortcut=True,
                )
            )
        self.conv_blocks = nn.Sequential(*conv_blocks)

        with torch.no_grad():
            dummy = torch.randn(1, 1, self.seq_length, self.input_channels)
            out = self.conv_blocks(dummy)
            self.conv_sequence_dim = int(out.size(1) * out.size(3))
            self.conv_time_steps = int(out.size(2))

        self.dropout = nn.Dropout(self.drop_prob)
        self.subject_norm = nn.LayerNorm(self.subject_embedding_dim)
        self.subject_projector = nn.Sequential(
            nn.Linear(self.subject_embedding_dim, self.condition_dim),
            nn.ReLU(),
            nn.Dropout(min(0.2, float(self.drop_prob))),
            nn.Linear(self.condition_dim, self.condition_dim),
            nn.ReLU(),
        )

        gru_input_dim = self.conv_sequence_dim
        if self.fusion_mode == "temporal_tiling":
            gru_input_dim += self.condition_dim
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=self.nb_units_gru,
            bidirectional=True,
            batch_first=True,
        )

        self.attention = nn.Linear(2 * self.nb_units_gru, 1)
        classifier_input_dim = 2 * self.nb_units_gru
        if self.fusion_mode == "late":
            classifier_input_dim += self.condition_dim
        self.classifier = nn.Sequential(nn.Linear(classifier_input_dim, self.nb_classes))

    def number_of_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_for_meta_training(
        self,
        *,
        freeze_conv_blocks: bool = True,
        train_subject_projector: bool = True,
        train_gru: bool = True,
        train_attention: bool = True,
        train_classifier: bool = True,
    ) -> None:
        for param in self.parameters():
            param.requires_grad = False

        if not freeze_conv_blocks:
            for param in self.conv_blocks.parameters():
                param.requires_grad = True
        if train_subject_projector:
            for module in (self.subject_norm, self.subject_projector):
                for param in module.parameters():
                    param.requires_grad = True
        if train_gru:
            for param in self.gru.parameters():
                param.requires_grad = True
        if train_attention:
            for param in self.attention.parameters():
                param.requires_grad = True
        if train_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = True

    def freeze_conv_batchnorm(self) -> None:
        for module in self.conv_blocks.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            conv_frozen = not any(p.requires_grad for p in self.conv_blocks.parameters())
            if conv_frozen:
                self.conv_blocks.eval()
        self.freeze_conv_batchnorm()
        return self

    def extract_conv_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        bsz, _, tlen, _ = x.shape
        x = x.permute(0, 2, 1, 3).reshape(bsz, tlen, -1)
        return self.dropout(x)

    def project_subject(self, c_subject: torch.Tensor) -> torch.Tensor:
        if c_subject.dim() != 2:
            raise ValueError(
                "Expected subject embedding with shape (batch, dim), "
                f"got {tuple(c_subject.shape)}."
            )
        if c_subject.size(-1) != self.subject_embedding_dim:
            raise ValueError(
                "Subject embedding dim mismatch: "
                f"expected {self.subject_embedding_dim}, got {c_subject.size(-1)}."
            )
        return self.subject_projector(self.subject_norm(c_subject))

    def encode(self, x: torch.Tensor, c_subject: torch.Tensor) -> torch.Tensor:
        x_seq = self.extract_conv_sequence(x)
        c = self.project_subject(c_subject)

        if self.fusion_mode == "temporal_tiling":
            c_seq = c.unsqueeze(1).expand(-1, x_seq.size(1), -1)
            x_seq = torch.cat([x_seq, c_seq], dim=-1)

        x_seq, _ = self.gru(x_seq)
        attn_weights = torch.softmax(self.attention(x_seq), dim=1)
        pooled = torch.sum(attn_weights * x_seq, dim=1)

        if self.fusion_mode == "late":
            pooled = torch.cat([pooled, c], dim=-1)
        return pooled

    def forward(self, x: torch.Tensor, c_subject: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x, c_subject))

    def forward_episode(
        self, x_query: torch.Tensor, c_subject: torch.Tensor
    ) -> torch.Tensor:
        if x_query.dim() != 5:
            raise ValueError(
                "Expected query tensor with shape (subjects, query, 1, time, sensors), "
                f"got {tuple(x_query.shape)}."
            )
        bsz, n_query = x_query.shape[:2]
        c_expanded = (
            c_subject.unsqueeze(1)
            .expand(-1, n_query, -1)
            .reshape(bsz * n_query, -1)
        )
        logits = self.forward(x_query.reshape(bsz * n_query, *x_query.shape[2:]), c_expanded)
        return logits.view(bsz, n_query, -1)

    def load_tinierhar_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor] | Mapping[str, Any],
        *,
        strict_base: bool = True,
    ) -> None:
        if "model" in state_dict and isinstance(state_dict["model"], Mapping):
            state_dict = state_dict["model"]  # type: ignore[assignment]
        if "state_dict" in state_dict and isinstance(state_dict["state_dict"], Mapping):
            state_dict = state_dict["state_dict"]  # type: ignore[assignment]

        own_state = self.state_dict()
        copied: set[str] = set()

        for key, value in state_dict.items():  # type: ignore[union-attr]
            if key.startswith("gru.weight_ih"):
                if key not in own_state:
                    continue
                target = own_state[key]
                if value.size(0) != target.size(0) or value.size(1) > target.size(1):
                    raise ValueError(
                        f"Cannot copy {key}: source={tuple(value.shape)}, "
                        f"target={tuple(target.shape)}."
                    )
                target.zero_()
                target[:, : value.size(1)].copy_(value)
                copied.add(key)
                continue

            if key == "classifier.0.weight" and self.fusion_mode == "late":
                target = own_state[key]
                if value.size(0) != target.size(0) or value.size(1) > target.size(1):
                    raise ValueError(
                        f"Cannot copy {key}: source={tuple(value.shape)}, "
                        f"target={tuple(target.shape)}."
                    )
                target.zero_()
                target[:, : value.size(1)].copy_(value)
                copied.add(key)
                continue

            if key in own_state and own_state[key].shape == value.shape:
                own_state[key].copy_(value)
                copied.add(key)

        missing_base = [
            key
            for key in state_dict.keys()  # type: ignore[union-attr]
            if (
                key.startswith("conv_blocks.")
                or key.startswith("gru.")
                or key.startswith("attention.")
                or key.startswith("classifier.")
            )
            and key not in copied
        ]
        if strict_base and missing_base:
            raise RuntimeError(
                "Could not copy all compatible TinierHAR base weights: "
                f"{missing_base}"
            )

    @classmethod
    def from_tinierhar(
        cls,
        base_model: TinierHAR,
        subject_embedding_dim: int,
        condition_dim: int = 64,
        fusion_mode: FusionMode = "temporal_tiling",
    ) -> "ConditionedTinierHAR":
        model = cls(
            num_channels=base_model.input_channels,
            num_classes=base_model.nb_classes,
            window_size=base_model.seq_length,
            subject_embedding_dim=subject_embedding_dim,
            condition_dim=condition_dim,
            fusion_mode=fusion_mode,
            nb_conv_blocks=base_model.nb_conv_blocks,
            nb_filters=base_model.nb_filters,
            nb_units_gru=base_model.nb_units_gru,
            drop_prob=base_model.drop_prob,
        )
        model.load_tinierhar_state_dict(base_model.state_dict())
        return model
