from __future__ import annotations

import torch
import torch.nn as nn

from hyper_har.backbone.tinierhar import TinierHAR

FILM_MODULATION_MODES = {"static", "dynamic_time"}


class FiLMModule(nn.Module):
    """Generate feature-wise scale and shift parameters from a subject embedding."""

    def __init__(
        self,
        subject_embedding_dim: int,
        feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        use_explosion_guard: bool = False,
        gamma_bound: float = 0.5,
        beta_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if subject_embedding_dim <= 0:
            raise ValueError("subject_embedding_dim must be positive.")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if gamma_bound <= 0:
            raise ValueError("gamma_bound must be positive.")
        if beta_bound <= 0:
            raise ValueError("beta_bound must be positive.")

        self.subject_embedding_dim = int(subject_embedding_dim)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_explosion_guard = bool(use_explosion_guard)
        self.gamma_bound = float(gamma_bound)
        self.beta_bound = float(beta_bound)
        self.subject_norm = nn.LayerNorm(
            self.subject_embedding_dim,
            elementwise_affine=False,
        )
        self.generator = nn.Sequential(
            nn.Linear(self.subject_embedding_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 2 * self.feature_dim),
        )
        self.reset_to_identity()

    def reset_to_identity(self) -> None:
        final = self.generator[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Expected final FiLM generator layer to be nn.Linear.")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor, c_subject: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                "Expected activations with shape (batch, time, feature_dim), "
                f"got {tuple(x.shape)}."
            )
        if c_subject.dim() != 2:
            raise ValueError(
                "Expected subject embedding with shape (batch, dim), "
                f"got {tuple(c_subject.shape)}."
            )
        if x.size(0) != c_subject.size(0):
            raise ValueError(
                "Batch mismatch between activations and subject embeddings: "
                f"{x.size(0)} != {c_subject.size(0)}."
            )
        if x.size(-1) != self.feature_dim:
            raise ValueError(
                f"Activation feature dim mismatch: expected {self.feature_dim}, "
                f"got {x.size(-1)}."
            )
        if c_subject.size(-1) != self.subject_embedding_dim:
            raise ValueError(
                "Subject embedding dim mismatch: "
                f"expected {self.subject_embedding_dim}, got {c_subject.size(-1)}."
            )

        params = self.generator(self.subject_norm(c_subject))
        gamma, beta = params.chunk(2, dim=-1)
        if self.use_explosion_guard:
            gamma = torch.tanh(gamma) * self.gamma_bound
            beta = torch.tanh(beta) * self.beta_bound
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return (1.0 + gamma) * x + beta


class HierarchicalFiLM(nn.Module):
    """Shared subject-conditioned FiLM generator with conv1 and pre-GRU heads."""

    def __init__(
        self,
        subject_embedding_dim: int,
        conv1_channels: int,
        conv_last_feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        use_explosion_guard: bool = False,
        gamma_bound: float = 0.5,
        beta_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if subject_embedding_dim <= 0:
            raise ValueError("subject_embedding_dim must be positive.")
        if conv1_channels <= 0:
            raise ValueError("conv1_channels must be positive.")
        if conv_last_feature_dim <= 0:
            raise ValueError("conv_last_feature_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if gamma_bound <= 0:
            raise ValueError("gamma_bound must be positive.")
        if beta_bound <= 0:
            raise ValueError("beta_bound must be positive.")

        self.subject_embedding_dim = int(subject_embedding_dim)
        self.conv1_channels = int(conv1_channels)
        self.conv_last_feature_dim = int(conv_last_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_explosion_guard = bool(use_explosion_guard)
        self.gamma_bound = float(gamma_bound)
        self.beta_bound = float(beta_bound)

        self.subject_norm = nn.LayerNorm(
            self.subject_embedding_dim,
            elementwise_affine=False,
        )
        self.shared_backbone = nn.Sequential(
            nn.Linear(self.subject_embedding_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.head_conv1 = nn.Linear(self.hidden_dim, 2 * self.conv1_channels)
        self.head_conv_last = nn.Linear(
            self.hidden_dim,
            2 * self.conv_last_feature_dim,
        )
        self.reset_to_identity()

    def reset_to_identity(self) -> None:
        for head in (self.head_conv1, self.head_conv_last):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def shared_features(self, c_subject: torch.Tensor) -> torch.Tensor:
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
        return self.shared_backbone(self.subject_norm(c_subject))

    def _guard(
        self,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_explosion_guard:
            gamma = torch.tanh(gamma) * self.gamma_bound
            beta = torch.tanh(beta) * self.beta_bound
        return gamma, beta

    def modulate_conv1(
        self,
        x: torch.Tensor,
        shared_features: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() not in {3, 4}:
            raise ValueError(
                "Expected conv1 activations with shape (batch, channels, time[, sensors]), "
                f"got {tuple(x.shape)}."
            )
        if x.size(1) != self.conv1_channels:
            raise ValueError(
                f"Conv1 channel mismatch: expected {self.conv1_channels}, got {x.size(1)}."
            )
        gamma, beta = self.head_conv1(shared_features).chunk(2, dim=-1)
        gamma, beta = self._guard(gamma, beta)
        view_shape = (x.size(0), x.size(1)) + (1,) * (x.dim() - 2)
        return (1.0 + gamma.view(view_shape)) * x + beta.view(view_shape)

    def modulate_conv_last(
        self,
        x: torch.Tensor,
        shared_features: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                "Expected pre-GRU activations with shape (batch, time, feature_dim), "
                f"got {tuple(x.shape)}."
            )
        if x.size(-1) != self.conv_last_feature_dim:
            raise ValueError(
                "Pre-GRU feature dim mismatch: "
                f"expected {self.conv_last_feature_dim}, got {x.size(-1)}."
            )
        gamma, beta = self.head_conv_last(shared_features).chunk(2, dim=-1)
        gamma, beta = self._guard(gamma, beta)
        return (1.0 + gamma.unsqueeze(1)) * x + beta.unsqueeze(1)


class DynamicTimeVariantFiLM(nn.Module):
    """Generate per-time-step FiLM parameters from x_t and compressed subject context."""

    def __init__(
        self,
        subject_embedding_dim: int,
        feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        use_explosion_guard: bool = False,
        gamma_bound: float = 0.5,
        beta_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if subject_embedding_dim <= 0:
            raise ValueError("subject_embedding_dim must be positive.")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if gamma_bound <= 0:
            raise ValueError("gamma_bound must be positive.")
        if beta_bound <= 0:
            raise ValueError("beta_bound must be positive.")

        self.subject_embedding_dim = int(subject_embedding_dim)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_explosion_guard = bool(use_explosion_guard)
        self.gamma_bound = float(gamma_bound)
        self.beta_bound = float(beta_bound)

        self.subject_norm = nn.LayerNorm(
            self.subject_embedding_dim,
            elementwise_affine=False,
        )
        self.subject_compressor = nn.Linear(
            self.subject_embedding_dim,
            self.feature_dim,
        )
        self.generator = nn.Sequential(
            nn.Linear(2 * self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 2 * self.feature_dim),
        )
        self.reset_to_identity()

    def reset_to_identity(self) -> None:
        final = self.generator[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Expected final FiLM generator layer to be nn.Linear.")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor, c_subject: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                "Expected activations with shape (batch, time, feature_dim), "
                f"got {tuple(x.shape)}."
            )
        if c_subject.dim() != 2:
            raise ValueError(
                "Expected subject embedding with shape (batch, dim), "
                f"got {tuple(c_subject.shape)}."
            )
        if x.size(0) != c_subject.size(0):
            raise ValueError(
                "Batch mismatch between activations and subject embeddings: "
                f"{x.size(0)} != {c_subject.size(0)}."
            )
        if x.size(-1) != self.feature_dim:
            raise ValueError(
                f"Activation feature dim mismatch: expected {self.feature_dim}, "
                f"got {x.size(-1)}."
            )
        if c_subject.size(-1) != self.subject_embedding_dim:
            raise ValueError(
                "Subject embedding dim mismatch: "
                f"expected {self.subject_embedding_dim}, got {c_subject.size(-1)}."
            )

        c_compressed = self.subject_compressor(self.subject_norm(c_subject))
        c_expanded = c_compressed.unsqueeze(1).expand(-1, x.size(1), -1)
        params = self.generator(torch.cat([x, c_expanded], dim=-1))
        gamma, beta = params.chunk(2, dim=-1)
        if self.use_explosion_guard:
            gamma = torch.tanh(gamma) * self.gamma_bound
            beta = torch.tanh(beta) * self.beta_bound
        return (1.0 + gamma) * x + beta


class FiLMTinierHAR(nn.Module):
    """Frozen TinierHAR with trainable post-conv FiLM modulation."""

    def __init__(
        self,
        base_model: TinierHAR,
        subject_embedding_dim: int,
        film_hidden_dim: int = 128,
        film_dropout: float = 0.0,
        film_use_explosion_guard: bool = False,
        film_gamma_bound: float = 0.5,
        film_beta_bound: float = 1.0,
        film_enable_conv1: bool = False,
        film_modulation_mode: str = "static",
    ) -> None:
        super().__init__()
        film_modulation_mode = film_modulation_mode.strip().lower()
        if film_modulation_mode not in FILM_MODULATION_MODES:
            raise ValueError(
                "film_modulation_mode must be one of "
                f"{sorted(FILM_MODULATION_MODES)}, got {film_modulation_mode!r}."
            )
        if film_modulation_mode == "dynamic_time" and film_enable_conv1:
            raise ValueError(
                "film_enable_conv1=True is not supported with "
                "film_modulation_mode='dynamic_time'."
            )
        self.base_model = base_model
        self.film_enable_conv1 = bool(film_enable_conv1)
        self.film_modulation_mode = film_modulation_mode
        self.input_channels = base_model.input_channels
        self.seq_length = base_model.seq_length
        self.nb_classes = base_model.nb_classes
        self.nb_conv_blocks = base_model.nb_conv_blocks
        self.nb_units_gru = base_model.nb_units_gru
        self.nb_filters = base_model.nb_filters
        self.drop_prob = base_model.drop_prob

        was_training = self.base_model.training
        self.base_model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 1, self.seq_length, self.input_channels)
            out_conv1 = self.base_model.conv_blocks[0](dummy)
            out = self.base_model.conv_blocks(dummy)
            self.conv1_channels = int(out_conv1.size(1))
            self.conv_sequence_dim = int(out.size(1) * out.size(3))
            self.conv_time_steps = int(out.size(2))
        self.base_model.train(was_training)

        if self.film_modulation_mode == "dynamic_time":
            self.film = DynamicTimeVariantFiLM(
                subject_embedding_dim=subject_embedding_dim,
                feature_dim=self.conv_sequence_dim,
                hidden_dim=film_hidden_dim,
                dropout=film_dropout,
                use_explosion_guard=film_use_explosion_guard,
                gamma_bound=film_gamma_bound,
                beta_bound=film_beta_bound,
            )
        elif self.film_enable_conv1:
            self.film = HierarchicalFiLM(
                subject_embedding_dim=subject_embedding_dim,
                conv1_channels=self.conv1_channels,
                conv_last_feature_dim=self.conv_sequence_dim,
                hidden_dim=film_hidden_dim,
                dropout=film_dropout,
                use_explosion_guard=film_use_explosion_guard,
                gamma_bound=film_gamma_bound,
                beta_bound=film_beta_bound,
            )
        else:
            self.film = FiLMModule(
                subject_embedding_dim=subject_embedding_dim,
                feature_dim=self.conv_sequence_dim,
                hidden_dim=film_hidden_dim,
                dropout=film_dropout,
                use_explosion_guard=film_use_explosion_guard,
                gamma_bound=film_gamma_bound,
                beta_bound=film_beta_bound,
            )
        self.freeze_base_model()

    def freeze_base_model(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()
        self._set_base_batchnorm_eval()

    def _set_base_batchnorm_eval(self) -> None:
        for module in self.base_model.conv_blocks.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        self._set_base_batchnorm_eval()
        return self

    def number_of_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extract_conv_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if self.film_enable_conv1:
            raise RuntimeError(
                "extract_conv_sequence does not apply conv1 FiLM. Use encode instead."
            )
        x = self.base_model.conv_blocks(x)
        bsz, _, tlen, _ = x.shape
        x = x.permute(0, 2, 1, 3).reshape(bsz, tlen, -1)
        return self.base_model.dropout(x)

    def encode(self, x: torch.Tensor, c_subject: torch.Tensor) -> torch.Tensor:
        if self.film_enable_conv1:
            if not isinstance(self.film, HierarchicalFiLM):
                raise TypeError("Expected HierarchicalFiLM when conv1 FiLM is enabled.")
            shared = self.film.shared_features(c_subject)
            x = self.base_model.conv_blocks[0](x)
            x = self.film.modulate_conv1(x, shared)
            for block in self.base_model.conv_blocks[1:]:
                x = block(x)
            bsz, _, tlen, _ = x.shape
            x_seq = x.permute(0, 2, 1, 3).reshape(bsz, tlen, -1)
            x_seq = self.base_model.dropout(x_seq)
            x_seq = self.film.modulate_conv_last(x_seq, shared)
        else:
            x_seq = self.extract_conv_sequence(x)
            x_seq = self.film(x_seq, c_subject)
        x_seq, _ = self.base_model.gru(x_seq)
        attn_weights = torch.softmax(self.base_model.attention(x_seq), dim=1)
        return torch.sum(attn_weights * x_seq, dim=1)

    def forward(self, x: torch.Tensor, c_subject: torch.Tensor) -> torch.Tensor:
        return self.base_model.classifier(self.encode(x, c_subject))

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
        logits = self.forward(
            x_query.reshape(bsz * n_query, *x_query.shape[2:]), c_expanded
        )
        return logits.view(bsz, n_query, -1)
