from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyper_har.backbone.tinierhar import ConvBlock, TinierHAR


class TinierHARLocationModulator(nn.Module):
    """Subject-conditioned deltas for TinierHAR pointwise BN and attention query."""

    def __init__(
        self,
        subject_embedding_dim: int,
        pointwise_bn_channels: list[int],
        attention_feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        use_tanh_gating: bool = True,
        gamma_bound: float = 0.5,
        beta_bound: float = 1.0,
        enable_pointwise_bn: bool = True,
        enable_attention_query: bool = True,
        attention_adapter_type: str = "feature_film",
        attention_score_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if subject_embedding_dim <= 0:
            raise ValueError("subject_embedding_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if gamma_bound <= 0:
            raise ValueError("gamma_bound must be positive.")
        if beta_bound <= 0:
            raise ValueError("beta_bound must be positive.")
        if attention_score_bound <= 0:
            raise ValueError("attention_score_bound must be positive.")
        attention_adapter_type = attention_adapter_type.strip().lower()
        if attention_adapter_type not in {"feature_film", "score_delta"}:
            raise ValueError(
                "attention_adapter_type must be 'feature_film' or 'score_delta'."
            )
        if not enable_pointwise_bn and not enable_attention_query:
            raise ValueError("At least one modulation location must be enabled.")
        if enable_pointwise_bn and not pointwise_bn_channels:
            raise ValueError("pointwise_bn_channels cannot be empty.")
        if enable_attention_query and attention_feature_dim <= 0:
            raise ValueError("attention_feature_dim must be positive.")

        self.subject_embedding_dim = int(subject_embedding_dim)
        self.pointwise_bn_channels = [int(ch) for ch in pointwise_bn_channels]
        self.attention_feature_dim = int(attention_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_tanh_gating = bool(use_tanh_gating)
        self.gamma_bound = float(gamma_bound)
        self.beta_bound = float(beta_bound)
        self.enable_pointwise_bn = bool(enable_pointwise_bn)
        self.enable_attention_query = bool(enable_attention_query)
        self.attention_adapter_type = attention_adapter_type
        self.attention_score_bound = float(attention_score_bound)

        self.subject_norm = nn.LayerNorm(
            self.subject_embedding_dim,
            elementwise_affine=False,
        )
        self.shared = nn.Sequential(
            nn.Linear(self.subject_embedding_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.pointwise_param_dim = (
            2 * sum(self.pointwise_bn_channels) if self.enable_pointwise_bn else 0
        )
        self.attention_param_dim = (
            2 * self.attention_feature_dim if self.enable_attention_query else 0
        )
        self.pointwise_head: nn.Linear | None = None
        self.attention_head: nn.Linear | None = None
        self.subject_attention_context: nn.Linear | None = None
        self.attention_score_head: nn.Linear | None = None
        if self.enable_pointwise_bn:
            self.pointwise_head = nn.Linear(self.hidden_dim, self.pointwise_param_dim)
        if (
            self.enable_attention_query
            and self.attention_adapter_type == "feature_film"
        ):
            self.attention_head = nn.Linear(self.hidden_dim, self.attention_param_dim)
        if self.enable_attention_query and self.attention_adapter_type == "score_delta":
            self.subject_attention_context = nn.Linear(
                self.hidden_dim,
                self.attention_feature_dim,
            )
            self.attention_score_head = nn.Sequential(
                nn.Linear(2 * self.attention_feature_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, 1),
            )
        self.reset_to_identity()

    def reset_to_identity(self) -> None:
        for head in (self.pointwise_head, self.attention_head):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        if self.attention_score_head is not None:
            final = self.attention_score_head[-1]
            if not isinstance(final, nn.Linear):
                raise TypeError("Expected final attention score layer to be nn.Linear.")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _guard(
        self,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_tanh_gating:
            gamma = torch.tanh(gamma) * self.gamma_bound
            beta = torch.tanh(beta) * self.beta_bound
        return gamma, beta

    def forward(
        self,
        c_subject: torch.Tensor,
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor]],
        tuple[torch.Tensor, torch.Tensor] | None,
        torch.Tensor | None,
    ]:
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

        shared = self.shared(self.subject_norm(c_subject))
        pointwise_params: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self.pointwise_head is not None:
            params = self.pointwise_head(shared)
            cursor = 0
            for channels in self.pointwise_bn_channels:
                gamma = params[:, cursor : cursor + channels]
                cursor += channels
                beta = params[:, cursor : cursor + channels]
                cursor += channels
                pointwise_params.append(self._guard(gamma, beta))

        attention_params: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.attention_head is not None:
            gamma, beta = self.attention_head(shared).chunk(2, dim=-1)
            attention_params = self._guard(gamma, beta)

        attention_context = None
        if self.subject_attention_context is not None:
            attention_context = self.subject_attention_context(shared)

        return pointwise_params, attention_params, attention_context

    def attention_score_delta(
        self,
        x_seq: torch.Tensor,
        attention_context: torch.Tensor,
    ) -> torch.Tensor:
        if self.attention_score_head is None:
            raise RuntimeError("Attention score adapter is not enabled.")
        if x_seq.dim() != 3:
            raise ValueError(
                "Expected GRU activations with shape (batch, time, feature_dim), "
                f"got {tuple(x_seq.shape)}."
            )
        if x_seq.size(-1) != self.attention_feature_dim:
            raise ValueError(
                "Attention feature dim mismatch: "
                f"expected {self.attention_feature_dim}, got {x_seq.size(-1)}."
            )
        if attention_context.shape != (x_seq.size(0), self.attention_feature_dim):
            raise ValueError(
                "Attention context shape mismatch: "
                f"expected {(x_seq.size(0), self.attention_feature_dim)}, "
                f"got {tuple(attention_context.shape)}."
            )
        context = attention_context.unsqueeze(1).expand(-1, x_seq.size(1), -1)
        raw_delta = self.attention_score_head(torch.cat([x_seq, context], dim=-1))
        return torch.tanh(raw_delta) * self.attention_score_bound


class PointwiseCBNAttentionTinierHAR(nn.Module):
    """Frozen TinierHAR with subject CBN on pointwise conv BNs and pre-attention FiLM.

    The GRU and classifier stay untouched. Attention modulation is used only for
    computing the frozen attention scores; aggregation still uses the original
    GRU hidden states.
    """

    def __init__(
        self,
        base_model: TinierHAR,
        subject_embedding_dim: int,
        modulator_hidden_dim: int = 128,
        modulator_dropout: float = 0.0,
        use_tanh_gating: bool = True,
        gamma_bound: float = 0.5,
        beta_bound: float = 1.0,
        enable_pointwise_bn: bool = True,
        enable_attention_query: bool = True,
        pointwise_block_start: int = 0,
        attention_adapter_type: str = "feature_film",
        attention_score_bound: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.subject_embedding_dim = int(subject_embedding_dim)
        self.input_channels = base_model.input_channels
        self.seq_length = base_model.seq_length
        self.nb_classes = base_model.nb_classes
        self.nb_conv_blocks = base_model.nb_conv_blocks
        self.nb_units_gru = base_model.nb_units_gru
        self.nb_filters = base_model.nb_filters
        self.drop_prob = base_model.drop_prob
        self.enable_pointwise_bn = bool(enable_pointwise_bn)
        self.enable_attention_query = bool(enable_attention_query)
        self.pointwise_block_start = int(pointwise_block_start)
        if self.pointwise_block_start < 0:
            raise ValueError("pointwise_block_start must be non-negative.")

        self.pointwise_bns = self._collect_pointwise_batchnorms()
        self.pointwise_block_indices = [
            idx
            for idx in range(len(self.pointwise_bns))
            if idx >= self.pointwise_block_start
        ]
        if self.enable_pointwise_bn and not self.pointwise_block_indices:
            raise ValueError(
                "No pointwise BN blocks selected. Lower pointwise_block_start."
            )
        selected_pointwise_bns = [
            self.pointwise_bns[idx] for idx in self.pointwise_block_indices
        ]
        self.pointwise_bn_channels = [
            int(bn.num_features) for bn in selected_pointwise_bns
        ]
        self.attention_feature_dim = int(self.base_model.attention.in_features)
        self.modulator = TinierHARLocationModulator(
            subject_embedding_dim=subject_embedding_dim,
            pointwise_bn_channels=self.pointwise_bn_channels,
            attention_feature_dim=self.attention_feature_dim,
            hidden_dim=modulator_hidden_dim,
            dropout=modulator_dropout,
            use_tanh_gating=use_tanh_gating,
            gamma_bound=gamma_bound,
            beta_bound=beta_bound,
            enable_pointwise_bn=enable_pointwise_bn,
            enable_attention_query=enable_attention_query,
            attention_adapter_type=attention_adapter_type,
            attention_score_bound=attention_score_bound,
        )
        self.freeze_base_model()

    def _collect_pointwise_batchnorms(self) -> list[nn.BatchNorm2d]:
        bns: list[nn.BatchNorm2d] = []
        for block in self.base_model.conv_blocks:
            if not isinstance(block, ConvBlock):
                raise TypeError(
                    "Expected TinierHAR conv_blocks to contain ConvBlock modules."
                )
            bn = block.conv[1]
            if not isinstance(bn, nn.BatchNorm2d):
                raise TypeError("Expected ConvBlock main path BN at block.conv[1].")
            bns.append(bn)
        return bns

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

    @staticmethod
    def _conditional_batch_norm(
        x: torch.Tensor,
        bn: nn.BatchNorm2d,
        delta_gamma: torch.Tensor,
        delta_beta: torch.Tensor,
    ) -> torch.Tensor:
        if x.size(1) != bn.num_features:
            raise ValueError(
                f"BN channel mismatch: expected {bn.num_features}, got {x.size(1)}."
            )
        if delta_gamma.shape != (x.size(0), bn.num_features):
            raise ValueError(
                "delta_gamma shape mismatch: "
                f"expected {(x.size(0), bn.num_features)}, "
                f"got {tuple(delta_gamma.shape)}."
            )
        if delta_beta.shape != (x.size(0), bn.num_features):
            raise ValueError(
                "delta_beta shape mismatch: "
                f"expected {(x.size(0), bn.num_features)}, "
                f"got {tuple(delta_beta.shape)}."
            )

        normalized = F.batch_norm(
            x,
            running_mean=bn.running_mean,
            running_var=bn.running_var,
            weight=None,
            bias=None,
            training=False,
            momentum=0.0,
            eps=bn.eps,
        )
        base_gamma = bn.weight.view(1, -1, 1, 1)
        base_beta = bn.bias.view(1, -1, 1, 1)
        delta_gamma = delta_gamma.view(x.size(0), -1, 1, 1)
        delta_beta = delta_beta.view(x.size(0), -1, 1, 1)
        return (base_gamma + delta_gamma) * normalized + base_beta + delta_beta

    def _forward_block(
        self,
        block: ConvBlock,
        x: torch.Tensor,
        pointwise_params: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        main = block.conv[0](x)
        if pointwise_params is None:
            main = block.conv[1](main)
        else:
            gamma, beta = pointwise_params
            main = self._conditional_batch_norm(main, block.conv[1], gamma, beta)
        main = block.conv[2](main)
        if block.use_maxpool:
            main = block.conv[3](main)

        if block.shortcut:
            return main + block.f_shortcut(x)
        return main

    def extract_conv_sequence(
        self,
        x: torch.Tensor,
        pointwise_params: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if self.enable_pointwise_bn and len(pointwise_params) != len(
            self.pointwise_block_indices
        ):
            raise ValueError(
                "Expected one CBN parameter pair per selected TinierHAR conv block, "
                f"got {len(pointwise_params)}."
            )
        params_by_block = dict(zip(self.pointwise_block_indices, pointwise_params))
        for idx, block in enumerate(self.base_model.conv_blocks):
            params = params_by_block.get(idx) if self.enable_pointwise_bn else None
            x = self._forward_block(block, x, params)

        bsz, _, tlen, _ = x.shape
        x = x.permute(0, 2, 1, 3).reshape(bsz, tlen, -1)
        return self.base_model.dropout(x)

    def encode(self, x: torch.Tensor, c_subject: torch.Tensor) -> torch.Tensor:
        pointwise_params, attention_params, attention_context = self.modulator(
            c_subject
        )
        x_seq = self.extract_conv_sequence(x, pointwise_params)
        x_seq, _ = self.base_model.gru(x_seq)

        if attention_params is not None:
            gamma, beta = attention_params
            attention_input = x_seq
            attention_input = (1.0 + gamma.unsqueeze(1)) * x_seq + beta.unsqueeze(1)
            scores = self.base_model.attention(attention_input)
        else:
            scores = self.base_model.attention(x_seq)
        if attention_context is not None:
            scores = scores + self.modulator.attention_score_delta(
                x_seq,
                attention_context,
            )
        attn_weights = torch.softmax(scores, dim=1)
        return torch.sum(attn_weights * x_seq, dim=1)

    def forward(self, x: torch.Tensor, c_subject: torch.Tensor) -> torch.Tensor:
        return self.base_model.classifier(self.encode(x, c_subject))

    def forward_episode(
        self,
        x_query: torch.Tensor,
        c_subject: torch.Tensor,
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
