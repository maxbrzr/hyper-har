import torch
from torch import nn

from hyper_har.backbone.tinierhar import TinierHAR
from hyper_har.config import SetEncoderConfig


class AttentionSetEncoder(nn.Module):
    def __init__(
        self,
        backbone: TinierHAR,
        num_classes: int,
        freeze_backbone: bool | None = None,
        backbone_train_mode: str = "freeze_all",
        force_conv_bn_eval: bool = True,
        label_embed_dim: int | None = None,
        hidden_dim: int | None = None,
        num_heads: int | None = None,
        set_encoder_config: SetEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = set_encoder_config or SetEncoderConfig()
        label_embed_dim = (
            label_embed_dim if label_embed_dim is not None else cfg.label_embed_dim
        )
        hidden_dim = hidden_dim if hidden_dim is not None else cfg.hidden_dim
        num_heads = num_heads if num_heads is not None else cfg.num_heads
        self.include_global_context = bool(cfg.include_global_context)
        self.backbone = backbone
        self.num_classes = num_classes
        if freeze_backbone is not None:
            self.backbone_train_mode = "freeze_all" if freeze_backbone else "unfreeze_all"
        else:
            self.backbone_train_mode = backbone_train_mode
        self.force_conv_bn_eval = force_conv_bn_eval

        # 1. Apply backbone parameter freezing strategy
        if self.backbone_train_mode == "freeze_all":
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif self.backbone_train_mode == "unfreeze_all":
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif self.backbone_train_mode == "freeze_conv_blocks":
            for param in self.backbone.parameters():
                param.requires_grad = True
            for param in self.backbone.conv_blocks.parameters():
                param.requires_grad = False
        else:
            raise ValueError(
                "Unsupported backbone_train_mode. Expected one of "
                "['freeze_all', 'unfreeze_all', 'freeze_conv_blocks']."
            )
        self._enforce_backbone_module_modes()

        self.feature_dim = 2 * backbone.nb_units_gru
        self.hidden_dim = hidden_dim
        self.output_dim = (
            (num_classes + 1) * hidden_dim
            if self.include_global_context
            else num_classes * hidden_dim
        )

        # 2. Label Fusion Layers
        self.label_embedding = nn.Embedding(num_classes, label_embed_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.feature_dim + label_embed_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        if self.include_global_context:
            self.raw_stats_mlp = nn.Sequential(
                nn.Linear(2 * backbone.input_channels, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.global_fusion_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # 3. Learnable Queries (One unique query per class)
        self.class_queries = nn.Parameter(torch.randn(num_classes, 1, hidden_dim))

        # Set Transformer Attention (batch_first=True for (B, L, E) shape)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_train_mode == "freeze_all":
            with torch.no_grad():
                out = self.backbone.encode(x)
            return out
        return self.backbone.encode(x)

    def _enforce_backbone_module_modes(self) -> None:
        # Keep frozen modules in eval mode so BatchNorm running stats don't update.
        if self.backbone_train_mode == "freeze_all":
            self.backbone.eval()
        elif self.backbone_train_mode == "freeze_conv_blocks":
            self.backbone.train()
            self.backbone.conv_blocks.eval()
        elif self.backbone_train_mode == "unfreeze_all":
            self.backbone.train()
        if self.force_conv_bn_eval:
            self._set_conv_block_batchnorm_eval()

    def _set_conv_block_batchnorm_eval(self) -> None:
        # Optional stabilization: keep conv-block BatchNorm in eval mode.
        for module in self.backbone.conv_blocks.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._enforce_backbone_module_modes()
        else:
            self.backbone.eval()
            if self.force_conv_bn_eval:
                self._set_conv_block_batchnorm_eval()
        return self

    def forward(self, x_support: torch.Tensor, y_support: torch.Tensor) -> torch.Tensor:
        B, N, C_in, T, S = x_support.shape

        # Flatten and Extract
        x_flat = x_support.view(B * N, C_in, T, S)
        h_flat = self.extract_features(x_flat)

        # Fuse with labels
        y_flat = y_support.view(-1)
        e_flat = self.label_embedding(y_flat)
        z_flat = self.fusion_mlp(torch.cat([h_flat, e_flat], dim=-1))

        # Reshape to set format: (B, N, hidden_dim)
        z = z_flat.view(B, N, -1)

        # 3. Class-wise Attention Aggregation
        class_representations = []
        for c in range(self.num_classes):
            # Create padding mask: PyTorch MHA ignores elements where mask is True.
            # We want to IGNORE samples that are NOT class 'c'.
            key_padding_mask = y_support != c  # (B, N)

            # Expand the learned query for this class across the batch
            query = (
                self.class_queries[c].unsqueeze(0).expand(B, -1, -1)
            )  # (B, 1, hidden_dim)

            # If all entries are masked for a given subject/class pair, MHA can produce NaNs.
            # Compute attention only for valid rows and keep a zero fallback otherwise.
            valid_rows = ~key_padding_mask.all(dim=1)  # (B,)
            attn_out = torch.zeros(
                (B, 1, z.size(-1)),
                device=z.device,
                dtype=z.dtype,
            )
            if valid_rows.any():
                attn_valid, _ = self.mha(
                    query[valid_rows],
                    z[valid_rows],
                    z[valid_rows],
                    key_padding_mask=key_padding_mask[valid_rows],
                )  # (B_valid, 1, hidden_dim)
                attn_out[valid_rows] = attn_valid

            # Extra guard against downstream NaN propagation from numerical edge cases.
            attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)

            # Remove the sequence dimension length of 1
            class_representations.append(attn_out.squeeze(1))  # (B, hidden_dim)

        # Concatenate all representations: (B, num_classes * hidden_dim)
        c_subject = torch.cat(class_representations, dim=-1)
        if self.include_global_context:
            z_global = z.mean(dim=1)
            raw = x_support.squeeze(2)
            raw_mean = raw.mean(dim=(1, 2))
            raw_std = raw.std(dim=(1, 2), unbiased=False)
            raw_context = self.raw_stats_mlp(torch.cat([raw_mean, raw_std], dim=-1))
            global_context = self.global_fusion_mlp(
                torch.cat([z_global, raw_context], dim=-1)
            )
            c_subject = torch.cat([c_subject, global_context], dim=-1)

        return c_subject
