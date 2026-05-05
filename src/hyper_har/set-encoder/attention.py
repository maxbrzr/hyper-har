import torch
from torch import nn

from hyper_har.backbone.tinierhar import TinierHAR


class AttentionSetEncoder(nn.Module):
    def __init__(
        self,
        backbone: TinierHAR,
        num_classes: int,
        label_embed_dim: int = 32,
        hidden_dim: int = 64,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        # 1. Freeze the pretrained TinierHAR backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.feature_dim = 2 * backbone.nb_units_gru

        # 2. Label Fusion Layers
        self.label_embedding = nn.Embedding(num_classes, label_embed_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.feature_dim + label_embed_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # 3. Learnable Queries (One unique query per class)
        self.class_queries = nn.Parameter(torch.randn(num_classes, 1, hidden_dim))

        # Set Transformer Attention (batch_first=True for (B, L, E) shape)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone.encode(x)
        return out

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

            # Apply Cross-Attention: Q=query, K=z, V=z
            attn_out, _ = self.mha(
                query, z, z, key_padding_mask=key_padding_mask
            )  # Output shape: (B, 1, hidden_dim)

            # Remove the sequence dimension length of 1
            class_representations.append(attn_out.squeeze(1))  # (B, hidden_dim)

        # Concatenate all representations: (B, num_classes * hidden_dim)
        c_subject = torch.cat(class_representations, dim=-1)

        return c_subject
