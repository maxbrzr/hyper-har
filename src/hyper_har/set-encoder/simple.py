import torch
import torch.nn as nn

from hyper_har.backbone.tinierhar import TinierHAR
from hyper_har.config import SetEncoderConfig


class PrototypicalSetEncoder(nn.Module):
    def __init__(
        self,
        backbone: TinierHAR,
        num_classes: int,
        label_embed_dim: int | None = None,
        hidden_dim: int | None = None,
        set_encoder_config: SetEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = set_encoder_config or SetEncoderConfig()
        label_embed_dim = (
            label_embed_dim if label_embed_dim is not None else cfg.label_embed_dim
        )
        hidden_dim = hidden_dim if hidden_dim is not None else cfg.hidden_dim

        self.backbone = backbone
        self.num_classes = num_classes

        # 1. Freeze the pretrained TinierHAR backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # TinierHAR outputs a feature vector of size 2 * nb_units_gru
        self.feature_dim = 2 * backbone.nb_units_gru

        # 2. Label Fusion Layers
        self.label_embedding = nn.Embedding(num_classes, label_embed_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.feature_dim + label_embed_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone.encode(x)
        return out

    def forward(self, x_support: torch.Tensor, y_support: torch.Tensor) -> torch.Tensor:
        # x_support: (Batch, N, 1, Window_Size, Num_Sensors)
        # y_support: (Batch, N)
        B, N, C_in, T, S = x_support.shape

        # Flatten for feature extraction
        x_flat = x_support.view(B * N, C_in, T, S)
        h_flat = self.extract_features(x_flat)  # (B*N, feature_dim)

        # Fuse with labels
        y_flat = y_support.view(-1)
        e_flat = self.label_embedding(y_flat)  # (B*N, label_embed_dim)
        z_flat = self.fusion_mlp(torch.cat([h_flat, e_flat], dim=-1))

        # Reshape back to sets
        z = z_flat.view(B, N, -1)  # (B, N, hidden_dim)

        # 3. Class-wise Prototypical Aggregation
        class_prototypes = []
        for c in range(self.num_classes):
            # Create a mask for class 'c': (B, N, 1)
            mask = (y_support == c).unsqueeze(-1).float()

            # Sum the features for class 'c'
            z_c_sum = (z * mask).sum(dim=1)  # (B, hidden_dim)

            # Divide by the count to get the mean (clamp to avoid div by zero)
            count = mask.sum(dim=1).clamp(min=1e-8)  # (B, 1)
            p_c = z_c_sum / count  # (B, hidden_dim)

            class_prototypes.append(p_c)

        # Concatenate all prototypes: (B, num_classes * hidden_dim)
        c_subject = torch.cat(class_prototypes, dim=-1)

        return c_subject
