from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BackboneConfig:
    nb_conv_blocks: int = 4
    nb_filters: int = 4
    nb_units_gru: int = 16
    drop_prob: float = 0.3


@dataclass(frozen=True)
class SetEncoderConfig:
    label_embed_dim: int = 32
    hidden_dim: int = 64
    num_heads: int = 4
    include_global_context: bool = True


@dataclass(frozen=True)
class HyperNetConfig:
    lora_rank: int = 8
    lora_alpha: float = 1.0
    enable_conv1_adapter: bool = True
    enable_conv_last_adapter: bool = True
    dropout: float = 0.05


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 100
    patience: int = 10
    weight_decay: float = 0.0


@dataclass(frozen=True)
class HyperHARConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    set_encoder: SetEncoderConfig = field(default_factory=SetEncoderConfig)
    hypernet: HyperNetConfig = field(default_factory=HyperNetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


DEFAULT_CONFIG = HyperHARConfig()
