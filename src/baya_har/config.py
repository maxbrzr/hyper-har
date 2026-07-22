from dataclasses import dataclass, field


@dataclass(frozen=True)
class BackboneConfig:
    nb_conv_blocks: int = 4
    nb_filters: int = 4
    nb_units_gru: int = 16
    drop_prob: float = 0.3


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 100
    patience: int = 10
    weight_decay: float = 0.0


@dataclass(frozen=True)
class BayaHARConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


DEFAULT_CONFIG = BayaHARConfig()
