from typing import List

import torch
import torch.nn as nn
from hyper_har.config import BackboneConfig


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1
    ) -> None:
        super().__init__()
        padding = (dilation * (kernel_size - 1) + 1) // 2

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            (kernel_size, 1),
            padding=(padding, 0),
            dilation=(dilation, 1),
            groups=in_channels,
        )

        self.pointwise = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, time, sensor_channels)

        x = self.depthwise(x)
        # x: (batch, in_channels, time, sensor_channels)

        x = self.pointwise(x)
        # x: (batch, out_channels, time, sensor_channels)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        use_maxpool: bool = True,
        shortcut: bool = True,
    ) -> None:
        super().__init__()
        self.use_maxpool = use_maxpool
        self.shortcut = shortcut

        conv_layers: List[nn.Module] = [
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]

        if self.use_maxpool:
            conv_layers.append(nn.MaxPool2d((2, 1)))

        self.conv = nn.Sequential(*conv_layers)
        self.f_shortcut = self._create_shortcut(in_channels, out_channels)

    def _create_shortcut(self, in_channels: int, out_channels: int) -> nn.Module:
        layers: List[nn.Module] = []
        if in_channels != out_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, (1, 1)))
            layers.append(nn.BatchNorm2d(out_channels))
        if self.use_maxpool:
            layers.append(nn.MaxPool2d((2, 1)))
        if not layers:
            return nn.Identity()
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, time, sensor_channels)

        if self.shortcut:
            main = self.conv(x)
            # main: (batch, out_channels, time_or_time/2, sensor_channels)

            skip = self.f_shortcut(x)
            # skip: (batch, out_channels, time_or_time/2, sensor_channels)

            x = main + skip
            # x: (batch, out_channels, time_or_time/2, sensor_channels)

            return x

        return self.conv(x)


class TinierHAR(nn.Module):
    # Direct adaptation of zhaxidele/TinierHAR/models/TinierHAR.py

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        window_size: int,
        nb_conv_blocks: int | None = None,
        nb_filters: int | None = None,
        nb_units_gru: int | None = None,
        drop_prob: float | None = None,
        backbone_config: BackboneConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = backbone_config or BackboneConfig()

        self.input_channels = num_channels
        self.seq_length = window_size
        self.nb_classes = num_classes

        self.nb_conv_blocks = (
            nb_conv_blocks if nb_conv_blocks is not None else cfg.nb_conv_blocks
        )
        self.nb_units_gru = (
            nb_units_gru if nb_units_gru is not None else cfg.nb_units_gru
        )
        self.nb_filters = nb_filters if nb_filters is not None else cfg.nb_filters
        self.drop_prob = drop_prob if drop_prob is not None else cfg.drop_prob

        conv_blocks: List[nn.Module] = []
        conv_blocks.append(
            ConvBlock(
                1,
                self.nb_filters,
                kernel_size=5,
                dilation=1,
                use_maxpool=True,
                shortcut=True,
            )
        )
        conv_blocks.append(
            ConvBlock(
                self.nb_filters,
                2 * self.nb_filters,
                kernel_size=5,
                dilation=1,
                use_maxpool=True,
                shortcut=True,
            )
        )
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
            gru_input_dim = out.size(1) * out.size(3)

        self.dropout = nn.Dropout(self.drop_prob)

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=self.nb_units_gru,
            bidirectional=True,
            batch_first=True,
        )

        self.attention = nn.Linear(2 * self.nb_units_gru, 1)

        self.classifier = nn.Sequential(
            nn.Linear(2 * self.nb_units_gru, self.nb_classes)
        )

    def number_of_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch, 1, window_size, num_channels)

        x = self.conv_blocks(x)
        # x: (batch, conv_channels, conv_time, conv_sensor_channels)

        bsz, _, tlen, cin = x.shape
        x = x.permute(0, 2, 1, 3).reshape(bsz, tlen, -1)
        # x: (batch, conv_time, conv_channels * conv_sensor_channels)

        x = self.dropout(x)
        # x: (batch, conv_time, conv_channels * conv_sensor_channels)

        x, _ = self.gru(x)
        # x: (batch, conv_time, 2 * nb_units_gru)

        attn_weights = torch.softmax(self.attention(x), dim=1)
        # attn_weights: (batch, conv_time, 1)

        x = torch.sum(attn_weights * x, dim=1)
        # x: (batch, 2 * nb_units_gru)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, window_size, num_channels)

        x = self.encode(x)
        # x: (batch, 2 * nb_units_gru)

        logits = self.classifier(x)
        # logits: (batch, num_classes)

        return logits
