"""Waypoint predictor (CLAUDE.md section 9.2).

Section 9.2 specifies the architecture exactly:

    Input: seg logits (8 x 96 x 48, the 768x384 head output average-pooled 8x, detached,
    produced by the frozen best seg checkpoint) + speed scalar.

    4 conv blocks (32, 64, 96, 128 channels, stride 2, GroupNorm(8), SiLU) -> global average
    pool -> concat with speed MLP (1 -> 32) -> MLP 160 -> 128 -> 10 (5 waypoints x,y in
    metres). Parameter budget must print at startup and stay under 2 M.

Why the input is logits rather than an image: the ctrl net never sees pixels. It reasons over
the segmentation the perception model already produced, which is what makes it small enough to
run in the 33 ms frame budget alongside seg (section 11) and what lets its training be
GPU-cheap once the logits are precomputed.

GroupNorm rather than BatchNorm is load-bearing at this size. The control batch is 256
(section 9.2), but at inference the batch is 1, and BatchNorm's train/eval discrepancy on a
network this shallow shifts the predicted waypoints by tens of centimetres. GroupNorm computes
the same statistic in both modes.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

#: Section 9.2's channel schedule.
CONV_CHANNELS = (32, 64, 96, 128)
#: GroupNorm groups. 8 divides every channel count above.
NORM_GROUPS = 8
#: Speed is embedded to this width before being concatenated with the pooled features.
SPEED_EMBED = 32
#: Hidden width of the head MLP.
HEAD_HIDDEN = 128
#: 5 waypoints x (x, y) in metres.
NUM_WAYPOINTS = 5
OUTPUT_DIM = NUM_WAYPOINTS * 2

#: Section 9.2's hard budget. Asserted at construction, not merely printed: a model that
#: silently grew past it would not meet the 33 ms frame budget the export gate measures.
MAX_PARAMETERS = 2_000_000


class ConvBlock(nn.Module):
    """3x3 stride-2 conv -> GroupNorm(8) -> SiLU (section 9.2)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(NORM_GROUPS, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class CtrlNet(nn.Module):
    """Predicts 5 ego-frame waypoints from segmentation logits and speed.

    Output is (B, 5, 2) in metres: x forward, y left, matching the ego frame defined in
    section 8.4 and produced by data/waypoints.py.
    """

    def __init__(self, in_channels: int = 8, *, num_waypoints: int = NUM_WAYPOINTS) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_waypoints = num_waypoints

        blocks: list[nn.Module] = []
        channels = in_channels
        for out_channels in CONV_CHANNELS:
            blocks.append(ConvBlock(channels, out_channels))
            channels = out_channels
        self.features = nn.Sequential(*blocks)

        # Global average pool: the waypoint depends on the scene's overall layout, not on
        # where in the tensor a feature happens to sit, and pooling makes the head independent
        # of the input resolution.
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.speed_mlp = nn.Sequential(
            nn.Linear(1, SPEED_EMBED),
            nn.SiLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(CONV_CHANNELS[-1] + SPEED_EMBED, HEAD_HIDDEN),
            nn.SiLU(inplace=True),
            nn.Linear(HEAD_HIDDEN, num_waypoints * 2),
        )

        self._init_weights()

        count = self.parameter_count()
        if count >= MAX_PARAMETERS:
            raise ValueError(
                f"CtrlNet has {count:,} parameters, over section 9.2's {MAX_PARAMETERS:,} budget."
            )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # The final layer starts at zero, so an untrained model predicts "stay put" rather
        # than a random trajectory. With L1 loss and mostly-straight data (docs/DECISIONS.md
        # D028) this puts the first steps in a sane basin instead of unwinding a random guess.
        final = self.head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, logits: Tensor, speed: Tensor) -> Tensor:
        """logits: (B, C, H, W) segmentation logits. speed: (B,) or (B, 1) in m/s."""
        features = self.pool(self.features(logits)).flatten(1)

        if speed.dim() == 1:
            speed = speed.unsqueeze(1)
        embedded = self.speed_mlp(speed.float())

        out = self.head(torch.cat([features, embedded], dim=1))
        return out.reshape(-1, self.num_waypoints, 2)


def build_ctrlnet(in_channels: int = 8, *, verbose: bool = True) -> CtrlNet:
    """Construct CtrlNet and print its parameter budget (section 9.2 requires this)."""
    model = CtrlNet(in_channels)
    count = model.parameter_count()
    if verbose:
        logger.info(
            "CtrlNet: %s parameters (section 9.2 budget: under %s, %.1f%% used)",
            f"{count:,}",
            f"{MAX_PARAMETERS:,}",
            100.0 * count / MAX_PARAMETERS,
        )
    return model
