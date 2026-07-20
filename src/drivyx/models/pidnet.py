"""PIDNet-S (CLAUDE.md section 9.1).

Implemented in-repo per section 9.1, loading the ImageNet backbone from `pretrained/`.

PIDNet is a three-branch network named for the PID controller it borrows from:

  P (proportional) branch  keeps high resolution detail, responds to the present
  I (integral) branch      aggregates context over a large receptive field, the accumulated view
  D (derivative) branch    detects boundaries, responds to change

The P and I branches exchange information twice through PagFM (pixel-attention-guided fusion)
modules, and all three merge in the Bag module at the end. The D branch's boundary map is what
gates that merge, which is why the boundary loss in losses.py is not decoration: it supervises
the signal that decides how P and I combine.

# Module naming

Section 16 lists "exact PIDNet checkpoint key names" as a runtime unknown. The answer, measured
from the checkpoint on this device (docs/DECISIONS.md D024), is mmsegmentation's layout:

    stem.0.conv.weight              not  conv1.0.weight
    i_branch_layers.0.0.conv1.conv  not  layer1.0.conv1
    pag_1.f_i.conv.weight           not  pag1.f_x.weight

So the pretrained modules here are named to match the file that exists, rather than the
upstream repository's convention which no file on this disk uses.

Two quirks of that checkpoint are load-bearing and cannot be guessed:

  - `stem.0.conv` and `stem.1.conv` carry a bias **and** a BatchNorm. Every other conv omits
    the bias, as is conventional with BN. A uniform bias=False fails to load those two.
  - The checkpoint contains no `d_branch_layers`, no SPP, and no head: ImageNet pretraining
    covers the stem and the P/I branches only. Those modules initialise randomly, which is
    normal, and load_pretrained() reports exactly which parameters came from the file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)

#: PIDNet-S hyperparameters. Section 9.1 names the S variant; these are its published widths.
STEM_CHANNELS = 32
PPM_CHANNELS = 96
HEAD_CHANNELS = 128
NUM_STEM_BLOCKS = 2
NUM_BRANCH_BLOCKS = 3

#: BN epsilon and momentum matching the checkpoint's training regime. Using torch's defaults
#: (1e-5 / 0.1) instead would shift the running statistics the pretrained BNs carry.
BN_EPS = 1e-3
BN_MOMENTUM = 0.01

#: Fraction of the loss weight on the auxiliary P-branch head during training (section 9.1's
#: "PIDNet boundary loss with its standard weighting").
AUX_WEIGHT = 0.4
BOUNDARY_WEIGHT = 20.0
BOUNDARY_AWARE_WEIGHT = 1.0


class ConvBN(nn.Module):
    """Conv + BatchNorm (+ optional ReLU), with submodules named `conv` and `bn`.

    The naming is not cosmetic: it is what the checkpoint's keys expect
    (`stem.0.conv.weight`, `stem.0.bn.weight`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        act: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn(self.conv(x))
        return self.act(x) if self.act is not None else x


class BasicBlock(nn.Module):
    """Residual block: 3x3 -> 3x3, with an optional 1x1 downsample on the identity path.

    `act_out` controls the ReLU *after* the residual add. PIDNet omits it on the last block of
    several stages so the branch fusion sees pre-activation features, which is why it is a
    parameter rather than always-on.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        *,
        stride: int = 1,
        downsample: nn.Module | None = None,
        act_out: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBN(in_channels, channels, 3, stride=stride, padding=1, act=True)
        self.conv2 = ConvBN(channels, channels, 3, padding=1, act=False)
        self.downsample = downsample
        self.act_out = act_out
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv2(self.conv1(x)) + identity
        return self.relu(out) if self.act_out else out


class Bottleneck(nn.Module):
    """1x1 -> 3x3 -> 1x1 residual block. Expansion 2, not ResNet's 4: PIDNet-S is narrow."""

    expansion = 2

    def __init__(
        self,
        in_channels: int,
        channels: int,
        *,
        stride: int = 1,
        downsample: nn.Module | None = None,
        act_out: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBN(in_channels, channels, 1, act=True)
        self.conv2 = ConvBN(channels, channels, 3, stride=stride, padding=1, act=True)
        self.conv3 = ConvBN(channels, channels * self.expansion, 1, act=False)
        self.downsample = downsample
        self.act_out = act_out
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv3(self.conv2(self.conv1(x))) + identity
        return self.relu(out) if self.act_out else out


class PagFM(nn.Module):
    """Pixel-attention-guided fusion (the `pag_1` / `pag_2` modules).

    Fuses the I branch into the P branch by asking, per pixel, how much the two agree. Both
    are projected to a shared embedding, and their cosine-like similarity (a sigmoid of their
    dot product over channels) becomes the mixing weight:

        sigma = sigmoid(sum_c(f_p(p) * f_i(i)))
        out   = (1 - sigma) * p + sigma * i_upsampled

    Where the branches agree the context branch is trusted and blended in; where they disagree
    the detail branch is preserved. This is what stops the low-resolution I branch from
    smearing away the thin structures P exists to keep.
    """

    def __init__(self, in_channels: int, channels: int) -> None:
        super().__init__()
        self.f_i = ConvBN(in_channels, channels, 1, act=False)
        self.f_p = ConvBN(in_channels, channels, 1, act=False)

    def forward(self, p: Tensor, i: Tensor) -> Tensor:
        size = p.shape[2:]
        i_up = F.interpolate(i, size=size, mode="bilinear", align_corners=False)
        sigma = torch.sigmoid(torch.sum(self.f_p(p) * self.f_i(i_up), dim=1, keepdim=True))
        return (1 - sigma) * p + sigma * i_up


class PAPPM(nn.Module):
    """Parallel aggregation pyramid pooling, the I branch's context aggregator (`spp`).

    Pools the deepest features at several scales (5x5/9x9/17x17 strided, plus global), lifts
    each back to the input resolution, and fuses them in one grouped convolution. Pooling at
    multiple scales in parallel is what gives the I branch a receptive field approaching the
    whole image without the depth that would cost.

    Not in the ImageNet checkpoint: this is segmentation-specific and initialises randomly.
    """

    def __init__(self, in_channels: int, branch_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branch_channels = branch_channels

        self.scale0 = ConvBN(in_channels, branch_channels, 1, act=True)
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(5, stride=2, padding=2), ConvBN(in_channels, branch_channels, 1, act=True)
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(9, stride=4, padding=4), ConvBN(in_channels, branch_channels, 1, act=True)
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(17, stride=8, padding=8), ConvBN(in_channels, branch_channels, 1, act=True)
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), ConvBN(in_channels, branch_channels, 1, act=True)
        )

        # One grouped 3x3 over the concatenated scales: each group refines its own scale before
        # the compression mixes them, which is cheaper than four separate convolutions.
        self.processes = ConvBN(
            branch_channels * 4, branch_channels * 4, 3, padding=1, act=True, groups=4
        )
        self.compression = ConvBN(branch_channels * 5, out_channels, 1, act=False)
        self.shortcut = ConvBN(in_channels, out_channels, 1, act=False)

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[2:]
        x0 = self.scale0(x)
        scales = [
            F.interpolate(branch(x), size=size, mode="bilinear", align_corners=False) + x0
            for branch in (self.scale1, self.scale2, self.scale3, self.scale4)
        ]
        processed = self.processes(torch.cat(scales, dim=1))
        return self.compression(torch.cat([x0, processed], dim=1)) + self.shortcut(x)


class LightBag(nn.Module):
    """Boundary-attention-guided fusion of P, I, and D (the `dfm` module).

    The D branch's boundary map, squashed to (0, 1), decides pixel by pixel whether to trust
    the detail branch or the context branch:

        out = conv_p((1 - sigma) * i + p) + conv_i(sigma * p + i)

    Near a boundary sigma approaches 1 and P dominates, keeping the edge sharp. In a region
    interior sigma approaches 0 and I dominates, keeping the label consistent. This is the
    mechanism the whole three-branch design exists to serve.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_p = ConvBN(in_channels, out_channels, 1, act=False)
        self.conv_i = ConvBN(in_channels, out_channels, 1, act=False)

    def forward(self, p: Tensor, i: Tensor, d: Tensor) -> Tensor:
        sigma = torch.sigmoid(d)
        return self.conv_p((1 - sigma) * i + p) + self.conv_i(sigma * p + i)


class SegHead(nn.Module):
    """BN -> 3x3 -> BN -> 1x1 classifier, used for the main, auxiliary, and boundary heads."""

    def __init__(self, in_channels: int, channels: int, out_channels: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(channels, out_channels, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(self.relu(self.bn1(x)))
        return self.conv2(self.relu(self.bn2(x)))


@dataclass(frozen=True)
class LoadReport:
    """What load_pretrained actually loaded.

    Reported rather than logged-and-forgotten because a loader that silently tolerates missing
    keys is how a randomly-initialised backbone gets trained while everyone believes it is
    pretrained (docs/DECISIONS.md D024).
    """

    loaded: int
    total: int
    skipped_shape_mismatch: list[str]
    missing_in_checkpoint: list[str]
    unexpected_in_checkpoint: list[str]

    @property
    def fraction(self) -> float:
        return self.loaded / self.total if self.total else 0.0

    def summary(self) -> str:
        return (
            f"loaded {self.loaded}/{self.total} tensors ({100 * self.fraction:.1f}%), "
            f"{len(self.unexpected_in_checkpoint)} unused in file, "
            f"{len(self.skipped_shape_mismatch)} shape mismatches"
        )


def _make_layer(
    block: type[nn.Module],
    in_channels: int,
    channels: int,
    num_blocks: int,
    *,
    stride: int = 1,
    act_out_last: bool = False,
) -> nn.Sequential:
    """Build a residual stage.

    The identity path gets a 1x1 downsample whenever the shape changes, either because the
    stride reduces resolution or because the block's expansion changes the channel count.

    The stage's last block ends **without** its output ReLU by default. That is PIDNet's
    design, not an economy: every consumer of a stage's output either applies the activation
    itself or deliberately wants the pre-activation features, because the fusion modules
    (PagFM, Bag) read features that have not yet been rectified.

    Leaving the activation on here also breaks training outright. The stem applies an explicit
    `nn.ReLU(inplace=True)` after each stage; if the stage's last block has already rectified
    in place, that second ReLU mutates a tensor autograd still needs, and backward fails with
    "a variable needed for gradient computation has been modified by an inplace operation".
    """
    downsample = None
    if stride != 1 or in_channels != channels * block.expansion:
        downsample = ConvBN(in_channels, channels * block.expansion, 1, stride=stride, act=False)

    layers: list[nn.Module] = [block(in_channels, channels, stride=stride, downsample=downsample)]
    in_channels = channels * block.expansion
    for i in range(1, num_blocks):
        last = i == num_blocks - 1
        layers.append(block(in_channels, channels, act_out=(not last) or act_out_last))
    return nn.Sequential(*layers)


class PIDNet(nn.Module):
    """PIDNet-S for `num_classes` collapsed classes (CLAUDE.md sections 7, 9.1).

    Outputs at 1/8 of the input resolution. Section 9.2 consumes the 768x384 head output,
    which is 96x48, average-pooled 8x, so the stride is part of the contract with the ctrl
    net and not an implementation detail to change freely.

    In training mode forward() returns (aux_logits, logits, boundary_logits): the two
    auxiliary outputs feed the losses in losses.py and are discarded at inference, which is
    why eval mode returns the main logits alone and the ONNX export (section 11) sees a single
    output.

    Training requires batch >= 2. The PAPPM pools globally to 1x1, and BatchNorm in training
    mode cannot normalise a single value per channel; at batch 1 it raises "Expected more than
    1 value per channel when training". Section 9.1 trains at batch 16, so this is a constraint
    rather than a defect. Inference at batch 1 is unaffected, which is what section 11's static
    batch-1 export needs.
    """

    def __init__(self, num_classes: int = 8, *, channels: int = STEM_CHANNELS) -> None:
        super().__init__()
        self.num_classes = num_classes

        # --- stem: 1/8 resolution, shared by every branch ---
        # stem.0 and stem.1 carry a conv bias as well as BN. This is not a style choice: it is
        # what the checkpoint contains, and bias=False here fails to load it (D024).
        # The stem halves resolution twice at `channels` width, runs two blocks there, and only
        # then widens to `channels * 2` while striding again: 1/8 resolution, 64 channels out.
        # Widening at stem.1 instead would be the natural guess and is wrong; the checkpoint
        # keeps stem.1 at 32 -> 32 (D024).
        self.stem = nn.Sequential(
            ConvBN(3, channels, 3, stride=2, padding=1, bias=True, act=True),
            ConvBN(channels, channels, 3, stride=2, padding=1, bias=True, act=True),
            _make_layer(BasicBlock, channels, channels, NUM_STEM_BLOCKS),
            nn.ReLU(inplace=True),
            _make_layer(BasicBlock, channels, channels * 2, NUM_STEM_BLOCKS, stride=2),
            nn.ReLU(inplace=True),
        )

        # --- I branch: 1/16 -> 1/32 -> 1/64, widening to 512 ---
        self.i_branch_layers = nn.ModuleList(
            [
                _make_layer(BasicBlock, channels * 2, channels * 4, NUM_BRANCH_BLOCKS, stride=2),
                _make_layer(BasicBlock, channels * 4, channels * 8, NUM_BRANCH_BLOCKS, stride=2),
                _make_layer(Bottleneck, channels * 8, channels * 8, 2, stride=2),
            ]
        )

        # --- P branch: stays at 1/8 and at `channels * 2` width throughout ---
        self.p_branch_layers = nn.ModuleList(
            [
                _make_layer(BasicBlock, channels * 2, channels * 2, NUM_STEM_BLOCKS),
                _make_layer(BasicBlock, channels * 2, channels * 2, NUM_STEM_BLOCKS),
                _make_layer(Bottleneck, channels * 2, channels * 2, 1),
            ]
        )

        # --- P <- I exchange ---
        self.compression_1 = ConvBN(channels * 4, channels * 2, 1, act=False)
        self.compression_2 = ConvBN(channels * 8, channels * 2, 1, act=False)
        self.pag_1 = PagFM(channels * 2, channels)
        self.pag_2 = PagFM(channels * 2, channels)

        # --- D branch: boundaries. Not in the ImageNet checkpoint (D024). ---
        # Three stages, narrowing to `channels` and then widening back to `channels * 4`, so
        # its output matches P and I at the Bag fusion. The final stage is not optional: the
        # Bag gates i and p by sigmoid(d) elementwise, so d must carry the same width, and a
        # two-stage D branch fails there with a 64-vs-128 broadcast error.
        self.d_branch_layers = nn.ModuleList(
            [
                _make_layer(BasicBlock, channels * 2, channels, 1),
                _make_layer(Bottleneck, channels, channels, 1),
                _make_layer(Bottleneck, channels * 2, channels * 2, 1),
            ]
        )
        self.diff_1 = ConvBN(channels * 4, channels, 3, padding=1, act=False)
        self.diff_2 = ConvBN(channels * 8, channels * 2, 3, padding=1, act=False)

        # --- fusion and heads. Not in the ImageNet checkpoint. ---
        self.spp = PAPPM(channels * 16, PPM_CHANNELS, channels * 4)
        self.dfm = LightBag(channels * 4, channels * 4)

        self.i_head = SegHead(channels * 4, HEAD_CHANNELS, num_classes)
        self.p_head = SegHead(channels * 2, HEAD_CHANNELS, num_classes)
        self.d_head = SegHead(channels * 2, channels, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming for convs, unit-scale for BN.

        Applied to everything, then overwritten for the pretrained modules by
        load_pretrained(). Doing it in this order means a module the checkpoint does not
        cover (the D branch, the SPP, the heads) is still deliberately initialised rather than
        left to torch's per-layer defaults.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        width, height = x.shape[-1] // 8, x.shape[-2] // 8

        x = self.stem(x)

        # Stage 1: I widens and strides; P and D read from I through compression/diff.
        i = self.i_branch_layers[0](x)
        p = self.p_branch_layers[0](x)
        d = self.d_branch_layers[0](x)

        i = F.relu(i)
        p = self.pag_1(p, self.compression_1(i))
        d = d + F.interpolate(
            self.diff_1(i), size=(height, width), mode="bilinear", align_corners=False
        )
        if self.training:
            temp_p = p

        # Stage 2.
        i = self.i_branch_layers[1](i)
        p = self.p_branch_layers[1](F.relu(p))
        d = self.d_branch_layers[1](F.relu(d))

        i = F.relu(i)
        p = self.pag_2(p, self.compression_2(i))
        d = d + F.interpolate(
            self.diff_2(i), size=(height, width), mode="bilinear", align_corners=False
        )
        if self.training:
            temp_d = d

        # Stage 3: P and D finish at `channels * 4`; I aggregates context and returns to 1/8.
        # All three must agree in width here: the Bag below gates them against each other
        # elementwise.
        p = self.p_branch_layers[2](F.relu(p))
        d = self.d_branch_layers[2](F.relu(d))
        i = self.i_branch_layers[2](i)
        i = F.interpolate(self.spp(i), size=(height, width), mode="bilinear", align_corners=False)

        logits = self.i_head(self.dfm(p, i, d))

        if self.training:
            return self.p_head(temp_p), logits, self.d_head(temp_d)
        return logits


#: Modules the ImageNet checkpoint is expected to cover. Anything outside this set is
#: segmentation-specific and initialises randomly by design (D024). Held explicitly so that a
#: checkpoint which unexpectedly omits a *backbone* module is an error, while one that omits
#: the head is not.
PRETRAINED_MODULES = (
    "stem",
    "i_branch_layers",
    "p_branch_layers",
    "compression_1",
    "compression_2",
    "pag_1",
    "pag_2",
)

#: Minimum fraction of the backbone that must load. A checkpoint matching less than this is
#: not the PIDNet-S ImageNet backbone, whatever its filename says, and training on it would
#: silently be training from scratch.
MIN_PRETRAINED_FRACTION = 0.90


def load_pretrained(model: PIDNet, path: Path, *, strict: bool = True) -> LoadReport:
    """Load the ImageNet backbone, reporting exactly what came from the file.

    Section 9.1: "loading the user-supplied ImageNet backbone from pretrained/. Abort with the
    download hint if missing."

    `strict` refers to the *backbone*, not to the whole model: the D branch, SPP, and heads are
    legitimately absent from an ImageNet checkpoint, so demanding a total match would reject
    every valid file. What strict does demand is that the backbone modules load essentially
    completely. Tolerating a near-empty match is how a randomly-initialised network gets
    trained for eight hours while its log says "loaded pretrained backbone".
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"PIDNet-S backbone not found at {path}. Download it and place it under "
            "pretrained/. See 'drivyx verify-data' for the hint, or "
            "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/ for a "
            "mirror with a checksummed filename (docs/DECISIONS.md D024)."
        )

    raw = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError(f"{path} does not contain a state dict.")

    model_sd = model.state_dict()
    to_load: dict[str, Tensor] = {}
    shape_mismatch: list[str] = []
    unexpected: list[str] = []

    for key, tensor in checkpoint.items():
        if key not in model_sd:
            unexpected.append(key)
        elif tuple(model_sd[key].shape) != tuple(tensor.shape):
            shape_mismatch.append(
                f"{key}: model {tuple(model_sd[key].shape)} vs file {tuple(tensor.shape)}"
            )
        else:
            to_load[key] = tensor

    backbone_keys = {k for k in model_sd if k.split(".")[0] in PRETRAINED_MODULES}
    loaded_backbone = backbone_keys & set(to_load)
    missing_backbone = sorted(backbone_keys - set(to_load))

    report = LoadReport(
        loaded=len(loaded_backbone),
        total=len(backbone_keys),
        skipped_shape_mismatch=shape_mismatch,
        missing_in_checkpoint=missing_backbone,
        unexpected_in_checkpoint=unexpected,
    )

    if strict:
        if shape_mismatch:
            raise ValueError(
                f"{path} does not match this PIDNet-S: {len(shape_mismatch)} tensors have the "
                f"wrong shape, e.g. {shape_mismatch[:3]}. Refusing to load a partial match."
            )
        if report.fraction < MIN_PRETRAINED_FRACTION:
            raise ValueError(
                f"{path} covers only {100 * report.fraction:.1f}% of the backbone "
                f"({report.loaded}/{report.total} tensors), below the "
                f"{100 * MIN_PRETRAINED_FRACTION:.0f}% required. This is not a PIDNet-S "
                f"ImageNet checkpoint. Missing, e.g.: {missing_backbone[:5]}"
            )

    model.load_state_dict(to_load, strict=False)
    logger.info("pretrained backbone from %s: %s", path.name, report.summary())
    if report.unexpected_in_checkpoint:
        logger.warning(
            "%d tensors in %s were not used: %s",
            len(report.unexpected_in_checkpoint),
            path.name,
            report.unexpected_in_checkpoint[:5],
        )
    return report


def build_pidnet(
    num_classes: int = 8, *, pretrained: Path | None = None, strict: bool = True
) -> tuple[PIDNet, LoadReport | None]:
    """Construct PIDNet-S, optionally loading the ImageNet backbone."""
    model = PIDNet(num_classes)
    report = load_pretrained(model, pretrained, strict=strict) if pretrained else None
    return model, report
