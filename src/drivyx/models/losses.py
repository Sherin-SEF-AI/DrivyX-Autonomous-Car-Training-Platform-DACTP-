"""Segmentation and control losses (CLAUDE.md sections 9.1, 9.2).

Section 9.1 specifies "OHEM cross entropy (thresh 0.9, min_kept 26000) with class weights from
the shard histogram (w = 1/log(1.02 + freq), capped at 10x min), plus PIDNet boundary loss with
its standard weighting". Section 9.2 specifies "L1 on waypoints".
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)

#: Section 7: pixels excluded from every loss.
IGNORE_ID = 255

#: Section 9.1's OHEM parameters.
OHEM_THRESH = 0.9
OHEM_MIN_KEPT = 26000

#: PIDNet's standard loss weighting. The boundary head is weighted heavily because it predicts
#: a one-channel map that is ~3% positive: without the weight its gradient is swamped by the
#: semantic heads and the D branch never learns, which silently disables the Bag fusion the
#: whole architecture is built around.
AUX_WEIGHT = 0.4
BOUNDARY_WEIGHT = 20.0
BOUNDARY_AWARE_WEIGHT = 1.0

#: Pixels whose boundary probability exceeds this are treated as "on a boundary" by the
#: boundary-aware semantic loss.
BOUNDARY_AWARE_THRESH = 0.8


class OhemCrossEntropy(nn.Module):
    """Online hard example mining cross entropy (section 9.1).

    Plain cross entropy over a road scene is dominated by easy pixels: the middle of the road
    and the sky are correct within a few epochs and then contribute gradient forever, while the
    thin pole and the distant rider that actually matter are a rounding error in the mean.

    OHEM keeps only pixels the model is not already confident about:

      - a pixel is "hard" when its predicted probability for the true class is below `thresh`;
      - if fewer than `min_kept` pixels qualify, the `min_kept` hardest are kept anyway.

    The second rule is what makes it stable. Once the model is good, almost nothing exceeds the
    threshold, and a strict threshold rule would compute the loss over a handful of pixels and
    produce a wildly noisy gradient. `min_kept` puts a floor under the batch.

    thresh is applied to the *probability*, not the loss, which is why it reads as 0.9: keep
    pixels whose true-class probability is under 0.9.
    """

    def __init__(
        self,
        *,
        thresh: float = OHEM_THRESH,
        min_kept: int = OHEM_MIN_KEPT,
        class_weights: Tensor | None = None,
        ignore_index: int = IGNORE_ID,
    ) -> None:
        super().__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        # A buffer, not a plain attribute: it must follow the module across .to(device) and be
        # saved with the checkpoint, since the weights are data-derived (section 9.1) and a
        # resumed run must use the same ones.
        self.register_buffer(
            "class_weights", class_weights if class_weights is not None else torch.tensor([])
        )

    def _weights(self) -> Tensor | None:
        return self.class_weights if self.class_weights.numel() else None

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits, size=target.shape[-2:], mode="bilinear", align_corners=False
            )

        # Per-pixel loss, unreduced, in fp32: the selection below sorts these values, and
        # doing that under bf16 autocast would sort quantised losses and pick the wrong pixels.
        pixel_loss = F.cross_entropy(
            logits.float(),
            target,
            weight=self._weights(),
            ignore_index=self.ignore_index,
            reduction="none",
        )

        valid = target != self.ignore_index
        if not valid.any():
            # A crop entirely of ignore pixels. Return a real zero that is still connected to
            # the graph, so the optimiser step is a no-op rather than a crash.
            return logits.sum() * 0.0

        with torch.no_grad():
            prob = F.softmax(logits.float(), dim=1)
            # The ignore label is 255, which is not a valid gather index into C=8 channels and
            # trips a device-side assert. Substituting a real class for those pixels keeps the
            # gather in bounds; their probability is overwritten immediately below, so the
            # choice of substitute is irrelevant. Note that clamp(min=0) does NOT do this:
            # 255 is already positive.
            safe_target = torch.where(valid, target, torch.zeros_like(target))
            true_prob = prob.gather(1, safe_target.unsqueeze(1).long()).squeeze(1)
            # Ignore pixels must never be selected as "hard": their target is meaningless.
            true_prob = torch.where(valid, true_prob, torch.ones_like(true_prob))

        hard = valid & (true_prob < self.thresh)
        num_hard = int(hard.sum())
        num_valid = int(valid.sum())

        if num_hard >= self.min_kept:
            selected = pixel_loss[hard]
        else:
            # Not enough hard pixels: take the min_kept hardest valid ones by loss. Sorting the
            # loss is equivalent to sorting by (1 - true_prob) but also respects class weights,
            # which is what the loss is actually optimising.
            keep = min(self.min_kept, num_valid)
            masked = torch.where(valid, pixel_loss, torch.full_like(pixel_loss, -1.0))
            selected = masked.flatten().topk(keep).values

        return selected.mean()


class BoundaryLoss(nn.Module):
    """Weighted BCE for the D branch's one-channel boundary map (section 9.1).

    Boundary pixels are a few percent of the image, so an unweighted BCE is minimised by
    predicting "not a boundary" everywhere. The positive class is therefore reweighted by the
    observed negative/positive ratio *per batch*, which keeps the two classes' total gradient
    contributions comparable regardless of how much edge happens to be in the crop.

    Computed per batch rather than from a fixed constant because the ratio varies a lot between
    a crop of open road and a crop of a crowded junction.
    """

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        logits = logits.float()
        target = target.float()

        positive = target.sum()
        total = target.numel()
        if positive == 0 or positive == total:
            # No edges at all (or nothing but): the ratio is undefined, so fall back to plain
            # BCE rather than dividing by zero.
            return F.binary_cross_entropy_with_logits(logits, target)

        pos_weight = (total - positive) / positive
        return F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight.clamp(max=100.0)
        )


class BoundaryAwareCrossEntropy(nn.Module):
    """Semantic CE restricted to pixels the D branch believes are boundaries.

    PIDNet's third loss term. It closes the loop between the branches: the D branch says where
    the edges are, and this term makes the *semantic* head pay extra attention exactly there.
    Without it the boundary map is trained but never used to improve labels, which is most of
    the point of having a D branch.

    Pixels below the threshold are set to ignore rather than dropped, so the tensor shape is
    unchanged and the ordinary weighted CE does the rest.
    """

    def __init__(
        self,
        *,
        thresh: float = BOUNDARY_AWARE_THRESH,
        class_weights: Tensor | None = None,
        ignore_index: int = IGNORE_ID,
    ) -> None:
        super().__init__()
        self.thresh = thresh
        self.ignore_index = ignore_index
        self.register_buffer(
            "class_weights", class_weights if class_weights is not None else torch.tensor([])
        )

    def forward(self, logits: Tensor, target: Tensor, boundary_logits: Tensor) -> Tensor:
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        if boundary_logits.shape[-2:] != target.shape[-2:]:
            boundary_logits = F.interpolate(
                boundary_logits, size=target.shape[-2:], mode="bilinear", align_corners=False
            )

        boundary = torch.sigmoid(boundary_logits.float()).squeeze(1)
        on_edge = boundary > self.thresh

        filtered = torch.where(on_edge, target, torch.full_like(target, self.ignore_index))
        if not (filtered != self.ignore_index).any():
            # The D branch is confident there are no boundaries here, which is common early in
            # training when it is barely initialised. Contribute nothing rather than a NaN.
            return logits.sum() * 0.0

        weights = self.class_weights if self.class_weights.numel() else None
        return F.cross_entropy(
            logits.float(), filtered, weight=weights, ignore_index=self.ignore_index
        )


class PIDNetLoss(nn.Module):
    """The full section 9.1 objective.

        loss = aux_weight * ohem(p_head) + ohem(main) + bd_weight * bce(d_head)
               + bd_aware_weight * ce(main | boundary)

    The main head is unweighted (weight 1.0) and everything else is relative to it, which is
    PIDNet's standard weighting.
    """

    def __init__(
        self,
        *,
        class_weights: Tensor | None = None,
        thresh: float = OHEM_THRESH,
        min_kept: int = OHEM_MIN_KEPT,
        aux_weight: float = AUX_WEIGHT,
        boundary_weight: float = BOUNDARY_WEIGHT,
        boundary_aware_weight: float = BOUNDARY_AWARE_WEIGHT,
        ignore_index: int = IGNORE_ID,
    ) -> None:
        super().__init__()
        self.semantic = OhemCrossEntropy(
            thresh=thresh,
            min_kept=min_kept,
            class_weights=class_weights,
            ignore_index=ignore_index,
        )
        self.aux = OhemCrossEntropy(
            thresh=thresh,
            min_kept=min_kept,
            class_weights=class_weights,
            ignore_index=ignore_index,
        )
        self.boundary = BoundaryLoss()
        self.boundary_aware = BoundaryAwareCrossEntropy(
            class_weights=class_weights, ignore_index=ignore_index
        )
        self.aux_weight = aux_weight
        self.boundary_weight = boundary_weight
        self.boundary_aware_weight = boundary_aware_weight

    def forward(
        self,
        outputs: tuple[Tensor, Tensor, Tensor],
        target: Tensor,
        boundary_target: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Returns (total, parts) where parts is for the events stream (section 6.4)."""
        aux_logits, logits, boundary_logits = outputs

        loss_main = self.semantic(logits, target)
        loss_aux = self.aux(aux_logits, target)
        loss_bd = self.boundary(boundary_logits, boundary_target)
        loss_bd_aware = self.boundary_aware(logits, target, boundary_logits)

        total = (
            loss_main
            + self.aux_weight * loss_aux
            + self.boundary_weight * loss_bd
            + self.boundary_aware_weight * loss_bd_aware
        )
        parts = {
            "loss/main": float(loss_main.detach()),
            "loss/aux": float(loss_aux.detach()),
            "loss/boundary": float(loss_bd.detach()),
            "loss/boundary_aware": float(loss_bd_aware.detach()),
        }
        return total, parts


def compute_class_weights(class_pixels: list[int], *, cap: float = 10.0) -> Tensor:
    """Section 9.1: w = 1/log(1.02 + freq), capped at 10x the minimum.

    Shared with data/shards.py's class_weights (which writes them into the shard index); this
    is the tensor form the loss consumes. Kept as one formula in two places rather than an
    import, because shards.py must not depend on torch: section 3 requires the data pipeline to
    be usable without CUDA.
    """
    counts = torch.tensor(class_pixels, dtype=torch.float64)
    total = counts.sum()
    if total == 0:
        raise ValueError("cannot compute class weights: the histogram is empty")
    freq = counts / total
    weights = 1.0 / torch.log(1.02 + freq)
    return (weights.clamp(max=float(weights.min()) * cap)).float()


def boundary_target_from_mask(mask: Tensor, *, ignore_index: int = IGNORE_ID) -> Tensor:
    """Derive the D branch's target: 1 where the class label changes, else 0.

    IDD ships no boundary annotation, so the boundary ground truth is computed from the
    semantic mask itself. A pixel is on a boundary when any 4-neighbour carries a different
    class. Comparing shifted copies of the mask does this in four vectorised ops, with no
    morphology dependency and no Python loop over pixels.

    Ignore pixels never mark a boundary: the edge of an unlabelled region is an artifact of the
    annotation, not a real object edge, and training the D branch on it would teach it to fire
    on the ego-vehicle mask.
    """
    mask = mask.long()
    valid = mask != ignore_index

    edge = torch.zeros_like(mask, dtype=torch.bool)
    # Compare against each 4-neighbour by slicing rather than padding, which avoids inventing
    # a border class at the image edge.
    edge[:, :, :-1] |= (mask[:, :, :-1] != mask[:, :, 1:]) & valid[:, :, :-1] & valid[:, :, 1:]
    edge[:, :, 1:] |= (mask[:, :, 1:] != mask[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    edge[:, :-1, :] |= (mask[:, :-1, :] != mask[:, 1:, :]) & valid[:, :-1, :] & valid[:, 1:, :]
    edge[:, 1:, :] |= (mask[:, 1:, :] != mask[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]

    return edge.unsqueeze(1).float()


class WaypointL1(nn.Module):
    """L1 on waypoints (section 9.2), with the metrics section 9.2 asks for.

    L1 rather than L2 because a single bad GPS fix should not dominate: the squared error of a
    2 m outlier is 100x a 0.2 m one, and the waypoint targets are derived from real GPS.
    """

    def forward(self, pred: Tensor, target: Tensor) -> tuple[Tensor, dict[str, float]]:
        """pred and target are (B, 5, 2) in metres, ego frame."""
        loss = F.l1_loss(pred, target)

        with torch.no_grad():
            # Displacement error is the Euclidean distance per waypoint, not the L1 the loss
            # uses: ADE/FDE are defined in metres of actual displacement (section 9.2).
            distance = torch.linalg.norm(pred.float() - target.float(), dim=-1)
            metrics = {
                "ade": float(distance.mean()),
                "fde": float(distance[:, -1].mean()),
                # Section 9.2: lateral error at 1.0 s, which is the second of the five horizons
                # {0.5, 1.0, 1.5, 2.0, 2.5}.
                "lateral_1s": float((pred[:, 1, 1] - target[:, 1, 1]).abs().mean()),
            }
        return loss, metrics
