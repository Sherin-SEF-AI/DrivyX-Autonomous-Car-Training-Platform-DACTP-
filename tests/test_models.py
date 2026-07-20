"""PIDNet-S and the losses (CLAUDE.md sections 9.1, 13).

Section 13 requires "seg forward/backward one step bf16 (device)". The CPU tests here cover
the parts that do not need a GPU: shapes, the loss maths, and the boundary derivation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drivyx.data.lut import IGNORE_ID
from drivyx.models.losses import (
    BoundaryLoss,
    OhemCrossEntropy,
    PIDNetLoss,
    WaypointL1,
    boundary_target_from_mask,
    compute_class_weights,
)
from drivyx.models.pidnet import PIDNet, build_pidnet

# --- architecture -----------------------------------------------------------------------


def test_pidnet_builds_at_expected_size() -> None:
    """PIDNet-S is ~7.6 M parameters. A wildly different count means a wrong variant."""
    model = PIDNet(8)
    params = sum(p.numel() for p in model.parameters())
    assert 6_000_000 < params < 9_000_000, f"{params:,} parameters is not PIDNet-S"


def test_eval_output_is_one_eighth_resolution() -> None:
    """Section 9.2 consumes 8x96x48 from a 768x384 input, so the stride is a contract."""
    model = PIDNet(8).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 384, 768))

    assert out.shape == (1, 8, 48, 96)


def test_train_mode_returns_three_heads() -> None:
    """Training needs the aux and boundary heads; inference must not pay for them.

    Batch 2, not 1: the PAPPM's global-pool branch produces a 1x1 map, and BatchNorm in
    training mode cannot normalise a single value per channel. Section 9.1 trains at batch
    16, so this is a constraint rather than a defect (see test_batch_one_training_is_rejected).
    """
    model = PIDNet(8).train()
    aux, main, boundary = model(torch.randn(2, 3, 128, 256))

    assert aux.shape == (2, 8, 16, 32)
    assert main.shape == (2, 8, 16, 32)
    assert boundary.shape == (2, 1, 16, 32), "the D head predicts one channel"


def test_batch_one_training_is_rejected_clearly() -> None:
    """Training at batch 1 fails, and it should fail loudly rather than silently misbehave.

    The PAPPM pools globally to 1x1; BatchNorm then sees one value per channel and raises.
    Pinned as a known constraint so nobody spends an afternoon on it: use batch >= 2 (section
    9.1 uses 16). Inference at batch 1 is unaffected, which is what export needs.
    """
    model = PIDNet(8).train()
    with pytest.raises(ValueError, match="more than 1 value per channel"):
        model(torch.randn(1, 3, 128, 256))


def test_batch_one_inference_works() -> None:
    """Section 11 exports at static batch 1, so eval mode must handle it."""
    model = PIDNet(8).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 128, 256)).shape == (1, 8, 16, 32)


def test_eval_mode_returns_a_single_tensor() -> None:
    """ONNX export (section 11) requires a single output."""
    model = PIDNet(8).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 128, 256))
    assert isinstance(out, torch.Tensor)


def test_num_classes_is_respected() -> None:
    model = PIDNet(4).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 128, 256)).shape[1] == 4


def test_backward_runs_on_cpu() -> None:
    """A CPU backward proves the graph has no in-place violation.

    This is the test that would have caught the double in-place ReLU: a stage ending with an
    in-place activation whose output the stem then rectifies again fails here with "a variable
    needed for gradient computation has been modified by an inplace operation".
    """
    model = PIDNet(8).train()
    aux, main, boundary = model(torch.randn(2, 3, 128, 256))
    (aux.mean() + main.mean() + boundary.mean()).backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_build_pidnet_without_pretrained() -> None:
    model, report = build_pidnet(8)
    assert isinstance(model, PIDNet)
    assert report is None


def test_load_pretrained_rejects_a_missing_file(tmp_path) -> None:
    from drivyx.models.pidnet import load_pretrained

    with pytest.raises(FileNotFoundError, match="not found"):
        load_pretrained(PIDNet(8), tmp_path / "absent.pth")


def test_load_pretrained_rejects_an_unrelated_checkpoint(tmp_path) -> None:
    """A file that loads but covers almost nothing must abort, not train from scratch.

    This is the D024 guarantee: silently tolerating missing keys is how a randomly
    initialised backbone gets trained while the log says "loaded pretrained".
    """
    from drivyx.models.pidnet import load_pretrained

    path = tmp_path / "wrong.pth"
    torch.save({"stem.0.conv.weight": torch.zeros(32, 3, 3, 3)}, path)

    with pytest.raises(ValueError, match="covers only"):
        load_pretrained(PIDNet(8), path)


# --- OHEM (section 9.1) -----------------------------------------------------------------


def test_ohem_ignores_the_ignore_label() -> None:
    """A target of all-ignore must produce a finite zero, not a NaN or an index error.

    255 is not a valid gather index into 8 channels; the naive clamp(min=0) does not fix it
    because 255 is already positive.
    """
    loss_fn = OhemCrossEntropy(min_kept=10)
    logits = torch.randn(1, 8, 16, 16)
    target = torch.full((1, 16, 16), IGNORE_ID, dtype=torch.long)

    loss = loss_fn(logits, target)

    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_ohem_handles_a_mixed_target() -> None:
    loss_fn = OhemCrossEntropy(min_kept=10)
    target = torch.randint(0, 8, (2, 16, 16))
    target[:, :4, :] = IGNORE_ID

    loss = loss_fn(torch.randn(2, 8, 16, 16), target)

    assert torch.isfinite(loss) and float(loss) > 0


def test_ohem_focuses_on_hard_pixels() -> None:
    """OHEM's whole purpose: a few wrong pixels among many correct ones must dominate.

    Plain cross entropy averages the mistake away; OHEM keeps it.
    """
    logits = torch.full((1, 8, 32, 32), -10.0)
    logits[:, 0] = 10.0  # confidently class 0 everywhere
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, :2, :2] = 3  # a small patch is actually class 3

    ohem = OhemCrossEntropy(thresh=0.9, min_kept=4)(logits, target)
    plain = torch.nn.functional.cross_entropy(logits, target)

    assert float(ohem) > float(plain) * 5, "OHEM did not concentrate on the hard pixels"


def test_ohem_min_kept_floors_the_selection() -> None:
    """When almost nothing is hard, min_kept keeps the gradient from going noisy."""
    logits = torch.full((1, 8, 32, 32), -10.0)
    logits[:, 0] = 10.0
    target = torch.zeros(1, 32, 32, dtype=torch.long)

    loss = OhemCrossEntropy(thresh=0.9, min_kept=100)(logits, target)

    assert torch.isfinite(loss)


def test_ohem_upsamples_logits_to_the_target() -> None:
    """The head is 1/8 resolution; the loss is defined against the full-size mask."""
    loss = OhemCrossEntropy(min_kept=10)(torch.randn(1, 8, 8, 8), torch.randint(0, 8, (1, 64, 64)))
    assert torch.isfinite(loss)


# --- boundary ---------------------------------------------------------------------------


def test_boundary_target_marks_class_changes() -> None:
    mask = torch.zeros(1, 8, 8, dtype=torch.long)
    mask[:, :, 4:] = 1

    boundary = boundary_target_from_mask(mask)

    assert boundary.shape == (1, 1, 8, 8)
    # The two columns either side of the change are edges.
    assert float(boundary[0, 0, 0, 3]) == 1.0
    assert float(boundary[0, 0, 0, 4]) == 1.0
    assert float(boundary[0, 0, 0, 0]) == 0.0


def test_boundary_target_is_empty_for_a_uniform_mask() -> None:
    assert float(boundary_target_from_mask(torch.zeros(1, 8, 8, dtype=torch.long)).sum()) == 0.0


def test_boundary_ignores_the_ignore_region_edge() -> None:
    """The edge of an unlabelled region is an annotation artifact, not an object edge.

    Training the D branch on it would teach it to fire on the ego-vehicle mask.
    """
    mask = torch.zeros(1, 8, 8, dtype=torch.long)
    mask[:, :, 4:] = IGNORE_ID

    boundary = boundary_target_from_mask(mask)

    assert float(boundary.sum()) == 0.0


def test_boundary_loss_reweights_the_positive_class() -> None:
    """Edges are a few percent of pixels; without reweighting, predicting 'no edge' wins."""
    logits = torch.zeros(1, 1, 16, 16)
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, :1, :1] = 1.0  # ~0.4% positive

    loss = BoundaryLoss()(logits, target)

    assert torch.isfinite(loss) and float(loss) > 0


def test_boundary_loss_handles_no_edges() -> None:
    loss = BoundaryLoss()(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
    assert torch.isfinite(loss)


# --- class weights (section 9.1) --------------------------------------------------------


def test_class_weights_match_the_spec_formula() -> None:
    """w = 1/log(1.02 + freq), capped at 10x the minimum."""
    weights = compute_class_weights([1000] * 8)

    expected = 1.0 / np.log(1.02 + 0.125)
    assert torch.allclose(weights, torch.full((8,), float(expected)), atol=1e-5)


def test_class_weights_are_capped() -> None:
    weights = compute_class_weights([10_000_000, 1, 1, 1, 1, 1, 1, 1], cap=10.0)
    assert float(weights.max()) <= float(weights.min()) * 10.0 + 1e-4


def test_class_weights_reject_an_empty_histogram() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_class_weights([0] * 8)


def test_class_weights_agree_with_the_shard_index_formula() -> None:
    """shards.py and losses.py implement the same formula in two places (by design: shards
    must not import torch). They must not drift apart."""
    from drivyx.data.shards import class_weights as numpy_weights

    counts = [500_000, 100_000, 60_000, 20_000, 15_000, 120_000, 300_000, 700_000]
    torch_weights = compute_class_weights(counts)

    assert np.allclose(numpy_weights(counts), torch_weights.numpy(), atol=1e-5)


# --- the combined objective -------------------------------------------------------------


def test_pidnet_loss_combines_every_term() -> None:
    loss_fn = PIDNetLoss(class_weights=compute_class_weights([1000] * 8))
    outputs = (torch.randn(1, 8, 16, 16), torch.randn(1, 8, 16, 16), torch.randn(1, 1, 16, 16))
    target = torch.randint(0, 8, (1, 16, 16))
    boundary = boundary_target_from_mask(target)

    total, parts = loss_fn(outputs, target, boundary)

    assert torch.isfinite(total)
    assert set(parts) == {"loss/main", "loss/aux", "loss/boundary", "loss/boundary_aware"}
    assert all(np.isfinite(v) for v in parts.values())


def test_pidnet_loss_weights_follow_the_config() -> None:
    outputs = (torch.zeros(1, 8, 8, 8), torch.zeros(1, 8, 8, 8), torch.zeros(1, 1, 8, 8))
    target = torch.randint(0, 8, (1, 8, 8))
    boundary = boundary_target_from_mask(target)

    low, _ = PIDNetLoss(boundary_weight=0.0)(outputs, target, boundary)
    high, _ = PIDNetLoss(boundary_weight=50.0)(outputs, target, boundary)

    assert float(high) > float(low)


# --- waypoint loss (section 9.2) --------------------------------------------------------


def test_waypoint_l1_metrics() -> None:
    """ADE/FDE are Euclidean displacement in metres, not the L1 the loss optimises."""
    pred = torch.zeros(4, 5, 2)
    target = torch.zeros(4, 5, 2)
    target[:, :, 0] = 3.0
    target[:, :, 1] = 4.0  # every waypoint is 5 m away

    loss, metrics = WaypointL1()(pred, target)

    assert float(loss) == pytest.approx(3.5)  # L1 mean of |3| and |4|
    assert metrics["ade"] == pytest.approx(5.0)
    assert metrics["fde"] == pytest.approx(5.0)
    assert metrics["lateral_1s"] == pytest.approx(4.0)


def test_waypoint_fde_is_the_last_horizon() -> None:
    pred = torch.zeros(1, 5, 2)
    target = torch.zeros(1, 5, 2)
    target[0, -1, 0] = 10.0

    _loss, metrics = WaypointL1()(pred, target)

    assert metrics["fde"] == pytest.approx(10.0)
    assert metrics["ade"] == pytest.approx(2.0)


# --- device (section 13: "seg forward/backward one step bf16") ---------------------------


@pytest.mark.device
def test_seg_forward_backward_one_step_bf16() -> None:
    """Section 13's named device test, in section 3's exact training configuration."""
    from drivyx.torch_setup import configure

    configure()
    device = torch.device("cuda")

    model = PIDNet(8).to(device, memory_format=torch.channels_last).train()
    criterion = PIDNetLoss(class_weights=compute_class_weights([1000] * 8)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    images = torch.randn(2, 3, 384, 768, device=device).contiguous(
        memory_format=torch.channels_last
    )
    targets = torch.randint(0, 8, (2, 384, 768), device=device)
    targets[:, :16, :] = IGNORE_ID
    boundary = boundary_target_from_mask(targets)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(images)
    loss, parts = criterion(outputs, targets, boundary)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    assert torch.isfinite(loss), f"non-finite loss: {parts}"
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.device
def test_real_backbone_loads_completely() -> None:
    """The D024 claim, asserted: the checkpoint must cover the backbone fully."""
    from drivyx.models.pidnet import load_pretrained
    from drivyx.paths import get_paths

    paths = get_paths()
    candidates = sorted(paths.pretrained.glob("*.pth")) + sorted(paths.pretrained.glob("*.pth.tar"))
    if not candidates:
        pytest.skip("no backbone in pretrained/")

    report = load_pretrained(PIDNet(8), candidates[0])

    assert report.fraction == 1.0, f"backbone only {100 * report.fraction:.1f}% loaded"
    assert not report.skipped_shape_mismatch
    assert not report.unexpected_in_checkpoint, "the model does not match the checkpoint"
