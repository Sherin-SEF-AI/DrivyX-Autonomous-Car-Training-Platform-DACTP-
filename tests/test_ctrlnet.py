"""Waypoint predictor (CLAUDE.md sections 9.2, 13).

Section 13 requires "ctrl param count < 2 M (cpu)" by name.
"""

from __future__ import annotations

import pytest
import torch

from drivyx.models.ctrlnet import (
    CONV_CHANNELS,
    MAX_PARAMETERS,
    NUM_WAYPOINTS,
    CtrlNet,
    build_ctrlnet,
)


def test_parameter_count_is_under_two_million() -> None:
    """Section 13's named test. Section 9.2: "stay under 2 M"."""
    count = CtrlNet(8).parameter_count()
    assert count < MAX_PARAMETERS, f"{count:,} parameters exceeds the 2 M budget"


def test_budget_is_asserted_not_merely_printed() -> None:
    """A model over budget must fail to construct, not log a warning and continue.

    Over budget means the 33 ms frame budget in section 11 cannot be met, which the export
    gate would only discover much later.
    """
    import drivyx.models.ctrlnet as module

    original = module.MAX_PARAMETERS
    try:
        module.MAX_PARAMETERS = 1000
        with pytest.raises(ValueError, match="over section 9.2"):
            CtrlNet(8)
    finally:
        module.MAX_PARAMETERS = original


def test_architecture_matches_the_spec() -> None:
    """Section 9.2: 4 conv blocks at 32, 64, 96, 128 with GroupNorm(8) and SiLU."""
    model = CtrlNet(8)

    assert CONV_CHANNELS == (32, 64, 96, 128)
    assert len(model.features) == 4
    for block, channels in zip(model.features, CONV_CHANNELS):
        assert isinstance(block.norm, torch.nn.GroupNorm)
        assert block.norm.num_groups == 8
        assert block.norm.num_channels == channels
        assert isinstance(block.act, torch.nn.SiLU)
        assert block.conv.stride == (2, 2)


def test_output_shape_is_five_waypoints() -> None:
    """Section 9.2: 10 outputs = 5 waypoints x (x, y) in metres."""
    out = CtrlNet(8)(torch.randn(4, 8, 48, 96), torch.rand(4) * 10)

    assert out.shape == (4, NUM_WAYPOINTS, 2)


def test_accepts_speed_as_column_or_flat() -> None:
    model = CtrlNet(8)
    logits = torch.randn(2, 8, 48, 96)

    flat = model(logits, torch.tensor([3.0, 7.0]))
    column = model(logits, torch.tensor([[3.0], [7.0]]))

    assert torch.allclose(flat, column)


def test_speed_changes_the_prediction() -> None:
    """The speed input must actually reach the output, or the MLP is dead weight.

    Zero-init makes the untrained head output zeros, so the head's final layer is perturbed
    first; otherwise both predictions would be zero and the test would pass vacuously.
    """
    model = CtrlNet(8)
    torch.nn.init.normal_(model.head[-1].weight, std=0.1)
    logits = torch.randn(1, 8, 48, 96)

    slow = model(logits, torch.tensor([1.0]))
    fast = model(logits, torch.tensor([15.0]))

    assert not torch.allclose(slow, fast), "speed has no influence on the waypoints"


def test_untrained_model_predicts_stay_put() -> None:
    """Zero-initialised head (docs/DECISIONS.md D028): the first prediction is the origin."""
    out = CtrlNet(8)(torch.randn(2, 8, 48, 96), torch.tensor([5.0, 5.0]))
    assert torch.allclose(out, torch.zeros_like(out))


def test_backward_reaches_every_parameter() -> None:
    model = CtrlNet(8)
    out = model(torch.randn(2, 8, 48, 96), torch.tensor([4.0, 6.0]))
    out.abs().sum().backward()

    without_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not without_grad, f"no gradient reached {without_grad}"


def test_batch_one_inference() -> None:
    """Section 11 exports at static batch 1."""
    model = CtrlNet(8).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 8, 48, 96), torch.tensor([5.0])).shape == (1, 5, 2)


def test_groupnorm_is_batch_size_invariant() -> None:
    """GroupNorm, not BatchNorm: a batch-1 inference must match the batched result.

    This is why section 9.2 specifies GroupNorm. With BatchNorm the same frame would predict
    differently depending on what else was in its batch.
    """
    model = CtrlNet(8).eval()
    logits = torch.randn(4, 8, 48, 96)
    speed = torch.tensor([2.0, 5.0, 8.0, 11.0])
    torch.nn.init.normal_(model.head[-1].weight, std=0.1)

    with torch.no_grad():
        batched = model(logits, speed)
        single = torch.cat([model(logits[i : i + 1], speed[i : i + 1]) for i in range(4)])

    assert torch.allclose(batched, single, atol=1e-5)


def test_train_and_eval_modes_agree() -> None:
    """GroupNorm has no running statistics, so the two modes must be identical."""
    model = CtrlNet(8)
    torch.nn.init.normal_(model.head[-1].weight, std=0.1)
    logits = torch.randn(2, 8, 48, 96)
    speed = torch.tensor([5.0, 9.0])

    model.train()
    with torch.no_grad():
        train_out = model(logits, speed)
    model.eval()
    with torch.no_grad():
        eval_out = model(logits, speed)

    assert torch.allclose(train_out, eval_out)


def test_build_ctrlnet_reports_the_budget(caplog: pytest.LogCaptureFixture) -> None:
    """Section 9.2: "Parameter budget must print at startup"."""
    import logging

    with caplog.at_level(logging.INFO, logger="drivyx.models.ctrlnet"):
        build_ctrlnet(8)

    assert any("parameters" in record.message for record in caplog.records)


def test_accepts_other_class_counts() -> None:
    assert CtrlNet(4)(torch.randn(1, 4, 48, 96), torch.tensor([5.0])).shape == (1, 5, 2)
