"""Export, benchmark parsing, and parity thresholds (CLAUDE.md section 11).

The trtexec fixture below is verbatim output captured from this device. Parsing it correctly
is not obvious: trtexec prints five timing sections that all carry percentiles, plus a "Total
GPU Compute Time" line that shares a label with a real section, and both facts have already
produced wrong numbers once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drivyx.export.parity import CTRL_ADE_TOLERANCE, SEG_MIOU_TOLERANCE, ParityResult
from drivyx.export.trt_build import FRAME_BUDGET_MS, budget_status, parse_bench

# Captured from trtexec on the Orin, 2026-07-20.
TRTEXEC_OUTPUT = """\
[07/20/2026-12:17:47] [I] === Performance summary ===
[07/20/2026-12:17:47] [I] Throughput: 258.892 qps
[07/20/2026-12:17:47] [I] Latency: min = 1.87549 ms, max = 6.78198 ms, mean = 4.0024 ms, median = 4.32526 ms, percentile(90%) = 4.54346 ms, percentile(95%) = 4.98193 ms, percentile(99%) = 6.56958 ms
[07/20/2026-12:17:47] [I] Enqueue Time: min = 0.685059 ms, max = 3.75745 ms, mean = 0.986205 ms, median = 0.947449 ms, percentile(90%) = 1.14355 ms, percentile(95%) = 1.25952 ms, percentile(99%) = 2.01233 ms
[07/20/2026-12:17:47] [I] H2D Latency: min = 0.120728 ms, max = 0.372559 ms, mean = 0.160828 ms, median = 0.161133 ms, percentile(90%) = 0.169922 ms, percentile(95%) = 0.173584 ms, percentile(99%) = 0.185791 ms
[07/20/2026-12:17:47] [I] GPU Compute Time: min = 1.71802 ms, max = 6.48462 ms, mean = 3.82667 ms, median = 4.15286 ms, percentile(90%) = 4.36743 ms, percentile(95%) = 4.79419 ms, percentile(99%) = 6.39062 ms
[07/20/2026-12:17:47] [I] D2H Latency: min = 0.00592041 ms, max = 0.0317383 ms, mean = 0.0149051 ms, median = 0.0136719 ms, percentile(90%) = 0.0196533 ms, percentile(95%) = 0.0219727 ms, percentile(99%) = 0.0251465 ms
[07/20/2026-12:17:47] [I] Total Host Walltime: 3.01285 s
[07/20/2026-12:17:47] [I] Total GPU Compute Time: 2.9848 s
[07/20/2026-12:17:47] [W] * GPU compute time is unstable, with coefficient of variance = 30.4707%.
"""

ENGINE = Path("/tmp/example.engine")


def test_parses_the_end_to_end_latency_section() -> None:
    result = parse_bench(TRTEXEC_OUTPUT, ENGINE)

    assert result.min_ms == pytest.approx(1.87549)
    assert result.max_ms == pytest.approx(6.78198)
    assert result.mean_ms == pytest.approx(4.0024)
    assert result.p50 == pytest.approx(4.32526)


def test_percentiles_come_from_the_latency_section_not_d2h() -> None:
    """The first bug this fixture pins.

    Every section carries percentiles, and a whole-output scan returns the last one, which is
    D2H Latency: a device copy roughly 200x faster than end-to-end latency. That produced a
    report where p99 (0.025 ms) was smaller than p50 (4.5 ms).
    """
    result = parse_bench(TRTEXEC_OUTPUT, ENGINE)

    assert result.p95 == pytest.approx(4.98193)
    assert result.p99 == pytest.approx(6.56958)
    # The D2H values must not appear anywhere.
    assert result.p99 != pytest.approx(0.0251465)


def test_percentiles_are_monotonic() -> None:
    """Percentiles cannot decrease. A violation means values came from two sections."""
    result = parse_bench(TRTEXEC_OUTPUT, ENGINE)

    assert result.p50 <= result.p95 <= result.p99
    ordered = [result.percentiles[k] for k in sorted(result.percentiles, key=int)]
    assert ordered == sorted(ordered)


def test_total_gpu_compute_time_does_not_overwrite_the_real_section() -> None:
    """The second bug this fixture pins.

    "Total GPU Compute Time: 2.9848 s" matches the same label as the real "GPU Compute Time"
    section but carries a single summed value and no statistics. Without a guard it overwrites
    the real section and the reported compute time becomes zero.
    """
    result = parse_bench(TRTEXEC_OUTPUT, ENGINE)

    assert result.gpu_compute_ms == pytest.approx(4.15286)
    assert result.gpu_compute_ms > 0.0


def test_transfer_overhead_is_visible() -> None:
    """End-to-end latency exceeds pure compute by the host and device copies."""
    result = parse_bench(TRTEXEC_OUTPUT, ENGINE)

    assert result.p50 > result.gpu_compute_ms
    assert result.p50 - result.gpu_compute_ms < 1.0


def test_throughput_and_variance_are_captured() -> None:
    result = parse_bench(TRTEXEC_OUTPUT, ENGINE)

    assert result.throughput_qps == pytest.approx(258.892)
    # A high coefficient of variance means the p99 is not comparable across runs, so it is
    # reported rather than dropped.
    assert result.compute_variance_pct == pytest.approx(30.4707)


def test_non_monotonic_output_is_rejected() -> None:
    """A future trtexec format change must fail loudly, not report impossible numbers."""
    broken = (
        "[I] Latency: min = 1.0 ms, max = 9.0 ms, mean = 5.0 ms, median = 5.0 ms, "
        "percentile(99%) = 0.02 ms\n"
    )
    with pytest.raises(ValueError, match="not monotonic"):
        parse_bench(broken, ENGINE)


def test_bench_result_serialises() -> None:
    payload = parse_bench(TRTEXEC_OUTPUT, ENGINE).to_dict()

    for key in ("p50_ms", "p95_ms", "p99_ms", "gpu_compute_ms", "throughput_qps"):
        assert key in payload


# --- the frame budget (section 11) ------------------------------------------------------


def test_frame_budget_is_one_frame_at_30fps() -> None:
    assert pytest.approx(33.0) == FRAME_BUDGET_MS


@pytest.mark.parametrize(
    ("total_ms", "expected"),
    [
        (4.3, "ok"),
        (26.0, "ok"),
        (26.5, "warn"),
        (33.0, "warn"),
        (33.1, "err"),
        (50.0, "err"),
    ],
)
def test_budget_status(total_ms: float, expected: str) -> None:
    """Amber starts below the budget, not at it.

    A pipeline at 95% of budget has no headroom for capture, preprocessing, or control output,
    so it should not read as comfortably green.
    """
    assert budget_status(total_ms) == expected


# --- parity thresholds (section 11) -----------------------------------------------------


def test_parity_tolerances_match_the_spec() -> None:
    assert pytest.approx(1.0) == SEG_MIOU_TOLERANCE
    assert pytest.approx(0.05) == CTRL_ADE_TOLERANCE


def test_parity_passes_within_tolerance() -> None:
    result = ParityResult(
        model="seg",
        precision="fp16",
        samples=200,
        torch_metric=46.019,
        engine_metric=45.951,
        delta=0.068,
        tolerance=SEG_MIOU_TOLERANCE,
        metric_name="mIoU (percentage points)",
        passed=SEG_MIOU_TOLERANCE >= 0.068,
    )
    assert result.passed
    assert result.to_dict()["delta"] == pytest.approx(0.068)


def test_parity_fails_outside_tolerance() -> None:
    """Section 11 makes this the only pass/fail gate in the project."""
    delta = 2.5
    result = ParityResult(
        model="seg",
        precision="int8",
        samples=200,
        torch_metric=46.0,
        engine_metric=43.5,
        delta=delta,
        tolerance=SEG_MIOU_TOLERANCE,
        metric_name="mIoU (percentage points)",
        passed=delta <= SEG_MIOU_TOLERANCE,
    )
    assert not result.passed


# --- ONNX export shapes (section 11) ----------------------------------------------------


def test_export_shapes_match_the_spec() -> None:
    """Section 11: 1x3x384x768 for seg, 1x8x48x96 plus 1x1 for ctrl."""
    from drivyx.export.onnx_export import (
        CTRL_LOGITS_INPUT,
        CTRL_SPEED_INPUT,
        OPSET,
        SEG_INPUT,
    )

    assert SEG_INPUT == (1, 3, 384, 768)
    assert CTRL_LOGITS_INPUT == (1, 8, 48, 96)
    assert CTRL_SPEED_INPUT == (1, 1)
    assert OPSET == 17


@pytest.mark.device
def test_real_engine_benchmarks() -> None:
    """Benchmark whatever engines exist on this device."""
    from drivyx.export.trt_build import benchmark_engine
    from drivyx.paths import get_paths

    engines = sorted(get_paths().export.glob("*.engine"))
    if not engines:
        pytest.skip("no engines built yet; run 'drivyx export' first")

    result = benchmark_engine(engines[0], iterations=30)

    assert result.p50 > 0
    assert result.p50 <= result.p95 <= result.p99
    assert result.gpu_compute_ms > 0
