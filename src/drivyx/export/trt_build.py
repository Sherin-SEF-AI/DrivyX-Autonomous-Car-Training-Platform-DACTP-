"""TensorRT engine building and benchmarking (CLAUDE.md section 11).

"Build with trtexec: fp16 flags, or int8 with an entropy calibrator over 512 val images
(calibration cache stored under export/)."

The master prompt is explicit about the split: shell out to trtexec and parse its stdout
rather than reimplementing engine building against the python API, except for the INT8
entropy calibrator, which uses the tensorrt python bindings.

That split is not arbitrary. trtexec already encodes NVIDIA's tested build and timing
behaviour, and reimplementing it would mean owning bugs that are not ours. Calibration is the
one thing trtexec cannot do from the command line without a cache that does not exist yet.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Section 11: "int8 with an entropy calibrator over 512 val images".
CALIBRATION_IMAGES = 512

#: Section 11: the GUI shows a green/amber indicator against a 33 ms frame budget for seg and
#: ctrl combined. 33 ms is one frame at 30 fps.
FRAME_BUDGET_MS = 33.0

# trtexec prints several timing sections, each on its own line and each carrying its own
# percentiles:
#
#   Latency:          min = 1.87 ms, ..., median = 4.32 ms, percentile(99%) = 6.56 ms
#   Enqueue Time:     min = 0.68 ms, ...
#   H2D Latency:      min = 0.12 ms, ...
#   GPU Compute Time: min = 1.71 ms, ...
#   D2H Latency:      min = 0.005 ms, ..., percentile(99%) = 0.025 ms
#
# Scanning the whole output for "percentile(NN%)" therefore returns the LAST section's values,
# which is D2H Latency: a device-to-host copy roughly 200x faster than the end-to-end latency.
# That produced a report where p99 was smaller than p50, which is impossible. Percentiles are
# now read from the same line as their section label.
_RE_SECTION = re.compile(
    r"^.*?\b(?P<name>Latency|Enqueue Time|H2D Latency|GPU Compute Time"
    r"|D2H Latency):\s*(?P<body>.*)$",
    re.MULTILINE,
)
_RE_STAT = re.compile(r"\b(min|max|mean|median)\s*=\s*([\d.eE+-]+)\s*ms", re.IGNORECASE)
_RE_PERCENTILE = re.compile(
    r"percentile\((\d+(?:\.\d+)?)%\)\s*=\s*([\d.eE+-]+)\s*ms", re.IGNORECASE
)
_RE_THROUGHPUT = re.compile(r"Throughput:\s*([\d.]+)\s*qps", re.IGNORECASE)
#: trtexec warns when timing is unstable. Worth carrying through: a high variance
#: means the p99 is not a number to compare across runs.
_RE_VARIANCE = re.compile(r"coefficient of variance\s*=\s*([\d.]+)%", re.IGNORECASE)


@dataclass
class BuildResult:
    """One engine build."""

    model: str
    precision: str
    engine: Path
    onnx: Path
    seconds: float
    log: str
    calibration_cache: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "precision": self.precision,
            "engine": str(self.engine),
            "onnx": str(self.onnx),
            "build_seconds": round(self.seconds, 1),
            "calibration_cache": str(self.calibration_cache) if self.calibration_cache else None,
        }


@dataclass
class BenchResult:
    """Parsed trtexec latency percentiles (section 11)."""

    engine: Path
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    percentiles: dict[str, float] = field(default_factory=dict)
    throughput_qps: float = 0.0
    #: Median pure compute time, excluding host/device transfers.
    gpu_compute_ms: float = 0.0
    #: trtexec's coefficient of variance for compute time, in percent.
    compute_variance_pct: float = 0.0

    @property
    def p50(self) -> float:
        return self.median_ms

    @property
    def p95(self) -> float:
        return self.percentiles.get("95", self.percentiles.get("99", self.max_ms))

    @property
    def p99(self) -> float:
        return self.percentiles.get("99", self.max_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": str(self.engine),
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
            "percentiles": self.percentiles,
            "gpu_compute_ms": self.gpu_compute_ms,
            "compute_variance_pct": self.compute_variance_pct,
            "throughput_qps": self.throughput_qps,
        }


def _run_trtexec(args: list[str], *, timeout: int = 3600) -> tuple[str, int]:
    """Run trtexec and return its combined output and exit code."""
    from drivyx.env_report import require_trtexec

    command = [str(require_trtexec()), *args]
    logger.info("trtexec %s", " ".join(args[:4]) + (" ..." if len(args) > 4 else ""))
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    return result.stdout + result.stderr, result.returncode


def build_engine(
    onnx: Path,
    engine: Path,
    *,
    model: str,
    precision: str = "fp16",
    calibration_cache: Path | None = None,
    workspace_mb: int = 4096,
) -> BuildResult:
    """Build a TensorRT engine with trtexec (section 11)."""
    import time

    if precision not in ("fp16", "int8"):
        raise ValueError(f"precision must be fp16 or int8, got {precision!r}")
    if not onnx.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx}")

    engine.parent.mkdir(parents=True, exist_ok=True)
    args = [
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        f"--memPoolSize=workspace:{workspace_mb}",
        "--skipInference",
    ]
    if precision == "fp16":
        args.append("--fp16")
    else:
        # INT8 still enables fp16: layers the calibrator cannot quantise safely fall back to
        # fp16 rather than fp32, which is what keeps a mixed-precision engine fast.
        args += ["--int8", "--fp16"]
        if calibration_cache is not None:
            args.append(f"--calib={calibration_cache}")

    started = time.monotonic()
    output, code = _run_trtexec(args)
    elapsed = time.monotonic() - started

    if code != 0 or not engine.is_file():
        tail = "\n".join(output.strip().splitlines()[-15:])
        raise RuntimeError(
            f"trtexec failed to build {model} at {precision} (exit {code}).\n"
            f"Last lines of output:\n{tail}"
        )

    logger.info(
        "built %s %s engine in %.1f s -> %s (%.1f MB)",
        model,
        precision,
        elapsed,
        engine.name,
        engine.stat().st_size / 1e6,
    )
    return BuildResult(
        model=model,
        precision=precision,
        engine=engine,
        onnx=onnx,
        seconds=elapsed,
        log=output,
        calibration_cache=calibration_cache,
    )


def parse_bench(output: str, engine: Path) -> BenchResult:
    """Parse trtexec's latency report (section 11).

    Reads each timing section separately and reports the end-to-end "Latency" section, which
    is what a frame budget is about: it includes the host-to-device copy, the compute, and the
    device-to-host copy. "GPU Compute Time" is captured alongside because the gap between the
    two is the transfer overhead, which is worth seeing when a model is close to budget.
    """
    result = BenchResult(engine=engine)
    sections: dict[str, dict[str, Any]] = {}

    for match in _RE_SECTION.finditer(output):
        name = match.group("name")
        body = match.group("body")
        stats = {key.lower(): float(value) for key, value in _RE_STAT.findall(body)}
        # A section is only real if it carries statistics. trtexec also prints
        # "Total GPU Compute Time: 3.00249 s", which matches the same label but holds a single
        # summed value; without this guard it overwrites the real GPU Compute Time section
        # with an empty one and the reported compute time becomes zero.
        if not stats:
            continue
        # trtexec reports "percentile(99%)"; normalise 99.0 and 99 to the same key.
        stats["percentiles"] = {
            str(int(float(p))): float(v) for p, v in _RE_PERCENTILE.findall(body)
        }
        sections[name] = stats

    latency = sections.get("Latency")
    if latency:
        result.min_ms = latency.get("min", 0.0)
        result.max_ms = latency.get("max", 0.0)
        result.mean_ms = latency.get("mean", 0.0)
        result.median_ms = latency.get("median", 0.0)
        result.percentiles = latency.get("percentiles", {})

    compute = sections.get("GPU Compute Time")
    if compute:
        result.gpu_compute_ms = compute.get("median", 0.0)

    throughput = _RE_THROUGHPUT.search(output)
    if throughput:
        result.throughput_qps = float(throughput.group(1))

    variance = _RE_VARIANCE.search(output)
    if variance:
        result.compute_variance_pct = float(variance.group(1))

    # Percentiles are monotonically non-decreasing by definition. A violation means the parse
    # picked values from more than one section, which is exactly the bug this rewrite fixes,
    # so it is worth catching rather than reporting numbers that cannot be true.
    ordered = [result.percentiles[k] for k in sorted(result.percentiles, key=int)]
    if ordered and (result.median_ms > ordered[0] or ordered != sorted(ordered)):
        raise ValueError(
            f"parsed percentiles are not monotonic (median {result.median_ms}, "
            f"percentiles {result.percentiles}). trtexec's output format may have changed."
        )

    return result


def benchmark_engine(engine: Path, *, iterations: int = 200, warmup_ms: int = 500) -> BenchResult:
    """Time an engine with trtexec and parse the percentiles (section 11)."""
    if not engine.is_file():
        raise FileNotFoundError(f"engine not found: {engine}")

    output, code = _run_trtexec(
        [
            f"--loadEngine={engine}",
            f"--iterations={iterations}",
            f"--warmUp={warmup_ms}",
            "--percentile=90,95,99",
            "--avgRuns=10",
        ]
    )
    if code != 0:
        tail = "\n".join(output.strip().splitlines()[-15:])
        raise RuntimeError(f"trtexec benchmark failed (exit {code}).\n{tail}")

    result = parse_bench(output, engine)
    if result.median_ms == 0.0:
        raise RuntimeError(
            "trtexec produced no parseable latency report. Its output format may have changed; "
            f"first lines:\n{output[:400]}"
        )

    logger.info(
        "%s: p50 %.2f ms, p95 %.2f ms, p99 %.2f ms, %.0f qps",
        engine.name,
        result.p50,
        result.p95,
        result.p99,
        result.throughput_qps,
    )
    return result


def budget_status(total_ms: float, budget_ms: float = FRAME_BUDGET_MS) -> str:
    """Green, amber, or over, against section 11's 33 ms frame budget.

    Amber below the budget rather than only at it: a pipeline at 95% of budget has no headroom
    for the rest of a real system (capture, preprocessing, control output), so it should not
    read as comfortably green.
    """
    if total_ms <= budget_ms * 0.8:
        return "ok"
    if total_ms <= budget_ms:
        return "warn"
    return "err"
