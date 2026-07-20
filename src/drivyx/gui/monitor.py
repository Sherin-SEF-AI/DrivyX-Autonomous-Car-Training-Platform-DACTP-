"""tegrastats monitor thread (CLAUDE.md section 12.5).

Spawns `tegrastats --interval 1000`, parses each line with compiled regexes into a
dataclass, and publishes it via a Qt signal. Degrades to jetson-stats' python API when
tegrastats is unavailable; when both fail the SYSTEM workspace shows ERR and everything else
keeps working (section 12.5).

The parsers live at module scope and take a string, so they are unit-testable on a CPU-only
machine with no Qt event loop and no Jetson.

Reference line from this device (JetPack 7.2), which the regexes are written against:

    07-17-2026 15:15:24 RAM 8182/62817MB (lfb 6x4MB) CPU [18%@729,12%@729,...] GR3D_FREQ 5%
    cpu@51.343C/51.343C soc2@48.843C/48.843C soc0@49.25C/49.25C gpu@46.843C/46.843C
    tj@51.343C/51.343C soc1@47.843C/47.843C VDD_GPU_SOC 3588mW/3588mW/3588mW
    VDD_CPU_CV 797mW/797mW/797mW VIN_SYS_5V0 5148mW/5148mW/5148mW

Rail names differ across JetPack releases (JetPack 6 used POM_5V_IN), so power rails are
matched generically rather than by name.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

TEGRASTATS_BIN = "tegrastats"
DEFAULT_INTERVAL_MS = 1000

#: `RAM 8182/62817MB`
_RE_RAM = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")
#: `GR3D_FREQ 5%`. GR3D is the GPU engine; its FREQ percentage is GPU utilisation.
_RE_GPU = re.compile(r"\bGR3D_FREQ\s+(\d+)%")
#: `CPU [18%@729,12%@729,...]`
_RE_CPU_BLOCK = re.compile(r"\bCPU\s+\[([^\]]*)\]")
_RE_CPU_CORE = re.compile(r"(\d+)%@(\d+)")
#: `gpu@46.843C`, `soc0@49.25C`, `tj@51.343C`. The trailing C is what distinguishes these
#: from the CPU block's `18%@729`, which has no unit suffix.
_RE_TEMP = re.compile(r"\b([a-zA-Z][a-zA-Z0-9_]*)@(-?[\d.]+)C")
#: `VDD_GPU_SOC 3588mW/3588mW/3588mW` -> name, instant, average, peak.
_RE_POWER = re.compile(r"\b([A-Z][A-Z0-9_]+)\s+(\d+)mW/(\d+)mW/(\d+)mW")


@dataclass(frozen=True)
class TegraSample:
    """One parsed tegrastats line.

    Fields are None when the line did not carry them, so a format change degrades one
    readout rather than the whole status bar.
    """

    gpu_pct: int | None = None
    ram_used_mb: int | None = None
    ram_total_mb: int | None = None
    cpu_pct: list[int] = field(default_factory=list)
    temps_c: dict[str, float] = field(default_factory=dict)
    power_mw: dict[str, int] = field(default_factory=dict)
    raw: str = ""

    @property
    def ram_used_gb(self) -> float | None:
        return self.ram_used_mb / 1024 if self.ram_used_mb is not None else None

    @property
    def ram_total_gb(self) -> float | None:
        return self.ram_total_mb / 1024 if self.ram_total_mb is not None else None

    @property
    def cpu_avg_pct(self) -> float | None:
        return sum(self.cpu_pct) / len(self.cpu_pct) if self.cpu_pct else None

    @property
    def soc_temp_c(self) -> float | None:
        """Hottest SoC sensor, for the section 12.3 `SOC 71C` readout.

        Orin exposes soc0/soc1/soc2 plus a tj (junction) aggregate. The maximum is the
        number that matters thermally; falling back to tj keeps the readout alive if the
        soc* sensors are renamed.
        """
        soc = [v for k, v in self.temps_c.items() if k.startswith("soc")]
        if soc:
            return max(soc)
        return self.temps_c.get("tj")

    @property
    def gpu_temp_c(self) -> float | None:
        return self.temps_c.get("gpu")

    @property
    def total_power_w(self) -> float | None:
        """Sum of every instantaneous power rail, in watts.

        Summing all rails rather than reading one named rail keeps this correct across
        JetPack releases, which rename them.
        """
        if not self.power_mw:
            return None
        return sum(self.power_mw.values()) / 1000.0


def parse_tegrastats(line: str) -> TegraSample:
    """Parse one tegrastats line. Never raises: an unrecognised line yields empty fields."""
    ram = _RE_RAM.search(line)
    gpu = _RE_GPU.search(line)

    cpu_pct: list[int] = []
    cpu_block = _RE_CPU_BLOCK.search(line)
    if cpu_block:
        cpu_pct = [int(m.group(1)) for m in _RE_CPU_CORE.finditer(cpu_block.group(1))]

    # Temperatures are matched outside the CPU block so a core frequency can never be read
    # as a sensor value.
    temp_region = line
    if cpu_block:
        temp_region = line[: cpu_block.start()] + line[cpu_block.end() :]
    temps = {m.group(1): float(m.group(2)) for m in _RE_TEMP.finditer(temp_region)}

    power = {m.group(1): int(m.group(2)) for m in _RE_POWER.finditer(line)}

    return TegraSample(
        gpu_pct=int(gpu.group(1)) if gpu else None,
        ram_used_mb=int(ram.group(1)) if ram else None,
        ram_total_mb=int(ram.group(2)) if ram else None,
        cpu_pct=cpu_pct,
        temps_c=temps,
        power_mw=power,
        raw=line.strip(),
    )


def format_status(sample: TegraSample) -> str:
    """Render the section 12.3 status bar readout: `GPU 87%  MEM 21.4/64G  SOC 71C  PWR 48W`.

    Monospace alignment matters here (section 12.1), so widths are fixed and a missing
    field renders as dashes rather than collapsing the layout.
    """
    gpu = f"{sample.gpu_pct:3d}%" if sample.gpu_pct is not None else "  --"
    if sample.ram_used_gb is not None and sample.ram_total_gb is not None:
        mem = f"{sample.ram_used_gb:4.1f}/{sample.ram_total_gb:.0f}G"
    else:
        mem = "   --"
    soc = f"{sample.soc_temp_c:3.0f}C" if sample.soc_temp_c is not None else " --"
    pwr = f"{sample.total_power_w:3.0f}W" if sample.total_power_w is not None else " --"
    return f"GPU {gpu}  MEM {mem}  SOC {soc}  PWR {pwr}"


class MonitorThread(QThread):
    """Runs tegrastats and emits a TegraSample per line.

    QThread rather than a timer polling a file: tegrastats is a long-lived process that
    pushes a line per interval, and reading its stdout must never touch the Qt main thread
    (section 3 and rule 26).
    """

    sample = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, interval_ms: int = DEFAULT_INTERVAL_MS, parent: object = None) -> None:
        super().__init__(parent)
        self._interval_ms = interval_ms
        self._proc: subprocess.Popen[str] | None = None
        self._stopping = False

    def run(self) -> None:
        if shutil.which(TEGRASTATS_BIN) is None:
            self._run_jtop_fallback("tegrastats not found on PATH")
            return
        try:
            self._run_tegrastats()
        except Exception as exc:
            if not self._stopping:
                logger.warning("tegrastats monitor failed: %s", exc)
                self._run_jtop_fallback(str(exc))

    def _run_tegrastats(self) -> None:
        self._proc = subprocess.Popen(
            [TEGRASTATS_BIN, "--interval", str(self._interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        logger.info("tegrastats started (pid %s)", self._proc.pid)
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stopping:
                break
            if line.strip():
                self.sample.emit(parse_tegrastats(line))

        code = self._proc.poll()
        if not self._stopping and code not in (0, None):
            raise RuntimeError(f"tegrastats exited with code {code}")

    def _run_jtop_fallback(self, reason: str) -> None:
        """Section 12.5: degrade to jetson-stats' python API, then to ERR state."""
        logger.info("falling back to jetson-stats (%s)", reason)
        try:
            from jtop import jtop
        except ImportError:
            self.failed.emit(
                f"tegrastats unavailable ({reason}) and jetson-stats is not installed. "
                "System telemetry is off; every other workspace keeps working."
            )
            return

        try:
            with jtop(interval=self._interval_ms / 1000.0) as jet:
                while jet.ok() and not self._stopping:
                    self.sample.emit(_sample_from_jtop(jet))
        except Exception as exc:
            if not self._stopping:
                self.failed.emit(
                    f"tegrastats unavailable ({reason}); jetson-stats also failed: {exc}"
                )

    def stop(self) -> None:
        """Terminate tegrastats and wait for the thread to unwind."""
        self._stopping = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.wait(3000)


def _sample_from_jtop(jet: object) -> TegraSample:
    """Adapt a jtop reading to TegraSample.

    jtop's schema differs across releases, so every field is fetched defensively; a missing
    one degrades that readout only.
    """
    stats = getattr(jet, "stats", {}) or {}
    memory = getattr(jet, "memory", {}) or {}
    ram = memory.get("RAM", {}) if isinstance(memory, dict) else {}

    temps: dict[str, float] = {}
    for name, value in (getattr(jet, "temperature", {}) or {}).items():
        raw = value.get("temp") if isinstance(value, dict) else value
        if isinstance(raw, (int, float)):
            temps[str(name).lower()] = float(raw)

    power_mw: dict[str, int] = {}
    power = getattr(jet, "power", {}) or {}
    rails = power.get("rail", {}) if isinstance(power, dict) else {}
    if isinstance(rails, dict):
        for name, value in rails.items():
            raw = value.get("power") if isinstance(value, dict) else value
            if isinstance(raw, (int, float)):
                power_mw[str(name)] = int(raw)

    gpu = stats.get("GPU")
    return TegraSample(
        gpu_pct=int(gpu) if isinstance(gpu, (int, float)) else None,
        ram_used_mb=int(ram.get("used", 0) / 1024) if ram.get("used") else None,
        ram_total_mb=int(ram.get("tot", 0) / 1024) if ram.get("tot") else None,
        temps_c=temps,
        power_mw=power_mw,
        raw="jetson-stats",
    )
