"""SYSTEM workspace (CLAUDE.md section 12.4).

"environment report (versions, wheel provenance, nvpmodel, disk free on NVMe), tegrastats
live charts (GPU util, RAM, temps, power) over the last 10 minutes, buttons to copy
diagnostics to clipboard".

The environment report is gathered on a QThread: env_report imports torch and shells out to
dpkg and nvpmodel, which would stall the main thread (rule 26).
"""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from drivyx.gui.monitor import TegraSample
from drivyx.gui.theme import tokens
from drivyx.gui.widgets.panel import Panel
from drivyx.gui.widgets.statrow import StatRow
from drivyx.gui.workspaces.base import Workspace

logger = logging.getLogger(__name__)

#: Section 12.4: charts cover the last 10 minutes. tegrastats ticks at 1 Hz.
HISTORY_SECONDS = 600

#: How long closeEvent waits for the environment thread. A cold torch import on this device
#: takes a few seconds and cannot be interrupted, so the bound is generous.
ENV_SHUTDOWN_WAIT_MS = 8000


class _EnvWorker(QObject):
    """Gathers the environment report off the main thread.

    env_report imports torch, which takes seconds. Doing that on the main thread would blow
    the 3 s launch budget (section 14) and stall the event loop (rule 26).
    """

    ready = pyqtSignal(object)

    def run(self) -> None:
        try:
            from drivyx.env_report import full_report

            report = full_report()
        except Exception as exc:
            # Broad by intent: the environment report is diagnostic, and no probe failure in
            # it may take the application down.
            logger.warning("environment report failed: %s", exc)
            report = {"error": str(exc)}

        # Emitting inside the try block would let a failure in a connected slot be caught
        # and re-emitted as an "error" report, reporting a UI bug as an environment fault.
        try:
            self.ready.emit(report)
        except RuntimeError:
            # The workspace was destroyed while this thread was still importing torch, i.e.
            # the app closed during startup. Nothing to deliver the report to.
            logger.debug("environment report discarded: receiver already gone")


class SystemWorkspace(Workspace):
    """Environment report and live telemetry."""

    title = "SYSTEM"

    #: Re-emitted once the worker returns, so the main window can fill the device badge and
    #: decide the MAXN banner without probing the environment a second time.
    environment_ready = pyqtSignal(object)

    def build(self) -> None:
        self._report: dict[str, Any] = {}
        self._history: deque[tuple[float, TegraSample]] = deque(maxlen=HISTORY_SECONDS)
        self._elapsed = 0.0

        env_panel = Panel("Environment")
        self._env_rows: dict[str, StatRow] = {}
        for key in (
            "drivyx",
            "python",
            "L4T",
            "JetPack",
            "torch",
            "provenance",
            "tensorrt",
            "PyQt6",
            "opencv",
            "power",
        ):
            row = StatRow(key, "...", label_width=76)
            self._env_rows[key] = row
            env_panel.add_widget(row)
        self.add_panel(env_panel)

        actions = Panel("Diagnostics")
        copy_btn = QPushButton("Copy diagnostics to clipboard")
        copy_btn.clicked.connect(self._copy_diagnostics)
        actions.add_widget(copy_btn)
        self._copied = QLabel("")
        self._copied.setProperty("dim", "true")
        actions.add_widget(self._copied)
        self.add_panel(actions)

        live = Panel("Live")
        self._live_rows = {
            name: StatRow(name, "--", label_width=76)
            for name in ("GPU", "CPU", "RAM", "SOC temp", "GPU temp", "power")
        }
        for row in self._live_rows.values():
            live.add_widget(row)
        self.add_panel(live)

        self._chart_host = QWidget()
        chart_layout = QVBoxLayout(self._chart_host)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        self._charts: dict[str, Any] = {}
        self._chart_status = QLabel("Waiting for tegrastats ...")
        self._chart_status.setProperty("dim", "true")
        chart_layout.addWidget(self._chart_status)
        self.add_main(self._chart_host)
        self._build_charts(chart_layout)

        self._start_env_worker()

    def _build_charts(self, layout: QVBoxLayout) -> None:
        """Create the pyqtgraph plots.

        pyqtgraph is imported here rather than at module scope: it costs roughly half a
        second, and section 14's M1 gate requires the app to launch in under 3 s.
        """
        try:
            import pyqtgraph as pg
        except ImportError as exc:
            self._chart_status.setText(f"pyqtgraph unavailable: {exc}")
            return

        pg.setConfigOption("background", tokens.BG_INPUT)
        pg.setConfigOption("foreground", tokens.TEXT_DIM)

        specs = [
            ("gpu", "GPU util (%)", tokens.ACCENT, (0, 100)),
            ("ram", "RAM (GB)", tokens.OK, None),
            ("temp", "Temps (C)", tokens.WARN, None),
            ("power", "Power (W)", tokens.ERR, None),
        ]
        for key, label, color, yrange in specs:
            plot = pg.PlotWidget()
            plot.setLabel("left", label)
            plot.showGrid(x=False, y=True, alpha=0.2)
            plot.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            if yrange:
                plot.setYRange(*yrange)
            curve = plot.plot(pen=pg.mkPen(color, width=1))
            self._charts[key] = (plot, curve)
            layout.addWidget(plot)

    def _start_env_worker(self) -> None:
        self._env_thread = QThread(self)
        self._env_worker = _EnvWorker()
        self._env_worker.moveToThread(self._env_thread)
        self._env_thread.started.connect(self._env_worker.run)
        self._env_worker.ready.connect(self._on_env_ready)
        self._env_worker.ready.connect(self._env_thread.quit)
        self._env_thread.start()

    def shutdown(self) -> None:
        """Wait for the environment thread before the widget tree is torn down.

        Closing the app during startup would otherwise destroy the QThread while it is
        still importing torch, which aborts the process ("QThread: Destroyed while thread is
        still running"). The wait is bounded because a torch import cannot be interrupted;
        exceeding it is logged rather than hung on.
        """
        thread = getattr(self, "_env_thread", None)
        if thread is None or not thread.isRunning():
            return
        thread.quit()
        if not thread.wait(ENV_SHUTDOWN_WAIT_MS):
            logger.warning(
                "environment thread did not finish within %d ms; it is blocked in an "
                "uninterruptible import and will be terminated",
                ENV_SHUTDOWN_WAIT_MS,
            )
            thread.terminate()
            thread.wait(1000)

    @property
    def environment(self) -> dict[str, Any]:
        """The gathered report, empty until the worker returns."""
        return self._report

    def _on_env_ready(self, report: dict) -> None:
        self._report = report
        self.environment_ready.emit(report)
        if "error" in report:
            self._env_rows["drivyx"].set_value(f"report failed: {report['error']}")
            return

        def put(key: str, value: str, state: str | None = None) -> None:
            self._env_rows[key].set_value(value)
            if state:
                self._env_rows[key].set_value_color(tokens.state_color(state))

        put("drivyx", f"{report.get('drivyx_version')}  ({report.get('git_sha')})")
        py = report.get("python", {})
        put("python", f"{py.get('version')}  venv={py.get('in_venv')}")

        l4t = report.get("l4t", {})
        put("L4T", l4t.get("release", "unknown") if l4t.get("present") else "not a Jetson")
        put("JetPack", str(report.get("jetpack") or "not installed"))

        torch_r = report.get("torch", {})
        if torch_r.get("installed"):
            cuda = torch_r.get("cuda_build")
            put(
                "torch",
                f"{torch_r.get('version')}  CUDA {cuda or 'none'}",
                "ok" if torch_r.get("is_cuda_build") else "err",
            )
            put("provenance", f"{torch_r.get('wheel_variant')} via {torch_r.get('source', '?')}")
        else:
            put("torch", "not installed", "err")

        trt = report.get("tensorrt", {})
        put("tensorrt", str(trt.get("version", "not installed")))

        pyqt = report.get("pyqt", {})
        put(
            "PyQt6",
            f"{pyqt.get('pyqt_version', '?')}  Qt {pyqt.get('qt_version', '?')}  "
            f"({pyqt.get('source', '?')})",
        )

        cv = report.get("opencv", {})
        put(
            "opencv",
            f"{cv.get('version', '?')}  headless={cv.get('headless_ok')}",
            "ok" if cv.get("headless_ok") else "warn",
        )

        power = report.get("power", {})
        is_maxn = power.get("is_maxn")
        put(
            "power",
            str(power.get("mode_name", "unknown")),
            "ok" if is_maxn else "warn",
        )

    def on_sample(self, sample: TegraSample) -> None:
        """Update readouts and charts from a tegrastats sample."""
        self._elapsed += 1.0
        self._history.append((self._elapsed, sample))
        self._chart_status.setVisible(False)

        if sample.gpu_pct is not None:
            self._live_rows["GPU"].set_value(f"{sample.gpu_pct}%")
        if sample.cpu_avg_pct is not None:
            self._live_rows["CPU"].set_value(
                f"{sample.cpu_avg_pct:.0f}% avg over {len(sample.cpu_pct)} cores"
            )
        if sample.ram_used_gb is not None and sample.ram_total_gb is not None:
            self._live_rows["RAM"].set_value(
                f"{sample.ram_used_gb:.1f} / {sample.ram_total_gb:.0f} GB"
            )
        if sample.soc_temp_c is not None:
            self._live_rows["SOC temp"].set_value(f"{sample.soc_temp_c:.1f} C")
        if sample.gpu_temp_c is not None:
            self._live_rows["GPU temp"].set_value(f"{sample.gpu_temp_c:.1f} C")
        if sample.total_power_w is not None:
            self._live_rows["power"].set_value(f"{sample.total_power_w:.1f} W")

        self._update_charts()

    def _update_charts(self) -> None:
        if not self._charts:
            return
        xs = [t for t, _ in self._history]
        series = {
            "gpu": [s.gpu_pct or 0 for _, s in self._history],
            "ram": [s.ram_used_gb or 0 for _, s in self._history],
            "temp": [s.soc_temp_c or 0 for _, s in self._history],
            "power": [s.total_power_w or 0 for _, s in self._history],
        }
        for key, values in series.items():
            entry = self._charts.get(key)
            if entry is not None:
                entry[1].setData(xs, values)

    def on_monitor_failed(self, message: str) -> None:
        """Section 12.5: SYSTEM shows ERR, everything else keeps working."""
        self._chart_status.setVisible(True)
        self._chart_status.setText(message)
        self._chart_status.setStyleSheet(f"color: {tokens.ERR}; background: transparent;")
        for row in self._live_rows.values():
            row.set_value("unavailable")

    def _copy_diagnostics(self) -> None:
        payload = json.dumps(self._report, indent=2, default=str)
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(payload)
            self._copied.setText(f"Copied {len(payload)} bytes.")
