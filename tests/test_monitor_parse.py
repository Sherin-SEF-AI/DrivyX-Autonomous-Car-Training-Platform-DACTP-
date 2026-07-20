"""tegrastats parsing (CLAUDE.md section 12.5).

The fixtures are verbatim lines captured from this Orin on JetPack 7.2, so the parser is
tested against the real format rather than an assumed one. These are CPU tests: parsing is a
pure string function and needs no Jetson and no Qt event loop.
"""

from __future__ import annotations

import pytest

from drivyx.gui.monitor import format_status, parse_tegrastats

# Captured from the device, 2026-07-17.
JP72_LINE = (
    "07-17-2026 15:15:27 RAM 8179/62817MB (lfb 1x4MB) CPU "
    "[35%@729,24%@729,16%@729,18%@729,21%@729,21%@729,15%@729,15%@729,"
    "27%@1113,20%@1113,28%@1113,32%@1113] GR3D_FREQ 32% "
    "cpu@51.968C/51.968C soc2@48.968C/48.968C soc0@49.406C/49.5C gpu@47.125C/47.125C "
    "tj@51.968C/51.968C soc1@47.906C/48.093C "
    "VDD_GPU_SOC 4385mW/3787mW/4385mW VDD_CPU_CV 1196mW/897mW/1196mW "
    "VIN_SYS_5V0 5552mW/5274mW/5552mW"
)

# JetPack 6 era rail naming, to prove rails are matched generically (not by name).
JP6_LINE = (
    "RAM 4096/30536MB (lfb 100x4MB) CPU [10%@2201,5%@2201] GR3D_FREQ 55% "
    "gpu@45C tj@50C POM_5V_IN 4000mW/3900mW/4200mW POM_5V_GPU 1000mW/900mW/1100mW"
)


def test_parses_jetpack72_line() -> None:
    s = parse_tegrastats(JP72_LINE)

    assert s.gpu_pct == 32
    assert s.ram_used_mb == 8179
    assert s.ram_total_mb == 62817
    assert s.cpu_pct == [35, 24, 16, 18, 21, 21, 15, 15, 27, 20, 28, 32]
    assert s.temps_c == {
        "cpu": 51.968,
        "soc2": 48.968,
        "soc0": 49.406,
        "gpu": 47.125,
        "tj": 51.968,
        "soc1": 47.906,
    }
    assert s.power_mw == {"VDD_GPU_SOC": 4385, "VDD_CPU_CV": 1196, "VIN_SYS_5V0": 5552}


def test_cpu_frequencies_are_not_read_as_temperatures() -> None:
    """`35%@729` and `gpu@47.125C` share an @; only the latter is a sensor.

    This is the parser's sharpest edge: a regex that ignores the C suffix would silently
    report core clocks as temperatures.
    """
    s = parse_tegrastats(JP72_LINE)

    assert 729.0 not in s.temps_c.values()
    assert 1113.0 not in s.temps_c.values()
    assert all(v < 200 for v in s.temps_c.values()), s.temps_c


def test_derived_readouts() -> None:
    s = parse_tegrastats(JP72_LINE)

    assert s.ram_used_gb == pytest.approx(7.98, abs=0.01)
    assert s.ram_total_gb == pytest.approx(61.34, abs=0.01)
    assert s.cpu_avg_pct == pytest.approx(22.67, abs=0.01)
    # Hottest of soc0/soc1/soc2, not tj and not cpu.
    assert s.soc_temp_c == 49.406
    assert s.gpu_temp_c == 47.125
    # Sum of the instantaneous fields of all three rails.
    assert s.total_power_w == pytest.approx((4385 + 1196 + 5552) / 1000)


def test_generic_rail_matching_handles_jetpack6_names() -> None:
    """Rail names differ across JetPack releases, so they must not be hardcoded."""
    s = parse_tegrastats(JP6_LINE)

    assert s.power_mw == {"POM_5V_IN": 4000, "POM_5V_GPU": 1000}
    assert s.total_power_w == pytest.approx(5.0)
    assert s.gpu_pct == 55
    assert s.cpu_pct == [10, 5]


def test_soc_temp_falls_back_to_tj() -> None:
    """JP6_LINE has no soc* sensors, so the junction temp stands in."""
    assert parse_tegrastats(JP6_LINE).soc_temp_c == 50.0


def test_garbage_line_degrades_without_raising() -> None:
    """A format change must degrade one readout, not kill the monitor thread."""
    s = parse_tegrastats("this is not tegrastats output")

    assert s.gpu_pct is None
    assert s.ram_used_mb is None
    assert s.cpu_pct == []
    assert s.soc_temp_c is None
    assert s.total_power_w is None


def test_empty_line() -> None:
    assert parse_tegrastats("").gpu_pct is None


def test_status_format_matches_spec() -> None:
    """Section 12.3: `GPU 87%  MEM 21.4/64G  SOC 71C  PWR 48W`."""
    text = format_status(parse_tegrastats(JP72_LINE))

    assert "GPU" in text and "MEM" in text and "SOC" in text and "PWR" in text
    assert "32%" in text
    assert "49C" in text
    assert "11W" in text


def test_status_format_handles_missing_fields() -> None:
    """Monospace alignment must survive a missing field (section 12.1)."""
    text = format_status(parse_tegrastats("garbage"))

    assert "--" in text
    assert text.startswith("GPU")
