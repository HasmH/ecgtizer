"""Tests for strict, lossless vector ECG recognition."""

from __future__ import annotations

import numpy as np
import pytest

from ecgtizer.vector_extraction import extract_vector_svg


def _synthetic_12x1_svg() -> str:
    paths = []
    baselines = [80 + 55 * index for index in range(12)]
    for baseline in baselines:
        pulse = (
            f"M 25 {baseline} L 35 {baseline} L 35 {baseline - 40} "
            f"L 55 {baseline - 40} L 55 {baseline} L 65 {baseline}"
        )
        paths.append(
            f'<path fill="none" stroke="black" stroke-width="1" d="{pulse}"/>'
        )
    x = np.linspace(100, 900, 121)
    for lead_index, baseline in enumerate(baselines):
        y = baseline - 10 * np.sin(np.linspace(0, 6 * np.pi, len(x)) + lead_index / 5)
        commands = [f"M {x[0]:.6f} {y[0]:.6f}"]
        commands.extend(f"L {px:.6f} {py:.6f}" for px, py in zip(x[1:], y[1:]))
        path_data = " ".join(commands)
        paths.append(
            f'<path fill="none" stroke="black" stroke-width="1" d="{path_data}"/>'
        )
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 800">' + "".join(paths) + "</svg>"


def _synthetic_6x2_svg() -> str:
    paths = []
    baselines = [120 + 100 * index for index in range(6)]
    for baseline in baselines:
        pulse = (
            f"M 25 {baseline} L 35 {baseline} L 35 {baseline - 40} "
            f"L 55 {baseline - 40} L 55 {baseline} L 65 {baseline}"
        )
        paths.append(f'<path stroke="black" stroke-width="1" d="{pulse}"/>')
    for start, end in ((80, 480), (520, 920)):
        x = np.linspace(start, end, 121)
        for lead_index, baseline in enumerate(baselines):
            y = baseline - 10 * np.sin(np.linspace(0, 5 * np.pi, len(x)) + lead_index / 4)
            commands = [f"M {x[0]:.6f} {y[0]:.6f}"]
            commands.extend(f"L {px:.6f} {py:.6f}" for px, py in zip(x[1:], y[1:]))
            path_data = " ".join(commands)
            paths.append(
                f'<path stroke="black" stroke-width="1" d="{path_data}"/>'
            )
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 800">' + "".join(paths) + "</svg>"


def test_extract_vector_svg_recognises_calibrated_12x1(tmp_path):
    svg = tmp_path / "ecg.svg"
    svg.write_text(_synthetic_12x1_svg())

    result = extract_vector_svg(svg)

    assert result is not None
    assert result.layout == "12x1"
    assert len(result.leads) == 12
    assert all(len(signal) == 5000 for signal in result.leads.values())
    # A 10-point waveform over a 40-point, 1 mV calibration pulse is 250 uV.
    assert np.max(np.abs(result.leads["I"])) == pytest.approx(250, abs=2)


def test_extract_vector_svg_rejects_unvalidated_drawing(tmp_path):
    svg = tmp_path / "drawing.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 800">'
        '<path stroke="black" stroke-width="1" d="M 10 10 L 990 790"/>'
        "</svg>"
    )

    assert extract_vector_svg(svg) is None


def test_extract_vector_svg_maps_6x2_time_windows(tmp_path):
    svg = tmp_path / "ecg-6x2.svg"
    svg.write_text(_synthetic_6x2_svg())

    result = extract_vector_svg(svg)

    assert result is not None
    assert result.layout == "6x2"
    assert np.any(result.leads["I"][:2500])
    assert not np.any(result.leads["I"][2500:])
    assert not np.any(result.leads["V1"][:2500])
    assert np.any(result.leads["V1"][2500:])
