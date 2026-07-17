"""Lossless extraction for ECGs whose PDF still contains vector traces.

Rasterising a vector PDF merges every lead, label and grid line into the same
pixel mask.  At a crossing there is then no image-only fact that says which
outgoing branch belongs to which lead.  This module avoids creating that
ambiguity: it recognises repeated monotone waveform paths and calibration
pulses in an SVG representation of the PDF, validates a standard ECG layout,
and returns calibrated samples.  Anything that does not satisfy the complete
model is rejected so the normal raster pipeline can take over.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import subprocess
import tempfile
from xml.etree import ElementTree

import numpy as np


logger = logging.getLogger(__name__)

SAMPLE_RATE_HZ = 500
RECORD_SECONDS = 10
SIGNAL_LENGTH = SAMPLE_RATE_HZ * RECORD_SECONDS
AMPLITUDE_SCALE_UV = 1000.0
MAX_SVG_BYTES = 128 * 1024 * 1024

LEAD_ORDER = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
LAYOUT_LEADS = {
    "3x4": (
        ("I", "II", "III"),
        ("AVR", "AVL", "AVF"),
        ("V1", "V2", "V3"),
        ("V4", "V5", "V6"),
    ),
    "6x2": (
        ("I", "II", "III", "AVR", "AVL", "AVF"),
        ("V1", "V2", "V3", "V4", "V5", "V6"),
    ),
}

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_MATRIX_RE = re.compile(
    rf"matrix\(\s*({_NUMBER})[ ,]+({_NUMBER})[ ,]+({_NUMBER})[ ,]+"
    rf"({_NUMBER})[ ,]+({_NUMBER})[ ,]+({_NUMBER})\s*\)"
)
_PATH_TOKEN_RE = re.compile(rf"[A-Za-z]|{_NUMBER}")


@dataclass(frozen=True)
class VectorLeadPath:
    """A source waveform in PDF points, before amplitude calibration."""

    x: np.ndarray
    y: np.ndarray
    baseline: float
    gain: float


@dataclass(frozen=True)
class VectorECG:
    """Validated vector ECG extraction result."""

    leads: dict[str, np.ndarray]
    layout: str
    paths: dict[str, VectorLeadPath]
    page_width: float
    page_height: float


@dataclass
class _Chain:
    points: np.ndarray
    stroke_width: float

    @property
    def segments(self) -> int:
        return len(self.points) - 1

    @property
    def x_span(self) -> float:
        return float(np.ptp(self.points[:, 0]))

    @property
    def y_span(self) -> float:
        return float(np.ptp(self.points[:, 1]))


def _matrix(value: str | None) -> np.ndarray:
    if not value:
        return np.eye(3)
    match = _MATRIX_RE.fullmatch(value.strip())
    if match is None:
        # Strict rejection is safer than silently applying the wrong geometry.
        raise ValueError(f"Unsupported SVG transform: {value!r}")
    a, b, c, d, e, f = map(float, match.groups())
    return np.asarray(((a, c, e), (b, d, f), (0.0, 0.0, 1.0)))


def _path_points(value: str) -> np.ndarray | None:
    """Parse an absolute SVG M/L polyline; reject curves and relative paths."""
    tokens = _PATH_TOKEN_RE.findall(value.replace(",", " "))
    points: list[tuple[float, float]] = []
    command: str | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            if token not in ("M", "L", "Z"):
                return None
            command = token
            index += 1
            if command == "Z":
                continue
        if command not in ("M", "L") or index + 1 >= len(tokens):
            return None
        if tokens[index].isalpha() or tokens[index + 1].isalpha():
            return None
        points.append((float(tokens[index]), float(tokens[index + 1])))
        index += 2
        if command == "M":
            command = "L"
    if len(points) < 2:
        return None
    return np.asarray(points, dtype=float)


def _stroke_width(attributes: dict[str, str]) -> float | None:
    value = attributes.get("stroke-width")
    if value is None and "style" in attributes:
        for declaration in attributes["style"].split(";"):
            key, separator, candidate = declaration.partition(":")
            if separator and key.strip() == "stroke-width":
                value = candidate.strip()
                break
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _read_svg(svg_path: str | Path) -> tuple[float, float, dict[float, list[_Chain]]]:
    """Read paths while composing nested affine transforms."""
    transforms: list[np.ndarray] = []
    chains_by_width: dict[float, list[_Chain]] = {}
    page_width = page_height = 0.0

    for event, element in ElementTree.iterparse(svg_path, events=("start", "end")):
        if event == "start":
            parent = transforms[-1] if transforms else np.eye(3)
            current = parent @ _matrix(element.attrib.get("transform"))
            transforms.append(current)
            if not page_width and element.tag.rsplit("}", 1)[-1] == "svg":
                viewbox = element.attrib.get("viewBox", "").replace(",", " ").split()
                if len(viewbox) == 4:
                    page_width, page_height = float(viewbox[2]), float(viewbox[3])

            if element.tag.rsplit("}", 1)[-1] != "path":
                continue
            width = _stroke_width(element.attrib)
            points = _path_points(element.attrib.get("d", ""))
            if width is None or points is None:
                continue
            homogeneous = np.column_stack((points, np.ones(len(points))))
            points = (current @ homogeneous.T).T[:, :2]
            key = round(width, 6)
            bucket = chains_by_width.setdefault(key, [])
            tolerance = max(page_width, page_height, 1.0) * 2e-6
            if bucket and np.linalg.norm(bucket[-1].points[-1] - points[0]) <= tolerance:
                bucket[-1].points = np.vstack((bucket[-1].points, points[1:]))
            else:
                bucket.append(_Chain(points=points, stroke_width=width))
        else:
            transforms.pop()
            element.clear()

    if page_width <= 0 or page_height <= 0:
        raise ValueError("SVG has no usable viewBox")
    return page_width, page_height, chains_by_width


def _waveforms(chains: list[_Chain], page_width: float) -> list[_Chain]:
    result = []
    for chain in chains:
        if chain.segments < 100 or chain.x_span < 0.15 * page_width:
            continue
        differences = np.diff(chain.points[:, 0])
        tolerance = page_width * 2e-6
        if np.mean(differences >= -tolerance) < 0.995:
            continue
        result.append(chain)
    return result


def _layout(waveforms: list[_Chain], page_width: float) -> tuple[str, list[_Chain], _Chain | None] | None:
    fractions = np.asarray([chain.x_span / page_width for chain in waveforms])
    quarter = [chain for chain, fraction in zip(waveforms, fractions) if 0.15 <= fraction <= 0.30]
    half = [chain for chain, fraction in zip(waveforms, fractions) if 0.30 < fraction <= 0.58]
    full = [chain for chain, fraction in zip(waveforms, fractions) if fraction >= 0.65]
    if len(quarter) == 12 and len(full) <= 1 and not half:
        return "3x4", quarter, full[0] if full else None
    if len(half) == 12 and not quarter and not full:
        return "6x2", half, None
    if len(full) == 12 and not quarter and not half:
        return "12x1", full, None
    return None


def _calibration_pulses(
    chains: list[_Chain], page_width: float, page_height: float, expected: int
) -> list[_Chain] | None:
    candidates = []
    for chain in chains:
        if not 3 <= chain.segments <= 10:
            continue
        if not 0.008 * page_width <= chain.x_span <= 0.08 * page_width:
            continue
        if not 0.008 * page_height <= chain.y_span <= 0.12 * page_height:
            continue
        delta = np.diff(chain.points, axis=0)
        vertical = np.sum(np.abs(delta[:, 1]) > 2 * np.abs(delta[:, 0]))
        horizontal = np.sum(np.abs(delta[:, 0]) > 2 * np.abs(delta[:, 1]))
        if vertical >= 2 and horizontal >= 2:
            candidates.append(chain)
    if len(candidates) < expected:
        return None

    # Calibration marks share side, height and stroke.  Grouping by x first
    # rejects small text glyphs that happen to be step-shaped.
    best: tuple[float, list[_Chain]] | None = None
    for anchor in candidates:
        centre = float(np.mean(anchor.points[:, 0]))
        height = anchor.y_span
        group = [
            pulse
            for pulse in candidates
            if abs(float(np.mean(pulse.points[:, 0])) - centre) <= 0.02 * page_width
            and abs(pulse.y_span - height) <= 0.12 * height
        ]
        if len(group) != expected:
            continue
        heights = np.asarray([pulse.y_span for pulse in group])
        score = float(np.std(heights) / np.mean(heights))
        if best is None or score < best[0]:
            best = score, group
    if best is None:
        return None
    pulses = sorted(best[1], key=lambda pulse: float(np.max(pulse.points[:, 1])))
    baselines = np.asarray([np.max(pulse.points[:, 1]) for pulse in pulses])
    if len(baselines) > 2:
        spacing = np.diff(baselines)
        if np.std(spacing) / np.mean(spacing) > 0.25:
            return None
    return pulses


def _resample(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == length:
        return values.copy()
    if len(values) < 2:
        raise ValueError("Vector waveform contains too few samples")
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, length)
    return np.interp(target, source, values)


def _calibrated_path(
    chain: _Chain, baseline: float, gain: float, length: int
) -> tuple[np.ndarray, VectorLeadPath]:
    # One source sample is represented by each segment; its start point is the
    # sample.  Excluding the terminal endpoint therefore preserves 5000-segment
    # traces as exactly 5000 samples rather than resampling them unnecessarily.
    x = chain.points[:-1, 0].copy()
    y = chain.points[:-1, 1].copy()
    signal = (baseline - _resample(y, length)) / gain * AMPLITUDE_SCALE_UV
    return signal, VectorLeadPath(x=x, y=y, baseline=baseline, gain=gain)


def _build_result(
    layout: str,
    waveforms: list[_Chain],
    rhythm: _Chain | None,
    pulses: list[_Chain],
    page_width: float,
    page_height: float,
) -> VectorECG | None:
    baselines = np.asarray([np.max(pulse.points[:, 1]) for pulse in pulses], dtype=float)
    gain = float(np.median([pulse.y_span for pulse in pulses]))
    spacing = float(np.median(np.diff(baselines))) if len(baselines) > 1 else gain
    if not np.isfinite(gain) or gain <= 0:
        return None

    def row(chain: _Chain) -> int:
        centre = float(np.median(chain.points[:, 1]))
        return int(np.argmin(np.abs(baselines - centre)))

    if any(abs(float(np.median(chain.points[:, 1])) - baselines[row(chain)]) > 0.55 * spacing for chain in waveforms):
        return None

    leads: dict[str, np.ndarray] = {}
    paths: dict[str, VectorLeadPath] = {}
    if layout == "12x1":
        ordered = sorted(waveforms, key=row)
        if [row(chain) for chain in ordered] != list(range(12)):
            return None
        for name, chain, baseline in zip(LEAD_ORDER, ordered, baselines):
            leads[name], paths[name] = _calibrated_path(chain, baseline, gain, SIGNAL_LENGTH)
    else:
        rows = 3 if layout == "3x4" else 6
        columns = 4 if layout == "3x4" else 2
        expected_samples = SIGNAL_LENGTH // columns
        ordered = sorted(waveforms, key=lambda chain: (float(np.min(chain.points[:, 0])), row(chain)))
        for column in range(columns):
            group = ordered[column * rows : (column + 1) * rows]
            if len(group) != rows or [row(chain) for chain in group] != list(range(rows)):
                return None
            starts = np.asarray([np.min(chain.points[:, 0]) for chain in group])
            if np.ptp(starts) > 0.02 * page_width:
                return None
            for row_index, (name, chain) in enumerate(zip(LAYOUT_LEADS[layout][column], group)):
                segment, source_path = _calibrated_path(
                    chain, baselines[row_index], gain, expected_samples
                )
                full_signal = np.zeros(SIGNAL_LENGTH)
                start = column * expected_samples
                full_signal[start : start + expected_samples] = segment
                leads[name] = full_signal
                paths[name] = source_path

        if layout == "3x4" and rhythm is not None:
            rhythm_row = row(rhythm)
            if rhythm_row != 3:
                return None
            leads["IIc"], paths["IIc"] = _calibrated_path(
                rhythm, baselines[rhythm_row], gain, SIGNAL_LENGTH
            )

    if set(LEAD_ORDER) - set(leads):
        return None
    if any(not np.all(np.isfinite(signal)) or np.max(np.abs(signal)) > 20_000 for signal in leads.values()):
        return None
    return VectorECG(
        leads=leads,
        layout=layout,
        paths=paths,
        page_width=page_width,
        page_height=page_height,
    )


def extract_vector_svg(svg_path: str | Path) -> VectorECG | None:
    """Extract a validated ECG from an SVG, returning ``None`` on mismatch."""
    try:
        page_width, page_height, chains_by_width = _read_svg(svg_path)
        results = []
        for chains in chains_by_width.values():
            detected = _layout(_waveforms(chains, page_width), page_width)
            if detected is None:
                continue
            layout, waveforms, rhythm = detected
            expected_tracks = {"3x4": 4, "6x2": 6, "12x1": 12}[layout]
            pulses = _calibration_pulses(chains, page_width, page_height, expected_tracks)
            if pulses is None:
                continue
            result = _build_result(
                layout, waveforms, rhythm, pulses, page_width, page_height
            )
            if result is not None:
                results.append((sum(chain.segments for chain in waveforms), result))
        if not results:
            return None
        return max(results, key=lambda item: item[0])[1]
    except (ElementTree.ParseError, OSError, ValueError, FloatingPointError) as error:
        logger.debug("Vector ECG recognition failed for %s: %s", svg_path, error)
        return None


def extract_vector_ecg(pdf_path: str | Path, timeout: float = 30.0) -> VectorECG | None:
    """Try lossless PDF extraction; safely return ``None`` for raster PDFs."""
    pdf_path = Path(pdf_path)
    if pdf_path.suffix.lower() != ".pdf":
        return None
    try:
        information = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        page_match = re.search(r"^Pages:\s+(\d+)\s*$", information.stdout, re.MULTILINE)
        if page_match is None or int(page_match.group(1)) != 1:
            return None
        with tempfile.TemporaryDirectory(prefix="ecgtizer-vector-") as directory:
            output = Path(directory) / "page.svg"
            subprocess.run(
                [
                    "pdftocairo",
                    "-f",
                    "1",
                    "-l",
                    "1",
                    "-svg",
                    str(pdf_path),
                    str(output),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            if output.stat().st_size > MAX_SVG_BYTES:
                logger.debug("Vector SVG rejected because it exceeds %d bytes", MAX_SVG_BYTES)
                return None
            return extract_vector_svg(output)
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as error:
        logger.debug("Vector PDF conversion failed for %s: %s", pdf_path, error)
        return None
