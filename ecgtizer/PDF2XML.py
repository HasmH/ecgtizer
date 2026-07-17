"""Core image-processing pipeline for ECG digitization.

Converts PDF/image ECGs into numerical signal arrays through noise detection,
adaptive binarization, track segmentation, waveform extraction and amplitude
calibration using a reference pulse.
"""

from __future__ import annotations

import logging

import numpy as np
from pdf2image import convert_from_path, exceptions
from .extraction_functions import lazy_extraction, full_extraction, fragmented_extraction
import cv2
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# --- Signal parameters ---
SAMPLING_FREQ = 500  # Hz
SIGNAL_LENGTH_STANDARD = 5000  # 10 seconds at 500 Hz (Wellue/other)
SIGNAL_LENGTH_CLASSIC = 5140  # Classic format signal length
SIGNAL_LENGTH_KARDIA = 4000  # Kardia format signal length
AMPLITUDE_SCALE_UV = 1000  # Scaling factor for µV conversion

# --- Reference pulse lengths (in samples) ---
REF_PULSE_APPLE = 180
REF_PULSE_KARDIA = 240
REF_PULSE_GENERIC = 300
REF_PULSE_CLASSIC = 330
# Share of a classic track's width taken up by the reference pulse, i.e. the
# leading slice the pipeline elsewhere treats as the pulse
# (SIGNAL_LENGTH_CLASSIC - SIGNAL_LENGTH_STANDARD, out of SIGNAL_LENGTH_CLASSIC).
# Searching wider than this for the calibration square runs into real beats --
# 15% of a 10-second track is 1.5s, two whole QRS complexes, whose spans are
# easily mistaken for the square.
REF_PULSE_WIDTH_FRACTION = (SIGNAL_LENGTH_CLASSIC - SIGNAL_LENGTH_STANDARD) / SIGNAL_LENGTH_CLASSIC

# How far a single lead's measured calibration height may sit from the median
# across leads before it is treated as a mismeasurement and replaced.  One page
# is printed at one gain, so the only honest spread here is a pixel or two of
# measurement noise on a ~200px square.
CALIB_AGREEMENT_MIN = 0.85
CALIB_AGREEMENT_MAX = 1.15

# --- Printed annotations (lead labels) ---
# Dilation radius used to pull a structure's parts into one blob before its
# size is judged, as a fraction of track height.  Large enough to join the
# strokes of a glyph ("III" is three separate bars ~23px apart), small enough
# not to weld a label onto a trace passing nearby.
TEXT_MERGE_FRACTION = 0.05
# A blob smaller than this on BOTH axes (as a fraction of track height) is
# printed text rather than signal.  Measured on a real 12x1 page: labels are
# 19-65px wide and 20-52px tall on a ~273px lane, while waveform blobs span
# 145-275px vertically -- so the two populations are far apart and this cutoff
# sits in the gap.
TEXT_MAX_SIZE_FRACTION = 0.25

# --- Image noise/variance thresholds ---
VARIANCE_NOISY = 3000  # Above this → image is noisy
VARIANCE_HIGH = 2000  # Above this or below LOW → might be noisy
VARIANCE_LOW = 600  # Below HIGH or above this → might be noisy
NOISE_PARTIAL = 0.5  # Intermediate noise state

# --- Image processing thresholds ---
LINE_VARIANCE_MIN = 1000  # Min row variance to keep during image cleanup
COLUMN_VARIANCE_MIN = 200  # Min column variance to keep during image cleanup
WAVEFORM_VARIANCE_MIN = 200  # Min vertical variance to detect signal presence

# --- Pixel values ---
WHITE_PIXEL = 255

# --- PDF rasterization safety caps (decompression-bomb defense) ---
MAX_PDF_PAGES = 5  # ECG printouts are single-page; allow margin
MAX_DPI = 1200  # 2.4x typical 500 DPI; refuse pathological values
MIN_DARK_CORE_DPI = 250  # ~3.5 pixels for a 1-point printed trace

# --- Lead timing boundaries (samples) ---
LEAD_TIME_3X4 = {
    "I": (0, 1250),
    "II": (0, 1250),
    "III": (0, 1250),
    "AVR": (1250, 2500),
    "AVL": (1250, 2500),
    "AVF": (1250, 2500),
    "V1": (2500, 3750),
    "V2": (2500, 3750),
    "V3": (2500, 3750),
    "V4": (3750, 5000),
    "V5": (3750, 5000),
    "V6": (3750, 5000),
    "IIc": (0, 5000),
}
LEAD_TIME_6X2 = {
    "I": (0, 2500),
    "II": (0, 2500),
    "III": (0, 2500),
    "AVR": (0, 2500),
    "AVL": (0, 2500),
    "AVF": (0, 2500),
    "V1": (2500, 5000),
    "V2": (2500, 5000),
    "V3": (2500, 5000),
    "V4": (2500, 5000),
    "V5": (2500, 5000),
    "V6": (2500, 5000),
}
# In a 12x1 layout every lead is printed on its own full-width track,
# so each one spans the whole 10-second record.
LEAD_TIME_12X1 = {
    "I": (0, 5000),
    "II": (0, 5000),
    "III": (0, 5000),
    "AVR": (0, 5000),
    "AVL": (0, 5000),
    "AVF": (0, 5000),
    "V1": (0, 5000),
    "V2": (0, 5000),
    "V3": (0, 5000),
    "V4": (0, 5000),
    "V5": (0, 5000),
    "V6": (0, 5000),
}

# Default lead order of a classic printout, top to bottom then left to right.
LEAD_ORDER_DEFAULT = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")

# Number of tracks on the page -> classic layout name.
CLASSIC_LAYOUTS = {4: "3x4", 6: "6x2", 12: "12x1"}

# Minimum spacing between track-detection peaks, as a fraction of page height.
# It has to land between two measured bounds:
#   - Tracks never span the whole page: on real 12x1 A4 printouts the header
#     leaves the 12 tracks ~H/15 apart (measured: 273px gaps on a 4135px page).
#     A coarser minimum makes neighbouring tracks suppress each other, which is
#     why H/10 and H/14 top out at 6-7 tracks on a 12x1 page.
#   - Each track also raises a weaker secondary peak about halfway to its
#     neighbour (~H/30).  A finer minimum admits those as phantom tracks.
# H/20 sits between the two.  Measured on the sample ECGs, 3x4 and 6x2 pages
# return 4 and 6 tracks for every value in this range, so widening it here does
# not disturb them.
TRACK_PEAK_DISTANCE_RATIO = 20

# Track finding is a model-selection problem, not a generic peak-picking
# problem.  Classic ECGs have one of three row counts and (within print/scan
# tolerance) equally spaced isoelectric baselines.  Requiring that geometry is
# what prevents a large QRS, a header rule, or footer text from becoming a
# phantom track.
TRACK_COUNTS = (12, 6, 4)
BASELINE_PROMINENCE_FRACTION = 0.18
BASELINE_POSITION_TOLERANCE = 0.25

# A track corridor deliberately overlaps its neighbours.  Half-way cuts lose
# every part of a QRS that leaves its lane; a +/-2.5-lane corridor retains
# clinically plausible excursions while the continuity optimiser assigns the
# correct trace within it.
TRACK_CORRIDOR_HALF_SPACING = 2.5
LONG_HORIZONTAL_RULE_FRACTION = 0.18
LONG_VERTICAL_RULE_FRACTION = 0.25


class ECGTrack(np.ndarray):
    """Binary track view carrying the geometry needed for reconstruction.

    It remains an ``ndarray`` so existing callers and OpenCV continue to work.
    The metadata removes two old guesses downstream: where the target baseline
    sits in an overlapping crop, and which horizontal interval is waveform
    rather than a calibration pulse.
    """

    _metadata = (
        "baseline_row",
        "spacing",
        "neighbour_baselines",
        "origin_x",
        "origin_y",
        "waveform_x0",
        "waveform_x1",
        "calibration_height",
        "calibration_side",
    )

    def __new__(cls, array: np.ndarray, **metadata):
        obj = np.asarray(array).view(cls)
        for name in cls._metadata:
            setattr(obj, name, metadata.get(name))
        return obj

    def __array_finalize__(self, source):
        if source is None:
            return
        for name in self._metadata:
            setattr(self, name, getattr(source, name, None))


class ExtractedSignal(np.ndarray):
    """One-dimensional trace with its exact page-image origin."""

    def __new__(cls, array: np.ndarray, origin_x: int = 0, origin_y: int = 0):
        obj = np.asarray(array, dtype=float).view(cls)
        obj.origin_x = int(origin_x)
        obj.origin_y = int(origin_y)
        return obj

    def __array_finalize__(self, source):
        if source is None:
            return
        self.origin_x = getattr(source, "origin_x", 0)
        self.origin_y = getattr(source, "origin_y", 0)


def classic_layout(n_tracks: int) -> str:
    """Return the classic layout name for a given number of tracks.

    Parameters
    ----------
    n_tracks : int
        Number of tracks segmented from the page.

    Returns
    -------
    str
        One of ``"3x4"``, ``"6x2"`` or ``"12x1"``.

    Raises
    ------
    ValueError
        If the track count matches no known classic layout.
    """
    try:
        return CLASSIC_LAYOUTS[n_tracks]
    except KeyError:
        raise ValueError(
            "Unsupported classic ECG layout: %d tracks detected, expected one of %s "
            "(4=3x4, 6=6x2, 12=12x1)." % (n_tracks, sorted(CLASSIC_LAYOUTS))
        ) from None


def _looks_like_ecg_grid(image: np.ndarray) -> bool:
    """Return whether an image contains a page-spanning ECG paper grid.

    Page orientation cannot identify a device format: two of the supplied
    classic 3x4 ECGs are portrait A4 pages containing a landscape ECG strip.
    A classic grid is a much stronger cue.  We count long dark horizontal
    rules at two thresholds so this works for both black vector grids and
    faint red/grey scanned paper.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    def long_rule_count(threshold: int) -> int:
        coverage = (gray < threshold).mean(axis=1)
        rows = np.flatnonzero(coverage > 0.25)
        if len(rows) == 0:
            return 0
        groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
        return sum(bool(len(group)) for group in groups)

    return max(long_rule_count(230), long_rule_count(245)) >= 6


def _fit_regular_baselines(
    candidates: np.ndarray,
    prominences: np.ndarray,
    count: int,
    image_bin: np.ndarray,
):
    """Fit one equally spaced baseline sequence to projection peaks."""
    if len(candidates) < count:
        return None

    best = None
    for first_index in range(len(candidates) - 1):
        for last_index in range(first_index + 1, len(candidates)):
            first = float(candidates[first_index])
            last = float(candidates[last_index])
            spacing = (last - first) / (count - 1)
            if spacing < 4:
                continue

            predicted = first + spacing * np.arange(count)
            nearest = np.abs(candidates[:, None] - predicted[None, :]).argmin(axis=0)
            if len(np.unique(nearest)) != count:
                continue
            sample_index = np.arange(count, dtype=float)
            fitted_spacing, fitted_origin = np.polyfit(sample_index, candidates[nearest], 1)
            fitted = fitted_origin + fitted_spacing * sample_index
            residual = np.abs(candidates[nearest] - fitted) / max(fitted_spacing, 1.0)
            if np.any(residual > BASELINE_POSITION_TOLERANCE):
                continue

            selected_prominence = prominences[nearest]
            typical_prominence = float(np.median(selected_prominence))
            # A false sequence on portrait scans consisted of the four real
            # baselines plus weak header and footer peaks.  Real layouts may
            # contain one low-amplitude lead, but not two weak bookends.
            weak = selected_prominence < 0.5 * typical_prominence
            if np.count_nonzero(weak) > 1:
                continue

            radius = max(2, int(round(0.06 * spacing)))
            coverage = np.asarray(
                [
                    np.any(image_bin[max(0, row - radius) : row + radius + 1] > 0, axis=0).mean()
                    for row in candidates[nearest]
                ]
            )
            # Header/footer additions to a real sequence are spatially local;
            # a waveform remains visible over most of the recording width.
            if np.count_nonzero(coverage < 0.6 * np.median(coverage)) > 1:
                continue

            # Prefer strong baseline evidence, then the most regular sequence.
            strength = float(np.mean(selected_prominence) / max(prominences.max(), 1.0))
            score = strength + float(np.mean(coverage)) - float(np.mean(residual)) + 0.01 * count
            if best is None or score > best[0]:
                rows = candidates[nearest] if count == 4 else np.rint(fitted)
                best = (score, np.sort(rows.astype(int)))
    return best


def _fit_run_endpoints(runs: list[tuple[int, int, int]], count: int):
    """Fit equal-height calibration edges to one regular baseline sequence."""
    if len(runs) < count:
        return None
    best = None
    lengths = np.asarray([run[2] for run in runs], dtype=float)
    for reference_height in lengths:
        height_keep = (lengths >= 0.75 * reference_height) & (lengths <= 1.25 * reference_height)
        selected_runs = [run for run, keep in zip(runs, height_keep) if keep]
        if len(selected_runs) < count:
            continue
        endpoints = np.asarray([run[1] for run in selected_runs], dtype=float)
        selected_heights = np.asarray([run[2] for run in selected_runs], dtype=float)
        order = np.argsort(endpoints)
        endpoints = endpoints[order]
        selected_heights = selected_heights[order]

        for first_index in range(len(endpoints) - 1):
            for last_index in range(first_index + 1, len(endpoints)):
                spacing = (endpoints[last_index] - endpoints[first_index]) / (count - 1)
                if spacing < 4:
                    continue
                predicted = endpoints[first_index] + spacing * np.arange(count)
                nearest = np.abs(endpoints[:, None] - predicted[None, :]).argmin(axis=0)
                if len(np.unique(nearest)) != count:
                    continue
                sample_index = np.arange(count, dtype=float)
                fitted_spacing, fitted_origin = np.polyfit(sample_index, endpoints[nearest], 1)
                fitted = fitted_origin + fitted_spacing * sample_index
                residual = np.abs(endpoints[nearest] - fitted) / max(fitted_spacing, 1.0)
                if np.any(residual > 0.2):
                    continue
                heights = selected_heights[nearest]
                height_cv = float(np.std(heights) / max(np.mean(heights), 1.0))
                score = -float(np.mean(residual)) - height_cv
                if best is None or score > best[0]:
                    # The fourth row in a 3x4 printout is a rhythm strip and
                    # may intentionally have a larger gap.  Preserve measured
                    # pulse endpoints there; 6x2/12x1 rows are regular and
                    # benefit from the least-squares fit at grid intersections.
                    rows = endpoints[nearest] if count == 4 else fitted
                    best = (score, rows, float(np.median(heights)))
    return best


def _detect_baselines_from_calibration(image_bin: np.ndarray):
    """Detect track baselines from the calibration square's repeated edges."""
    height, width = image_bin.shape
    scale = min(height, width)
    close_radius = max(2, int(round(0.002 * scale)))
    closed = cv2.morphologyEx(
        image_bin,
        cv2.MORPH_CLOSE,
        np.ones((2 * close_radius + 1, 1), np.uint8),
    )
    search_width = max(1, int(0.2 * width))
    search_columns = np.concatenate(
        [np.arange(search_width), np.arange(max(search_width, width - search_width), width)]
    )
    min_height = max(8, int(round(0.012 * scale)))
    max_height = max(min_height + 1, int(round(0.09 * height)))

    column_models = []
    for x in search_columns:
        positions = np.flatnonzero(closed[:, x] > 0)
        if len(positions) == 0:
            continue
        groups = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
        runs = []
        for group in groups:
            top, bottom = int(group[0]), int(group[-1])
            run_height = bottom - top + 1
            if min_height <= run_height <= max_height:
                runs.append((top, bottom, run_height))
        for count in TRACK_COUNTS:
            model = _fit_run_endpoints(runs, count)
            if model is not None:
                model_score, endpoints, pulse_height = model
                column_models.append((count, model_score, int(x), endpoints, pulse_height))
                break

    if not column_models:
        return None

    # A real square supplies the same endpoint sequence twice.  Requiring the
    # pair rejects aligned QRS strokes and page furniture.
    best = None
    for i, left in enumerate(column_models[:-1]):
        for right in column_models[i + 1 :]:
            if left[0] != right[0]:
                continue
            separation = right[2] - left[2]
            if not (0.01 * width <= separation <= 0.08 * width):
                continue
            if (left[2] < width / 2) != (right[2] < width / 2):
                continue
            spacing = float(np.median(np.diff(left[3])))
            endpoint_error = float(np.mean(np.abs(left[3] - right[3])) / max(spacing, 1.0))
            height_error = abs(left[4] - right[4]) / max(0.5 * (left[4] + right[4]), 1.0)
            if endpoint_error > 0.15 or height_error > 0.25:
                continue
            # A high-resolution grid can itself produce a short regular run
            # sequence at one x coordinate.  Real calibration baselines span
            # the track block; a 12-row sequence occupying only the bottom
            # third of the page is a grid harmonic, not a 12x1 layout.
            baseline_span = float(
                0.5 * ((left[3][-1] - left[3][0]) + (right[3][-1] - right[3][0]))
            )
            minimum_span = {12: 0.5, 6: 0.25, 4: 0.12}[left[0]] * height
            if baseline_span < minimum_span:
                continue
            score = 10 * left[0] + left[1] + right[1] - endpoint_error - height_error
            if best is None or score > best[0]:
                baselines = np.rint(0.5 * (left[3] + right[3])).astype(int)
                pulse_height = float(np.median([left[4], right[4]]))
                side = "left" if 0.5 * (left[2] + right[2]) < width / 2 else "right"
                best = (score, baselines, pulse_height, side, left[2], right[2])
    return None if best is None else best[1:]


def _detect_track_baselines(image_bin: np.ndarray) -> np.ndarray:
    """Detect the 4, 6, or 12 regularly spaced classic ECG baselines."""
    calibration_model = _detect_baselines_from_calibration(image_bin)
    if calibration_model is not None and len(calibration_model[0]) == 12:
        return calibration_model[0]

    height = image_bin.shape[0]
    projection = gaussian_filter1d((image_bin > 0).sum(axis=1).astype(float), sigma=max(1.0, height / 1200))
    peaks, properties = find_peaks(
        projection,
        distance=max(3, height // 100),
        prominence=1,
    )
    if len(peaks) == 0:
        if calibration_model is not None:
            return calibration_model[0]
        raise ValueError("No ECG track baselines were detected.")

    prominence = properties["prominences"]
    keep = prominence >= BASELINE_PROMINENCE_FRACTION * prominence.max()
    candidates = peaks[keep]
    candidate_prominence = prominence[keep]

    fits = []
    for count in TRACK_COUNTS:
        fitted = _fit_regular_baselines(candidates, candidate_prominence, count, image_bin)
        if fitted is not None:
            _, fitted_rows = fitted
            calibration_support = _detect_horizontal_geometry(image_bin, fitted_rows)[-1]
            has_full_calibration = calibration_support >= int(np.ceil(0.8 * count))
            fits.append((fitted[0], fitted_rows, count, has_full_calibration))
    if fits:
        # A true 12x1 baseline sequence contains valid 6- and 4-row subsets.
        # At low gain those subsets can score slightly higher because their
        # traces are stronger.  Page coverage resolves the nesting without a
        # DPI- or amplitude-specific threshold.
        full_height_twelve = [
            fit
            for fit in fits
            if fit[2] == 12 and (fit[1][-1] - fit[1][0]) >= 0.5 * height
        ]
        if full_height_twelve:
            return max(full_height_twelve, key=lambda fit: fit[0])[1]
        if calibration_model is not None:
            return calibration_model[0]
        pulse_validated = [fit for fit in fits if fit[3]]
        if pulse_validated:
            # Every subset of a 12x1 layout can look regular; the repeated
            # calibration edge tells us that all twelve rows are real tracks.
            return max(pulse_validated, key=lambda fit: (fit[2], fit[0]))[1]
        return max(fits, key=lambda fit: fit[0])[1]

    if calibration_model is not None:
        return calibration_model[0]
    raise ValueError(
        "Could not fit detected row peaks to a supported classic ECG layout "
        "(expected 4, 6, or 12 regularly spaced tracks)."
    )


def _vertical_run_at_baseline(column: np.ndarray, baseline: int, spacing: float):
    """Return calibration-like run height at a baseline, or ``None``."""
    tolerance = max(2, int(0.08 * spacing))
    lo = max(0, int(baseline - 1.3 * spacing))
    hi = min(len(column), int(baseline + 1.3 * spacing) + 1)
    positions = np.flatnonzero(column[lo:hi])
    if len(positions) == 0:
        return None
    groups = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
    for group in groups:
        top = lo + int(group[0])
        bottom = lo + int(group[-1])
        run_height = bottom - top + 1
        if (
            0.2 * spacing <= run_height <= 1.25 * spacing
            and (abs(bottom - baseline) <= tolerance or abs(top - baseline) <= tolerance)
        ):
            return float(run_height)
    return None


def _detect_horizontal_geometry(image_bin: np.ndarray, baselines: np.ndarray):
    """Find waveform bounds and an edge calibration square.

    Calibration pulses are recognised by repeated vertical edges terminating
    at every track baseline.  This is independent of whether the vendor prints
    the pulse to the left (the vector samples) or right (the scanned 3x4
    samples).  The returned waveform bounds exclude the pulse.
    """
    spacing = float(np.median(np.diff(baselines)))
    y0 = max(0, int(baselines[0] - 0.5 * spacing))
    y1 = min(image_bin.shape[0], int(baselines[-1] + 0.5 * spacing) + 1)
    column_ink = (image_bin[y0:y1] > 0).sum(axis=0)
    active_columns = np.flatnonzero(column_ink)
    if len(active_columns) == 0:
        return 0, image_bin.shape[1], 0.0, None, 0, image_bin.shape[1], 0

    active_x0 = int(np.quantile(active_columns, 0.001))
    active_x1 = int(np.quantile(active_columns, 0.999)) + 1
    active_width = max(1, active_x1 - active_x0)

    # Grid removal can leave one-pixel cuts in a pulse edge.  Close only in the
    # vertical direction before measuring those edges.
    close_radius = max(1, int(0.025 * spacing))
    closed = cv2.morphologyEx(
        image_bin,
        cv2.MORPH_CLOSE,
        np.ones((2 * close_radius + 1, 1), np.uint8),
    )
    edge_span = max(1, int(0.15 * active_width))
    edge_columns = np.concatenate(
        [
            np.arange(active_x0, min(active_x1, active_x0 + edge_span)),
            np.arange(max(active_x0, active_x1 - edge_span), active_x1),
        ]
    )
    scores = np.zeros(image_bin.shape[1], dtype=int)
    heights: list[list[float]] = [[] for _ in range(image_bin.shape[1])]
    for x in edge_columns:
        for baseline in baselines:
            run_height = _vertical_run_at_baseline(closed[:, x] > 0, int(baseline), spacing)
            if run_height is not None:
                scores[x] += 1
                heights[x].append(run_height)

    required = max(2, int(np.ceil(0.5 * len(baselines))))
    pulse_columns = np.flatnonzero(scores >= required)
    groups = (
        np.split(pulse_columns, np.flatnonzero(np.diff(pulse_columns) > 1) + 1)
        if len(pulse_columns)
        else []
    )
    groups = [group for group in groups if len(group)]
    if len(groups) < 2:
        return active_x0, active_x1, 0.0, None, active_x0, active_x1, 0

    # A square contributes two high-scoring vertical-edge groups.  Select the
    # pair on one page edge with the greatest cross-track support.
    best_pair = None
    for left, right in zip(groups[:-1], groups[1:]):
        separation = int(right[0] - left[-1])
        if not (0.005 * active_width <= separation <= 0.08 * active_width):
            continue
        pair_score = int(scores[left].max() + scores[right].max())
        pair_center = 0.5 * (left.mean() + right.mean())
        side = "left" if pair_center < active_x0 + active_width / 2 else "right"
        if best_pair is None or pair_score > best_pair[0]:
            best_pair = (pair_score, left, right, side)
    if best_pair is None:
        return active_x0, active_x1, 0.0, None, active_x0, active_x1, 0

    _, left_group, right_group, side = best_pair
    calibration_support = int(min(scores[left_group].max(), scores[right_group].max()))
    pulse_left = int(left_group[0])
    pulse_right = int(right_group[-1])
    pulse_width = max(1, pulse_right - pulse_left)
    measured_heights = [
        height
        for x in np.concatenate([left_group, right_group])
        for height in heights[int(x)]
    ]
    calibration_height = float(np.median(measured_heights)) if measured_heights else 0.0

    # The short baseline tail between the square and waveform is about a fifth
    # of the square width.  Keeping it out avoids a systematic time offset.
    tail = max(1, int(round(0.2 * pulse_width)))
    if side == "left":
        waveform_x0 = min(active_x1 - 1, pulse_right + tail)
        waveform_x1 = active_x1
    else:
        waveform_x0 = active_x0
        waveform_x1 = max(active_x0 + 1, pulse_left - tail)
    return waveform_x0, waveform_x1, calibration_height, side, active_x0, active_x1, calibration_support


def overlay_coordinates(track: np.ndarray, extracted_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a raw extracted track signal back into page-image coordinates."""
    origin_x = int(getattr(extracted_signal, "origin_x", getattr(track, "origin_x", 0)) or 0)
    origin_y = int(getattr(extracted_signal, "origin_y", getattr(track, "origin_y", 0)) or 0)
    values = np.asarray(extracted_signal, dtype=float)
    return np.arange(origin_x, origin_x + len(values)), values + origin_y


# A row/column is treated as a grid line once this fraction of it is lit.
# Measured on real 12x1 pages: grid rows/columns are lit ~55-90% of the way
# across the full page, while rows/columns that merely pass through the
# trace (including its flat, isoelectric stretches) stay under ~15%, since
# the trace is confined to one lane and never spans the entire page on its
# own. The gap between those two populations is wide and empirically stable
# from 0.3 to 0.7, so this is not a fragile cutoff.
GRID_LINE_FRACTION = 0.4

# ...but lit fraction alone is not enough.  Every lane carries a calibration
# pulse, and they are all printed at the same x, so their vertical strokes
# stack into a column that is lit ~0.6-0.8 of the page -- indistinguishable
# from a grid line by fraction, and removing it destroys the reference the
# amplitude calibration is measured from.  A grid line is one unbroken run
# down the page; a stack of calibration pulses is lit in bursts, one per lane.
# Measured: grid columns run 3050-3058px unbroken, calibration strokes only
# ~201px, so requiring a run of this fraction of the page separates them with
# room to spare.
GRID_MIN_RUN_FRACTION = 0.25

# How far the first/last track are pulled in from the page edge to avoid
# header and footer text, as a fraction of their own lane height (capped at
# the historical 2%-of-page value so roomier layouts are unaffected).
EDGE_TRACK_INSET_FRACTION = 0.1

# Kernel length used to bridge the cuts that grid removal leaves in the trace.
# Must exceed the thickest grid line (measured: 3px horizontal, 3px vertical,
# occasionally 5px on major lines) so the two sides of a cut can meet, while
# staying short enough not to join genuinely separate parts of the waveform.
GRID_REPAIR_SPAN = 7


def _longest_run(flags: np.ndarray) -> int:
    """Length of the longest unbroken run of ``True`` in a 1-D boolean array."""
    idx = np.where(flags)[0]
    if len(idx) == 0:
        return 0
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(idx) - 1]))
    return int((idx[ends] - idx[starts] + 1).max())


def _grid_lines(lit: np.ndarray, axis: int) -> np.ndarray:
    """Flag the rows (axis=1) or columns (axis=0) that are grid lines.

    Two tests, both needed: the line must be mostly lit
    (``GRID_LINE_FRACTION``) *and* lit as one unbroken run
    (``GRID_MIN_RUN_FRACTION``).  The run test is what keeps the stacked
    calibration pulses -- which are mostly lit but in per-lane bursts -- out
    of the grid mask.
    """
    mostly_lit = lit.mean(axis=axis) > GRID_LINE_FRACTION
    span = lit.shape[axis]
    min_run = GRID_MIN_RUN_FRACTION * span
    is_grid = np.zeros(lit.shape[1 - axis], dtype=bool)
    # Only the candidates need the (costlier) run check.
    for i in np.where(mostly_lit)[0]:
        line = lit[:, i] if axis == 0 else lit[i, :]
        is_grid[i] = _longest_run(line) > min_run
    return is_grid


def _strip_grid_lines(image_bin: np.ndarray) -> np.ndarray:
    """Remove full-width/full-height grid lines from a binary mask.

    Used when the grid could not be told apart from the trace by colour (e.g.
    a black/grayscale grid printed on the same channel as the trace, which is
    common in "print to PDF" exports of an on-screen review rather than real
    coloured thermal paper). A horizontal grid line spans nearly the entire
    printable width; a real trace, even during a flat/isoelectric stretch, is
    confined to a single lane and essentially never lights up most of the
    page end to end. Zeroing rows/columns whose lit fraction clears
    ``GRID_LINE_FRACTION`` removes the grid while leaving the trace -- flats
    included -- alone.

    Zeroing a whole row/column also punches a hole through the trace wherever
    it happens to cross a grid line, which leaves the waveform dashed and
    starves the per-column extractors in :mod:`extraction_functions` of data.
    Since grid lines are thin and the trace continues on both sides of such a
    cut, a morphological closing restricted to the grid lines themselves
    restores the trace.  The restriction is what makes this safe: away from
    the trace there is nothing on either side of a grid line to bridge, so the
    grid does not come back.

    An earlier version used morphological opening to strip any long straight
    run.  That caught the grid but also ate genuine flat trace (a page is
    mostly baseline), dropping over 90% of lit pixels and breaking track
    detection.  Removing only full-span rows/columns, then repairing, avoids
    both failure modes.

    Parameters
    ----------
    image_bin : np.ndarray
        Binary mask (255 = candidate signal) to clean.

    Returns
    -------
    np.ndarray
        ``image_bin`` with grid lines removed and the trace left continuous.
    """
    lit = image_bin > 0
    grid_rows = _grid_lines(lit, axis=1)
    grid_cols = _grid_lines(lit, axis=0)

    stripped = image_bin.copy()
    stripped[grid_rows, :] = 0
    stripped[:, grid_cols] = 0

    # Bridge the cuts: horizontal grid lines break the trace vertically and
    # vice versa, so each direction is closed with the perpendicular kernel.
    span = GRID_REPAIR_SPAN
    bridged_v = cv2.morphologyEx(stripped, cv2.MORPH_CLOSE, np.ones((span, 1), np.uint8))
    bridged_h = cv2.morphologyEx(stripped, cv2.MORPH_CLOSE, np.ones((1, span), np.uint8))
    stripped[grid_rows, :] = bridged_v[grid_rows, :]
    stripped[:, grid_cols] = np.maximum(stripped[:, grid_cols], bridged_h[:, grid_cols])
    return stripped


def _binarize_image(
    image: np.ndarray, TYPE: str, NOISE: bool | float, DPI: int | None = None
) -> np.ndarray:
    """Binarize a grayscale or BGR image using the appropriate thresholding method.

    For clean colour images (NOISE=False, not Wellue), Otsu thresholding is
    followed by a colour-based filter that removes ECG grid lines.  Standard
    ECG paper has an orange/pink grid whose brightest RGB channel is well above
    128, whereas the black trace has max channel below 50.  Discarding lit
    pixels with ``max(R,G,B) > 128`` cleanly separates trace from grid.

    The ``NOISE`` path has no colour information to rely on (it thresholds
    on brightness alone), so a black/grayscale grid survives thresholding
    indistinguishable from the trace.  :func:`_strip_grid_lines` removes it
    there using shape instead of colour.
    """
    if image.ndim == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    if NOISE and (DPI is None or DPI < MIN_DARK_CORE_DPI):
        # A fixed dark-pixel cutoff breaks at low DPI: a one-point vector
        # stroke is mostly anti-aliased grey, so calibration
        # edges become a set of short fragments and 12x1 can be mistaken for
        # 3x4.  Otsu on the lightly blurred page preserves those strokes while
        # adapting to scans with different exposure.  At >=250 DPI the stroke
        # has a stable dark core, and the conservative branch below avoids
        # admitting minor black-grid lines.
        _, image_bin = cv2.threshold(
            img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # No colour channel to separate grid from trace here (see docstring),
        # so fall back to a shape-based split.
        image_bin = _strip_grid_lines(image_bin)
    elif NOISE:
        _, image_bin = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY_INV)
        image_bin = _strip_grid_lines(image_bin)
    elif TYPE.lower() == "wellue":
        _, image_bin = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        _, image_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Remove coloured grid lines: on standard ECG paper the grid is
        # orange/pink (max channel ~150-240) while the trace is black
        # (max channel ~0-50).  Any lit pixel whose brightest channel
        # exceeds 128 is grid, not trace.
        if image.ndim == 3:
            max_channel = np.max(image, axis=2)
            image_bin[max_channel > 128] = 0
    return image_bin


def _baseline_row(track_bin: np.ndarray) -> int:
    """Estimate a lead's isoelectric baseline row within its track.

    The baseline is the one row the trace spends most of its time on, so it
    lights up far more columns than any other.
    """
    return int(np.argmax((track_bin > 0).sum(axis=1)))


def _calibrate_from_binary(
    track_bin: np.ndarray, DPI: int = 0, above: np.ndarray | None = None
) -> tuple[float, float]:
    """Measure the calibration square directly from a binary track image.

    The square is printed at the very start of the track, so only the leading
    reference-pulse slice is searched -- the same
    ``REF_PULSE_CLASSIC``-of-``SIGNAL_LENGTH_CLASSIC`` share the rest of the
    pipeline treats as the pulse.  Keeping that window narrow is what makes
    the measurement trustworthy: search further in and it starts measuring
    QRS complexes and lead labels, which are just as tall.

    Within the window the square is simply the tallest vertical run.  There is
    no need to work out which lead a run belongs to: one page is printed at
    one gain, so every square on it is the same height.

    ``above`` is the preceding track, and matters on dense layouts: 1 mV is
    197px at 500 dpi / 10 mm per mV but a 12x1 lane leaves only ~136px of
    headroom above the baseline, so the square is cut off by the track
    boundary and cannot be measured from ``track_bin`` alone.  Passing the
    track above restores the rows the cut removed.  The top lead has no track
    above and may still come back short; ``lead_cutting`` cross-checks each
    factor against the median across leads and replaces the outliers.

    Returns (pixel_zero, factor).  ``pixel_zero`` is the baseline row (bottom
    of the square) in ``track_bin`` coordinates and ``factor`` is the square
    height in pixels, corresponding to 1 mV.  Falls back to a DPI-based
    estimate when the square cannot be found.
    """
    metadata_height = getattr(track_bin, "calibration_height", None)
    metadata_baseline = getattr(track_bin, "baseline_row", None)
    if metadata_height is not None and metadata_height > 0 and metadata_baseline is not None:
        return float(metadata_baseline), float(metadata_height)

    h, w = track_bin.shape
    ref_width = max(10, int(REF_PULSE_WIDTH_FRACTION * w))
    dpi_estimate = (10 * DPI) / 25.4 if DPI > 0 else 1.0

    # Restore the rows the track cut removed, so a tall square is not measured
    # short. Only the track above matters: the square rises from the baseline.
    if above is not None and above.ndim == 2 and above.shape[1] == w:
        window = np.vstack([above, track_bin])
        offset = above.shape[0]
    else:
        window = track_bin
        offset = 0

    # Anything shorter than this is trace, not a square edge.
    min_edge = max(3, int(0.03 * h))

    runs = []  # (height, bottom_row) in window coordinates
    for c in range(ref_width):
        lit = np.where(window[:, c] == 255)[0]
        if len(lit) < 2:
            continue
        breaks = np.where(np.diff(lit) > 1)[0]
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [len(lit) - 1]))
        for s, e in zip(starts, ends):
            top, bottom = float(lit[s]), float(lit[e])
            if bottom - top > min_edge:
                runs.append((bottom - top, bottom))

    if not runs:
        return float(_baseline_row(track_bin)), dpi_estimate

    f = max(r[0] for r in runs)
    # The zero line is the base of this track's own square, so of the runs that
    # measure a full square, keep the ones ending inside this track.
    bottoms = [b - offset for (height, b) in runs
               if height >= f * 0.9 and offset <= b < offset + h]
    if bottoms:
        pixel_zero = float(np.median(bottoms))
    else:
        pixel_zero = float(_baseline_row(track_bin))
    if f < 1:
        f = dpi_estimate
    return pixel_zero, float(f)


def _calibrate_ref_pulse(ref_pulse: np.ndarray, DPI: int = 0) -> tuple[float, float]:
    """Compute calibration (pixel_zero, factor) from a reference pulse segment.

    Returns (pixel_zero, factor) where factor converts pixel distance to µV.
    """
    pixel_zero = float(max(ref_pulse[:10]) if len(ref_pulse) >= 10 else max(ref_pulse))
    pixel_one = float(min(ref_pulse))

    # Flat pulse fallback
    if np.all(np.diff(ref_pulse) == 0):
        pixel_zero = float(np.mean(ref_pulse))
        pixel_one = float(min(ref_pulse))

    f = pixel_zero - pixel_one
    if f == 0:
        if DPI > 0:
            f = (10 * DPI) / 25.4
        else:
            f = 1.0
    return pixel_zero, f


def convert_PDF2image(path_input: str, DPI: int) -> np.ndarray:
    """
    Convert the PDF file into array (images).

    We use the library pdf2image to transform the input file into an array

    Parameters
    ----------
    path_input : str, path of the pdf file to convert
    DPI :int, dots per inch (resolution of the image)

    Returns
    -------
    list : list of all the pages of the PDF in PIL format
    int  : number of pages
    bool : True: The conversion has worked / False :  The conversion has not worked
    """
    if DPI > MAX_DPI:
        logger.error("DPI %d exceeds MAX_DPI=%d; refusing to rasterize.", DPI, MAX_DPI)
        return ("_", "_", False)
    try:
        pages = convert_from_path(path_input, dpi=DPI, first_page=1, last_page=MAX_PDF_PAGES)
    except exceptions.PDFPageCountError:
        logger.error("Impossible conversion. The input file is not a PDF.")
        return ("_", "_", False)
    return (pages, len(pages), True)


def check_noise_type(image: np.ndarray, DPI: int, DEBUG: bool) -> tuple[str, bool | float]:
    """
    Check the noise level of the image. Check the type of the image.

    Parameters
    ----------
    image : np.array, image
    DPI   : int, dots per inch (resolution of the image)
    DEBUG : bool, show the image

    Returns
    -------
    str : Type of image
    bool : True: The image is noised / False : The image is not noised
    """

    # Check color diversity along the middle column (vectorized)
    mid_col = image[:, image.shape[1] // 2, :]  # shape: (H, 3)
    # Kardia: only one unique green-channel value in rows where R==255 or B is truthy
    mask = (mid_col[:, 0] == 255) | (mid_col[:, 2] != 0)
    unique_colors = np.unique(mid_col[mask, 1]) if np.any(mask) else np.array([])
    if len(unique_colors) <= 1:
        return ("Kardia", False)

    # Check the variance in the image (compute once)
    image_var = np.var(image)
    if image_var > VARIANCE_HIGH or image_var < VARIANCE_LOW:
        if image_var > VARIANCE_NOISY:
            NOISE = True
        else:
            NOISE = NOISE_PARTIAL
    else:
        NOISE = False

    # A classic ECG may be embedded sideways in a portrait PDF.  Detect the
    # paper grid before considering page aspect ratio; orientation alone was
    # misclassifying the supplied 3x4 scans as Wellue recordings.
    if _looks_like_ecg_grid(image):
        return ("classic", NOISE)

    if len(image) > len(image[0]):
        # the Wellue format offers images that are taller than they are wide
        return ("Wellue", NOISE)

    else:
        # Convert image in gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Binarize the image
        ret, thresh1 = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        # Define the rectangle original size
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.03 * len(image)), int(0.03 * len(image))))
        # Dilate the image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
        # Find contour by applying rectangle
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        im2 = image.copy()
        nbr = 0
        # Count the number of rectangles in the apple watch format there is 3 record rectangle
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w - x > len(image) / 3:
                rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 2)
                nbr += 1
        # Plot the image with the different rectangle(s) find
        if DEBUG:
            try:
                plt.figure(figsize=(20, 14))
                plt.imshow(rect)
                plt.show()
            except UnboundLocalError:
                pass
        # There are more than 3 record rectangle it is apple watch format
        if nbr >= 3:
            return ("apple", False)
        # There is less than 3 record rectangle it is a classical format
        else:
            return ("classic", NOISE)


def text_extraction(
    image: np.ndarray, page: int, DPI: int, NOISE: bool | float, TYPE: str, DEBUG: bool
) -> tuple[dict, np.ndarray, str, str]:
    """
    Extract the texte from the image and mask the task on the image
    For Kardia it mask the gride line

    Parameters
    ----------
    image : np.array, image
    DPI   : int, dots per inch (resolution of the image)
    NOISE : bool, if the image is noised or not
    TYPE  : str, format of the image
    DEBUG : bool, show the image

    Returns
    -------
    array : The image without the text
    DataFrame : The dataframe with the extracted text in it
    """
    df = []

    if TYPE.lower() == "kardia":
        # Isolate the record region
        work_image = np.array(image)[DPI : int(10 * DPI), int(0.3 * DPI) : int(8 * DPI)]
        # Convert the image in gray scale
        image_gray = cv2.cvtColor(work_image, cv2.COLOR_BGR2GRAY)
        # Binarize the image thanks to the gray scale
        new_image = np.where(image_gray == 0, WHITE_PIXEL, 0).astype(image_gray.dtype)

        # Compute the vertical variance
        var_line = np.var(new_image, axis=1)
        # Compute the horizontal variance
        var_column = np.var(new_image, axis=0)
        # Define a second image to work with
        working_image = np.copy(new_image)

        for i in range(len(new_image)):
            if var_line[i] < LINE_VARIANCE_MIN:
                working_image[i, :] = 0
        for i in range(len(new_image[0])):
            if var_column[i] < COLUMN_VARIANCE_MIN:
                working_image[:, i] = 0

        if DEBUG:
            plt.figure(figsize=(20, 14))
            plt.imshow(working_image)
            plt.show()
        return (working_image, df)

    # Classic track detection now models the repeated baseline geometry and
    # therefore does not need destructive header masking.  The old noisy-image
    # branch inferred the header from its first two variance peaks; on a dense
    # 12x1 page those peaks are leads I and II, so it erased real ECG data (the
    # first three leads in one supplied sample).  Printed labels are handled by
    # the continuity-aware trace optimiser instead of deleting image regions.
    if TYPE.lower() == "classic":
        return image

    # Table with the information patient in it

    # Convert image in gray scale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian Blur
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    # If the image is noised we apply a deterministic threshold
    if NOISE:
        # Binarize the image with the deterministic threshold
        ret, image_bin = cv2.threshold(image_gray, 40, 100, cv2.THRESH_BINARY_INV)
        # Compute the horizontal variance
        horizontal_variance = np.var(image_bin, axis=1)
        # Detect the variance peaks
        peaks = signal.argrelextrema(horizontal_variance, np.greater, order=int(len(image) / 10))[
            0
        ]  # Compute the pikes position
        # Mask the text region if peaks were found
        if len(peaks) >= 2:
            # starting position on the x-axis
            x = 0
            # Ending position on the x-axis
            w = len(image[0])
            # Starting position on the y-axis
            y = 0
            # Ending position on the y-axis
            h = int(peaks[0] + (peaks[1] - peaks[0]) / 2)
            im2 = image.copy()
            # Define and apply a mask on the text region
            rect = cv2.rectangle(image_bin, (x, peaks[0]), (x + w, y + h), (255, 0, 0), 2)
            # The mask must have the same color as the rest of the image
            image[y : y + h, x : x + w] = np.mean(image[y : y + h, x : x + w])

    # If the image is not noised we apply a Otsu detection threshold
    else:
        # Binarize the image with the Otsu threshold
        ret, image_bin = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img_h, img_w = image.shape[:2]

        # ── Pass 1: Contour-based removal of medium text blocks ──
        if TYPE == "apple":
            k_size = int(0.03 * img_h)
        else:
            k_size = max(int(0.0075 * img_h), 8)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        dilation = cv2.dilate(image_bin, rect_kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        im2 = image.copy()

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area_ratio = (w * h) / (img_w * img_h)
            if area_ratio > 0.25:
                continue
            if img_w > img_h:  # landscape
                if w < img_w / 3:
                    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    image[y : y + h, x : x + w] = (255, 255, 255)
            else:  # portrait
                if h < img_h / 4:
                    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    image[y : y + h, x : x + w] = np.mean(image[y : y + h, x : x + w])

        # ── Passes 2 & 3 apply only to low-res images (photos). ──
        # High-res PDFs have clean contours and don't need aggressive
        # character-level removal — Pass 1 is sufficient for them.
        # Threshold: ~2 megapixels separates photos from 500-DPI PDFs.
        is_lowres = (img_h * img_w) < 2_000_000

        if is_lowres:
            # ── Pass 2: Zone-based header / footer removal ──
            # In standard ECG printouts, text lives above the first
            # track and below the last track.  Use the horizontal
            # projection of the binary image to find the signal band
            # and blank everything outside it.
            row_proj = np.sum(image_bin, axis=1).astype(float)
            from scipy.ndimage import uniform_filter1d

            row_proj_smooth = uniform_filter1d(row_proj, size=max(img_h // 40, 5))
            proj_thresh = row_proj_smooth.max() * 0.05
            active_rows = np.where(row_proj_smooth > proj_thresh)[0]

            if len(active_rows) > 2:
                first_active = int(active_rows[0])
                last_active = int(active_rows[-1])
                # Header: blank above first active row
                header_end = max(0, first_active - max(int(img_h * 0.005), 2))
                if header_end > int(img_h * 0.02):
                    image[:header_end, :] = (255, 255, 255) if image.ndim == 3 else 255
                # Footer: blank below last active row
                footer_start = min(img_h, last_active + max(int(img_h * 0.005), 2))
                if (img_h - footer_start) > int(img_h * 0.02):
                    image[footer_start:, :] = (255, 255, 255) if image.ndim == 3 else 255

            # NOTE: A character-level pass (connected-component or
            # morphological) was tested here but removed — it
            # reliably destroys calibration squares on photos because
            # those squares are similar in size/shape to text blobs.
            # Lead labels within the signal area remain; they are
            # handled as noise during waveform extraction.

    # Plot the image with the detected rectangles
    if DEBUG:
        try:
            plt.figure(figsize=(20, 14))
            plt.imshow(rect)
            plt.show()
            plt.imshow(image)
            plt.show()
        except UnboundLocalError:
            plt.imshow(image)
            plt.show()
    return image


def _remove_long_rules(image_bin: np.ndarray) -> np.ndarray:
    """Remove only page-spanning horizontal and vertical print rules."""
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, int(round(image_bin.shape[1] * LONG_HORIZONTAL_RULE_FRACTION))), 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(3, int(round(image_bin.shape[0] * LONG_VERTICAL_RULE_FRACTION)))),
    )
    horizontal_rules = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_rules = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, vertical_kernel)
    return cv2.bitwise_and(
        image_bin, cv2.bitwise_not(cv2.bitwise_or(horizontal_rules, vertical_rules))
    )


def tracks_extraction(
    image: np.ndarray, TYPE: str, DPI: int, FORMAT: str, NOISE: bool | float = False, DEBUG: bool = False
) -> dict[int, np.ndarray]:
    """
    Extract the tracks from the image

    Parameters
    ----------
    image : np.array, image
    TYPE  : str, format of the image
    DPI   : int, dots per inch (resolution of the image)
    FORMAT: str, multi or unilead for Kardia
    NOISE : bool, if the image is noised or not
    DEBUG : bool, show the image

    Returns
    -------
    dictionary  : dictionary of the different extracted tracks with their position
                  (key : position / Value: Track images)
    """
    # dictionary of all tracks
    dic_tracks = {}
    if TYPE.lower() == "kardia":
        var_line = np.var(image, axis=1)
        peaks, _ = find_peaks(var_line, height=2 * DPI, distance=DPI)
        start = 0
        it = 0
        for p in range(len(peaks) - 1):
            end = (peaks[p] + peaks[p + 1]) / 2
            dic_tracks[it] = image[start : int(end), :]
            start = int(end)
            it += 1
        dic_tracks[it] = image[start:, :]

        dic_tracks_temp = {}
        it = 0
        if FORMAT == "unilead":
            for i in dic_tracks:
                if i % 2 == 0:
                    dic_tracks_temp[it] = dic_tracks[i]
                    it += 1
            if DEBUG:
                for im in dic_tracks_temp:
                    plt.imshow(dic_tracks_temp[im])
                    plt.show()
            return dic_tracks_temp

        else:
            if DEBUG:
                for im in dic_tracks:
                    plt.imshow(dic_tracks[im])
                    plt.show()
            return dic_tracks

    image_bin = _binarize_image(image, TYPE, NOISE, DPI)
    peaksh = _detect_track_baselines(image_bin)
    spacing = float(np.median(np.diff(peaksh)))
    (
        waveform_x0,
        waveform_x1,
        calibration_height,
        calibration_side,
        active_x0,
        active_x1,
        _calibration_support,
    ) = _detect_horizontal_geometry(image_bin, peaksh)

    # Major grid axes, table borders and header rules can be darker than the
    # trace and may run through an entire track.  Extract only structures that
    # span a substantial fraction of the *page* in one direction, after all
    # calibration/layout measurements have been made.  Calibration squares,
    # lead labels and even tall QRS complexes are far shorter than these
    # kernels.  Removing the rules creates at most one-pixel gaps where the
    # waveform crosses them; the trace optimiser explicitly bridges such gaps.
    extraction_bin = _remove_long_rules(image_bin)

    # Sparse layouts already give each row ample vertical room.  Their wider
    # corridors would admit header/footer text without adding real waveform;
    # dense 12x1 pages need the full overlap to retain tall QRS complexes.
    corridor_factor = {4: 1.25, 6: 1.75, 12: TRACK_CORRIDOR_HALF_SPACING}.get(
        len(peaksh), TRACK_CORRIDOR_HALF_SPACING
    )
    half_height = int(round(corridor_factor * spacing))
    for index, baseline in enumerate(peaksh):
        origin_y = max(0, int(baseline) - half_height)
        end_y = min(image_bin.shape[0], int(baseline) + half_height + 1)
        local_baselines = peaksh[(peaksh >= origin_y) & (peaksh < end_y)] - origin_y
        target_baseline = int(baseline) - origin_y
        neighbours = local_baselines[local_baselines != target_baseline].astype(float)
        track_view = extraction_bin[origin_y:end_y, active_x0:active_x1]
        dic_tracks[index] = ECGTrack(
            track_view,
            baseline_row=float(target_baseline),
            spacing=spacing,
            neighbour_baselines=neighbours,
            origin_x=active_x0,
            origin_y=origin_y,
            waveform_x0=max(0, waveform_x0 - active_x0),
            waveform_x1=min(active_x1 - active_x0, waveform_x1 - active_x0),
            calibration_height=calibration_height,
            calibration_side=calibration_side,
        )

    if DEBUG:
        projection = gaussian_filter1d((image_bin > 0).sum(axis=1).astype(float), sigma=max(1.0, len(image) / 1200))
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(image)
        for baseline in peaksh:
            axes[0].axhline(baseline, c="r", alpha=0.7)
        axes[0].axvline(waveform_x0, c="g")
        axes[0].axvline(waveform_x1, c="g")
        axes[1].plot(projection, np.arange(len(projection)))
        axes[1].invert_yaxis()
        plt.show()

    return (dic_tracks, peaksh, active_x0)


def clean_tracks(
    dic_tracks: dict[int, np.ndarray], TYPE: str, NOISE: bool | float, DEBUG: bool
) -> dict[int, np.ndarray]:
    """
    Remove printed annotations (lead labels) from binary track images.

    NOT WIRED INTO THE PIPELINE -- read this before calling it.
    ----------------------------------------------------------
    The lead name is printed inside the track, and on a dense layout the
    neighbouring lanes' labels bleed in too.  The extractors read one value
    per column and cannot tell a glyph from the trace, so a label drags the
    signal towards it wherever the two share a column.  Removing labels is
    therefore worth doing -- but this size test only holds when the leads do
    not overlap, and it is on overlapping pages that the labels appear.

    Labels are told apart from the waveform by size.  Dilating first merges
    each into a single blob: on a layout with room to spare the trace grows
    into chunks that span the track vertically, while a label stays small in
    both directions (measured: 19-65px wide, 20-52px tall on a ~273px lane).

    Where that breaks: when a QRS is taller than its lane it leaves the track
    and comes back, and the pieces stranded near the track edge are small
    isolated blobs -- the same size and in the same place as a label.  Enabling
    this on a real 12x1 page at 10 mm/mV deleted 0.3-9.6% of each lead's
    waveform, 97.7% of it at the lane edges, visibly flattening the R peaks.
    Synthetic pages at 5 mm/mV do not overlap and so do not show the damage;
    do not take a clean result there as evidence this is safe.

    Separating a label from a stranded QRS tip needs to know which lead the
    blob belongs to, which is the same trace-assignment problem noted in the
    module docstring of ``extraction_functions``.

    Parameters
    ----------
    dic_tracks: dictionary, dictionary of binary track images (modified in place)
    TYPE  : str, format of the image
    NOISE : bool, if the image is noised or not
    DEBUG : bool, show the image

    Returns
    -------
    dictionary: the same dictionary, with annotations blanked
    """
    for d in dic_tracks:
        track = dic_tracks[d]
        if track.ndim != 2 or track.size == 0:
            continue
        h, w = track.shape
        image_bin = track.astype("uint8")

        # Protect the calibration square: it lives at the start of the track.
        ref_width = max(10, int(REF_PULSE_WIDTH_FRACTION * w))
        # Merge each structure into one blob before measuring it, so a label's
        # separate strokes are judged together rather than one stroke at a time.
        k = max(3, int(TEXT_MERGE_FRACTION * h))
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        max_text = TEXT_MAX_SIZE_FRACTION * h

        # Dilate the image
        dilation = cv2.dilate(image_bin, rect_kernel, iterations=1)
        # Find contour by applying rectangle
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Blank every blob that is small on both axes and clear of the pulse
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if x < ref_width:
                continue
            if bw < max_text and bh < max_text:
                track[y : y + bh, x : x + bw] = 0

        # Plot the image and the associated masks
        if DEBUG:
            plt.figure(figsize=(20, 14))
            plt.imshow(track)
            plt.show()

    return dic_tracks


def sup_holes(signal: list | np.ndarray, TYPE: str) -> np.ndarray:
    """
    Fill the holes in the extracted signal

    Parameters
    ----------
    signal: array, contain the extracted signal
    TYPE: str, it can be :
            - "classic"
            - "heartcheck"
            - "duoek"

    Returns
    -------
    list: list of the extracted signal without hole
    """

    # if the signal is constant then we set the signal to 0
    signal = np.asarray(signal, dtype=float)
    if np.all(np.diff(signal) == 0):
        signal = np.zeros(len(signal))
        return signal

    end = -1
    # Treat both zeros and NaN as holes to interpolate
    hole_mask = (signal == 0) | np.isnan(signal)

    # If the first value is a hole, find the next valid point
    if hole_mask[0]:
        j = 1
        while j < len(signal) and hole_mask[j]:
            j += 1
        if j < len(signal):
            signal[0] = signal[j]
        else:
            signal[0] = len(signal) / 2  # fallback to midpoint

    # If the last value is a hole, find the previous valid point
    if hole_mask[-1]:
        j = 1
        while j < len(signal) and hole_mask[-j]:
            j += 1
        if j < len(signal):
            signal[-1] = signal[-j]
        else:
            signal[-1] = signal[0]

    # Recompute mask after fixing endpoints
    hole_mask = (signal == 0) | np.isnan(signal)

    # Interpolate interior holes using nearest valid neighbours
    if np.any(hole_mask):
        valid_idx = np.where(~hole_mask)[0]
        if len(valid_idx) > 0:
            signal[hole_mask] = np.interp(np.where(hole_mask)[0], valid_idx, signal[valid_idx])

    return signal[:end]


def lead_extraction(
    dic_tracks: dict[int, np.ndarray], extraction_method: str, TYPE: str, NOISE: bool | float, DEBUG: bool = False
) -> dict[str, np.ndarray]:
    """
    Extract the digital information from images

    Parameters
    ----------
    dic_tracks: dictionary, dictionary of track images
    extraction_method: str, one of "trace", "fragmented", "lazy", or "full"
    TYPE  : str, format of the image
    NOISE : bool, if the image is noised or not
    DEBUG : bool, show the image

    Returns
    -------
    dictionary: dictionary of digital tracks

    ``fragmented`` (also accepted as ``trace``) uses the geometry stored on an
    :class:`ECGTrack` to resolve overlapping signals by global continuity.
    """

    # Digital tracks dictionary
    dic_extracted_tracks = {}
    dic_image_bin = {}
    dic_extracted_track_not_scale = {}
    for d in dic_tracks:
        # Kardia Files are already binarize
        image_bin = dic_tracks[d]
        # Plot the binarized image
        if DEBUG:
            plt.imshow(image_bin)
            plt.show()
            plt.imshow(image_bin)

        has_geometry = getattr(image_bin, "baseline_row", None) is not None
        waveform_x0 = int(getattr(image_bin, "waveform_x0", 0) or 0)
        waveform_x1 = int(getattr(image_bin, "waveform_x1", image_bin.shape[1]) or image_bin.shape[1])
        waveform_x0 = max(0, min(waveform_x0, image_bin.shape[1] - 1))
        waveform_x1 = max(waveform_x0 + 1, min(waveform_x1, image_bin.shape[1]))
        extraction_image = image_bin[:, waveform_x0:waveform_x1] if has_geometry else image_bin
        if has_geometry:
            # ndarray slicing preserves ECGTrack metadata; only its page x
            # origin changes.
            extraction_image.origin_x = int(getattr(image_bin, "origin_x", 0) or 0) + waveform_x0

        if extraction_method == "lazy":
            extraction = lazy_extraction(extraction_image)
        elif extraction_method == "full":
            extraction = full_extraction(extraction_image)
        elif extraction_method in ("fragmented", "trace", "continuous"):
            extraction = fragmented_extraction(extraction_image)
        else:
            raise ValueError(
                "Unknown extraction method %r; expected lazy, full, fragmented, or trace."
                % extraction_method
            )

        if has_geometry and TYPE.lower() == "classic":
            # The waveform bounds exclude a pulse on either page edge.  Scale
            # the true 10-second trace directly, then add the legacy 140-sample
            # prefix expected by lead_cutting.  Its amplitude calibration comes
            # from the measured square metadata, not from this synthetic prefix.
            signal = np.asarray(extraction, dtype=float)
            if len(signal) == 1:
                signal_scale_waveform = np.full(SIGNAL_LENGTH_STANDARD, signal[0])
            else:
                source_x = np.arange(len(signal), dtype=float)
                target_x = np.linspace(0, len(signal) - 1, SIGNAL_LENGTH_STANDARD)
                signal_scale_waveform = np.interp(target_x, source_x, signal)
            prefix_length = SIGNAL_LENGTH_CLASSIC - SIGNAL_LENGTH_STANDARD
            baseline = float(getattr(image_bin, "baseline_row"))
            signal_scale = np.concatenate([np.full(prefix_length, baseline), signal_scale_waveform])
            raw_signal = ExtractedSignal(
                signal,
                origin_x=int(getattr(image_bin, "origin_x", 0) or 0) + waveform_x0,
                origin_y=int(getattr(image_bin, "origin_y", 0) or 0),
            )
        else:
            # Removing the holes in legacy/non-classic signals.
            signal = sup_holes(extraction, TYPE)

            # Scale the signal in time: every track is 10 seconds at 500 Hz,
            # plus a format-specific reference pulse where applicable.
            x = [i for i in range(len(signal))]
            y = signal
            if TYPE.lower() == "classic":
                new_x = [i for i in np.arange(0, len(signal), len(signal) / SIGNAL_LENGTH_CLASSIC)]
            elif TYPE.lower() == "kardia":
                new_x = [i for i in np.arange(0, len(signal), len(signal) / SIGNAL_LENGTH_KARDIA)]
            else:
                new_x = [i for i in np.arange(0, len(signal), len(signal) / SIGNAL_LENGTH_STANDARD)]
            signal_scale = np.interp(new_x, x, y)
            raw_signal = signal

        dic_extracted_track_not_scale[d] = raw_signal
        dic_extracted_tracks[d] = signal_scale
        dic_image_bin[d] = image_bin

        # Plot the signal before its scale
        if DEBUG:
            plt.plot(signal, c="r")
            plt.show()

    return (dic_extracted_tracks, dic_image_bin, dic_extracted_track_not_scale)


def lead_cutting(
    dic_tracks: dict[int, np.ndarray],
    DPI: int,
    TYPE: str,
    FORMAT: str,
    page: int,
    NOISE: bool | float,
    DEBUG: bool,
    dic_image_bin: dict[int, np.ndarray] | None = None,
) -> dict[str, np.ndarray] | np.ndarray:
    """
    Cut each tracks into leads

    Parameters
    ----------
    dic_tracks: dictionary, dictionary of track images
    DPI   : int, resolution
    TYPE  : str, format of the image
    NOISE : bool, if the image is noised or not
    DEBUG : bool, show the image
    dic_image_bin : dict, optional binary track images for calibration

    Returns
    -------
    dictionary: dictionary of leads
    """
    # Dictionary with reference pulse for each tracks
    dic_ref_pulse = {}
    # Dictionary with the lead
    dic_leads = {}
    LEAD_LENGTH = 0
    LEAD_NUMBER = 1
    dic_association = {0: "II"}
    # If the it is a classical format
    if TYPE.lower() == "classic" or (TYPE.lower() == "kardia" and FORMAT == "multilead"):
        if TYPE.lower() != "classic":
            if page == 0:
                # the reference pulse lasts 0.28sec
                LENGTH_PULSE = REF_PULSE_KARDIA
            else:
                LENGTH_PULSE = 0

        dic_time = {}
        # The disposition of the ECG is 4x4
        if len(dic_tracks) == 4:
            # leads lasts 2.5sec if there are 4 tracks
            LEAD_NUMBER = 4
            dic_association = {
                0: ["I", "AVR", "V1", "V4"],
                1: ["II", "AVL", "V2", "V5"],
                2: ["III", "AVF", "V3", "V6"],
                3: ["II"],
            }
            dic_time = LEAD_TIME_3X4
        # The disposition of the ECG is 6x2
        elif len(dic_tracks) == 6:
            # leads last 5sec if there are 6 tracks
            LEAD_NUMBER = 2
            dic_association = {
                0: ["I", "V1"],
                1: ["II", "V2"],
                2: ["III", "V3"],
                3: ["AVR", "V4"],
                4: ["AVL", "V5"],
                5: ["AVF", "V6"],
            }
            dic_time = LEAD_TIME_6X2
        # The disposition of the ECG is 12x1
        elif len(dic_tracks) == 12:
            # each track holds one whole 10sec lead if there are 12 tracks
            LEAD_NUMBER = 1
            dic_association = {i: [name] for i, name in enumerate(LEAD_ORDER_DEFAULT)}
            dic_time = LEAD_TIME_12X1
        elif TYPE.lower() == "classic":
            # Kardia multilead also reaches this block but brings its own track
            # count and lead mapping (set just below), so only classic pages are
            # required to match a known layout.  Without this, an unexpected
            # count would fall through with dic_association = {0: "II"} and
            # silently character-index that string into bogus lead names.
            classic_layout(len(dic_tracks))  # no classic layout has this many tracks: raises

        if TYPE.lower() == "kardia" and FORMAT.lower() == "multilead":
            # leads last 10sec in kardia

            dic_association = {
                0: "I",
                1: "II",
                2: "III",
                3: "AVR",
                4: "AVL",
                5: "AVF",
            }

        # Pre-compute calibration factors for all tracks.
        # When binary track images are available (dic_image_bin), measure the
        # calibration square height directly from the image for higher accuracy.
        # Then cross-validate across tracks to replace outlier values.
        _calib = {}  # {track_idx: (pixel_zero, f, LENGTH_PULSE)}
        if TYPE.lower() != "kardia":
            for t in dic_tracks:
                _lp = REF_PULSE_CLASSIC
                if len(dic_tracks) in CLASSIC_LAYOUTS:
                    _lp = len(dic_tracks[t]) - SIGNAL_LENGTH_STANDARD

                # Prefer binary-image calibration (measures actual square height).
                # The track above is passed in because the square can be taller
                # than the headroom a dense layout leaves, and is then cut off
                # by the track boundary (see _calibrate_from_binary).
                if dic_image_bin is not None and t in dic_image_bin:
                    _pz, _f = _calibrate_from_binary(dic_image_bin[t], DPI, dic_image_bin.get(t - 1))
                else:
                    _ref = dic_tracks[t][:_lp]
                    _pz = float(max(_ref))
                    _p1 = float(min(_ref))
                    _f = _pz - _p1
                _calib[t] = (_pz, max(_f, 0.001), _lp)
            # Cross-validate against the median.  Every lead on a page is
            # printed at the one gain the machine was set to, so the squares
            # must all measure the same height -- a lead that disagrees has
            # had its square mismeasured, not a different gain.  Only
            # measurement noise (a pixel or two on ~200) is legitimate, so the
            # band is tight: it has to catch cases like the top lead, whose
            # square is cut by the page header with no track above it to
            # restore from, and which lands ~0.54x of the true height.
            if len(_calib) > 1:
                all_f = [v[1] for v in _calib.values()]
                median_f = float(np.median(all_f))
                for t in _calib:
                    pz, f_val, lp = _calib[t]
                    if not (CALIB_AGREEMENT_MIN * median_f <= f_val <= CALIB_AGREEMENT_MAX * median_f):
                        logger.info("Track %d: calibration f=%.1f replaced by median %.1f", t, f_val, median_f)
                        _calib[t] = (pz, median_f, lp)

        # Plot each tracks
        for t in dic_tracks:
            if DEBUG:
                LENGTH_PULSE = 140
                logger.debug("Track: %s", t)
                plt.figure(figsize=(20, 14))
                plt.plot(dic_tracks[t])
                plt.axvline(LENGTH_PULSE, c="r")

            if TYPE.lower() != "kardia":
                pixel_zero, f, LENGTH_PULSE = _calib[t]
                dic_ref_pulse[t] = dic_tracks[t][:LENGTH_PULSE]

                # Define the beggining of lead part
                LEAD_LENGTH = int(len(dic_tracks[t][LENGTH_PULSE:]) / LEAD_NUMBER)
                length = LENGTH_PULSE
                # length = LENGTH_PULSE
                # Define the lead position
                it = 0

                # special case on the disposition 4x4 the last track containe 10sec of the lead II
                if len(dic_tracks) == 4 and t == 3:
                    dic_leads["IIc"] = (
                        (pixel_zero - dic_tracks[t][LENGTH_PULSE : 4 * LEAD_LENGTH + LENGTH_PULSE]) / f
                    ) * AMPLITUDE_SCALE_UV
                    if DEBUG:
                        plt.show()

                # extract each lead from the tracks
                elif LEAD_LENGTH != 0:
                    while length < len(dic_tracks[t]):
                        try:
                            dic_leads[dic_association[t][it]] = (
                                (pixel_zero - dic_tracks[t][length : length + LEAD_LENGTH]) / f
                            ) * AMPLITUDE_SCALE_UV  # We fill the leads dictionnary with the name of the lead and the image of it
                            length += int(len(dic_tracks[t][LENGTH_PULSE:]) / LEAD_NUMBER)
                            it += 1
                            if DEBUG:
                                plt.axvline(length, c="r")
                        except Exception:
                            length += int(len(dic_tracks[t][LENGTH_PULSE:]) / LEAD_NUMBER)
                else:
                    return 0
                if DEBUG:
                    plt.show()

            else:
                if page == 0:
                    ref_pulse = dic_tracks[t][:LENGTH_PULSE]
                    # Pixel of amplitude 0mV
                    pixel_zero = max(ref_pulse)
                    # Pixel of amplitude 1mV
                    pixel_one = min(ref_pulse)
                    # Define the factor
                    f = pixel_zero - pixel_one
                    if f == 0:
                        f = 1
                    # Define the beggining of lead part
                    length = LENGTH_PULSE
                    dic_leads["ref"] = [pixel_zero, f]

                    # Scale the signal in amplitude
                    dic_leads[dic_association[t]] = ((pixel_zero - dic_tracks[t][length:]) / f) * AMPLITUDE_SCALE_UV

                else:
                    length = 0
                    dic_leads[dic_association[t]] = dic_tracks[t][length:]

        try:
            for k in dic_leads:
                zero_vector = np.zeros(SIGNAL_LENGTH_STANDARD)
                lead_data = dic_leads[k]
                t_start, t_end = dic_time[k]
                expected_len = t_end - t_start
                if len(lead_data) > expected_len:
                    lead_data = lead_data[:expected_len]
                elif len(lead_data) < expected_len:
                    padded = np.zeros(expected_len)
                    padded[: len(lead_data)] = lead_data
                    lead_data = padded
                zero_vector[t_start:t_end] = lead_data
                dic_leads[k] = zero_vector
        except Exception as e:
            logger.warning("Lead placement failed: %s", e)
        return dic_leads

    # If the format is not classic
    else:
        if TYPE.lower() == "apple":
            LENGTH_PULSE = REF_PULSE_APPLE
        elif TYPE.lower() == "kardia":
            LENGTH_PULSE = REF_PULSE_KARDIA
        else:
            LENGTH_PULSE = REF_PULSE_GENERIC

        for t in dic_tracks:
            if t == 0:
                # Plot each tracks
                if DEBUG:
                    plt.figure(figsize=(20, 14))
                    plt.plot(dic_tracks[t])
                    plt.axvline(LENGTH_PULSE, c="r")
                    plt.show()

                # Isolate and calibrate the reference pulse
                dic_ref_pulse = dic_tracks[t][:LENGTH_PULSE]
                pixel_zero, f = _calibrate_ref_pulse(dic_ref_pulse, DPI)

                # Separate the signal from the reference pulse
                all_signal = dic_tracks[t][LENGTH_PULSE:]

            # Concatane the signal if it is on more than one track
            else:
                dist = np.mean(all_signal) - np.mean(dic_tracks[t])
                all_signal = np.concatenate((all_signal, dic_tracks[t] + dist), axis=0)

            # Plot the different pixel
            if DEBUG:
                logger.debug("0: %s", pixel_zero)
                logger.debug("1: %s", pixel_one)
                logger.debug("1st pixel: %s", all_signal[0])

        # Scale the signal in amplitude
        new_signal = ((pixel_zero - all_signal) / f) * AMPLITUDE_SCALE_UV

        return new_signal
