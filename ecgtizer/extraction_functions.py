"""Waveform extraction algorithms for binarized ECG track images.

Three strategies with different speed/accuracy trade-offs:

* **lazy** -- fast, noise-tolerant, but smooths peaks.
* **full** -- fast, high fidelity, but may include annotation artifacts.
* **fragmented** -- continuity-optimised tracing through overlapping lanes.

The fragmented method operates on an overlapping, baseline-aware corridor.
It uses dynamic programming over all column fragments, with costs for trace
discontinuity, curvature, missing ink, and dwelling on a neighbouring lead's
baseline.  This preserves tall QRS complexes while preventing the permanent
lead switches caused by a greedy left-to-right walk.
"""

from __future__ import annotations

import numpy as np


MAX_COLUMN_CANDIDATES = 24
MISSING_INK_COST = 8.0
BASELINE_COST = 0.04
NEIGHBOUR_BASELINE_COST = 6.0
NEIGHBOUR_BASELINE_WIDTH = 0.18
NEIGHBOUR_ESCAPE_SLOPE = 0.04
STEP_COST = 0.02
CURVATURE_COST = 0.2


def lazy_extraction(image_bin: np.ndarray) -> list[int]:
    """Extract a waveform by following the nearest lit pixel from an anchor.

    Fast and noise-tolerant. Starts from the average lit-pixel position
    in the first column and walks column-by-column, always jumping to the
    closest lit pixel within a 1000-pixel window.

    Parameters
    ----------
    image_bin : numpy.ndarray
        Binarized track image (255 = signal, 0 = background).

    Returns
    -------
    list[int]
        Vertical pixel positions representing the extracted waveform.
    """
    # Find anchor point: average of all lit pixels in the first column
    first_col_lit = np.where(image_bin[:, 0] == 255)[0]
    if len(first_col_lit) > 0:
        anchor = int(np.mean(first_col_lit))
    else:
        anchor = image_bin.shape[0] // 2
    signal = [anchor]

    # We then go through the image column by column, looking for the lit pixel closest to the anchor pixel.
    for i in range(1, len(image_bin[0])):
        # If we can stay at the same level as the anchor pixel, we do so
        if image_bin[anchor, i] == 255:
            signal.append(anchor)
        else:
            # Otherwise we look for the nearest lit pixel at the top and bottom, and as soon as we find one we stop and store it.
            # We search within a window of 1000 pixels to avoid searching too far.
            try:
                for j in range(1000):
                    if image_bin[anchor + j, i] == 255:
                        signal.append(anchor + j)
                        anchor = anchor + j
                        break
                    elif image_bin[anchor - j, i] == 255:
                        signal.append(anchor - j)
                        anchor = anchor - j
                        break
            except IndexError:
                signal.append(anchor)
    return signal


def full_extraction(image_bin: np.ndarray) -> np.ndarray:
    """Extract a waveform by averaging all lit-pixel positions per column.

    Fast with high fidelity. Computes the mean row position of all lit
    pixels in each column. May include annotation artifacts if text
    overlaps the signal region.

    Parameters
    ----------
    image_bin : numpy.ndarray
        Binarized track image (255 = signal, 0 = background).

    Returns
    -------
    numpy.ndarray
        Mean vertical positions per column.
    """
    # Vectorized: mean row position of lit pixels per column
    mask = image_bin == 255
    row_indices = np.arange(image_bin.shape[0], dtype=float)
    # Weighted sum of row indices where mask is True, per column
    weighted_sum = np.dot(row_indices, mask)  # shape: (width,)
    count = mask.sum(axis=0).astype(float)  # shape: (width,)
    # Avoid division by zero: where no lit pixels, return 0.0
    extraction = np.divide(weighted_sum, count, out=np.zeros(image_bin.shape[1]), where=count > 0)
    return extraction


def fragmented_extraction(image_bin: np.ndarray) -> list[float]:
    """Extract the globally most plausible continuous ECG trace.

    Consecutive lit pixels form one candidate fragment in each column.  A
    second-order dynamic program selects a path through those candidates.  Its
    objective combines ink support, distance from the target lead's baseline,
    a penalty for lingering on neighbouring baselines, step size, and local
    curvature.  A virtual baseline candidate lets the path cross genuinely
    missing columns; those points are interpolated after backtracking.

    Parameters
    ----------
    image_bin : numpy.ndarray
        Binarized track image (255 = signal, 0 = background).

    Returns
    -------
    list[float]
        Vertical pixel positions, one per image column.
    """
    h, w = image_bin.shape
    if w == 0:
        return []

    target = float(getattr(image_bin, "baseline_row", None) or h / 2.0)
    spacing = float(getattr(image_bin, "spacing", None) or max(h / 2.0, 1.0))
    neighbours = getattr(image_bin, "neighbour_baselines", None)
    neighbours = np.asarray(neighbours if neighbours is not None else [], dtype=float)

    candidates: list[np.ndarray] = []
    is_ink: list[np.ndarray] = []
    for column_index in range(w):
        positions = np.flatnonzero(image_bin[:, column_index] > 0)
        if len(positions):
            groups = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
            centres = np.asarray([group.mean() for group in groups], dtype=float)
            lengths = np.asarray([len(group) for group in groups], dtype=float)
            if len(centres) > MAX_COLUMN_CANDIDATES:
                # Retain both locally plausible fragments and long vertical
                # QRS fragments.  This bounds cubic DP work on noisy scans
                # without imposing a narrow amplitude window.
                near = np.argsort(np.abs(centres - target))[: MAX_COLUMN_CANDIDATES // 2]
                long = np.argsort(lengths)[-(MAX_COLUMN_CANDIDATES - len(near)) :]
                selected = np.unique(np.concatenate([near, long]))
                if len(selected) > MAX_COLUMN_CANDIDATES:
                    selected = selected[:MAX_COLUMN_CANDIDATES]
                centres = centres[selected]
        else:
            centres = np.empty(0, dtype=float)

        # The final state is virtual and represents missing ink.
        candidates.append(np.concatenate([centres, [target]]))
        is_ink.append(np.concatenate([np.ones(len(centres), dtype=bool), [False]]))

    def unary(values: np.ndarray, actual_ink: np.ndarray) -> np.ndarray:
        cost = BASELINE_COST * ((values - target) / spacing) ** 2
        cost = cost + (~actual_ink) * MISSING_INK_COST
        return cost

    def neighbour_dwell(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Penalise following another lane, but allow a QRS to cross it.

        A positional penalty alone creates an artificial barrier at every
        neighbouring baseline.  That clips high-amplitude QRS complexes --
        precisely the case where lanes overlap.  A trace belonging to another
        lane dwells near its baseline, whereas a legitimate excursion from the
        target lane moves through it with appreciable slope.  Modulating by
        the local step preserves that distinction without an amplitude limit.
        """
        if not len(neighbours):
            return np.zeros(np.broadcast_shapes(previous.shape, current.shape), dtype=float)
        neighbour_distance = np.min(
            np.abs(current[..., None] - neighbours), axis=-1
        ) / spacing
        speed = np.abs(current - previous) / spacing
        near_neighbour = np.exp(
            -0.5 * (neighbour_distance / NEIGHBOUR_BASELINE_WIDTH) ** 2
        )
        dwelling = np.exp(-0.5 * (speed / NEIGHBOUR_ESCAPE_SLOPE) ** 2)
        return NEIGHBOUR_BASELINE_COST * near_neighbour * dwelling

    if w == 1:
        return [float(candidates[0][np.argmin(unary(candidates[0], is_ink[0]))])]

    # State (p, c) stores the best path ending at candidates p and c in the
    # previous two columns.  Keeping slope in the state is what lets a trace
    # pass through a crossing instead of taking whichever outgoing branch is
    # nearest to the current pixel.
    first = candidates[0]
    second = candidates[1]
    cost = (
        unary(first, is_ink[0])[:, None]
        + unary(second, is_ink[1])[None, :]
        + STEP_COST * np.abs(second[None, :] - first[:, None])
        + neighbour_dwell(first[:, None], second[None, :])
    )
    back_pointers: list[np.ndarray] = []

    for column_index in range(2, w):
        previous_previous = candidates[column_index - 2][:, None, None]
        previous = candidates[column_index - 1][None, :, None]
        current = candidates[column_index][None, None, :]
        transition = STEP_COST * np.abs(current - previous) + CURVATURE_COST * np.abs(
            current - 2 * previous + previous_previous
        )
        transition += neighbour_dwell(previous, current)
        alternatives = cost[:, :, None] + transition
        best_previous = np.argmin(alternatives, axis=0)
        previous_indices = np.arange(len(candidates[column_index - 1]))[:, None]
        current_indices = np.arange(len(candidates[column_index]))[None, :]
        cost = alternatives[best_previous, previous_indices, current_indices]
        cost += unary(candidates[column_index], is_ink[column_index])[None, :]
        back_pointers.append(best_previous.astype(np.int16))

    previous_index, current_index = np.unravel_index(np.argmin(cost), cost.shape)
    path = np.empty(w, dtype=float)
    selected_ink = np.ones(w, dtype=bool)
    path[-2] = candidates[-2][previous_index]
    path[-1] = candidates[-1][current_index]
    selected_ink[-2] = is_ink[-2][previous_index]
    selected_ink[-1] = is_ink[-1][current_index]
    for column_index in range(w - 1, 1, -1):
        earlier_index = int(back_pointers[column_index - 2][previous_index, current_index])
        current_index = previous_index
        previous_index = earlier_index
        path[column_index - 2] = candidates[column_index - 2][previous_index]
        selected_ink[column_index - 2] = is_ink[column_index - 2][previous_index]

    missing = ~selected_ink
    if np.any(missing):
        present = np.flatnonzero(~missing)
        if len(present):
            path[missing] = np.interp(np.flatnonzero(missing), present, path[present])
        else:
            path[:] = target
    return path.tolist()
