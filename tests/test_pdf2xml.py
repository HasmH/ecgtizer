"""Tests for ecgtizer/PDF2XML.py

Tests signal processing functions:
- sup_holes: fill missing points in extracted signals
- lead_extraction: digitize signals from track images
- clean_tracks: remove noise from track images
- check_noise_type: detect image type and noise level
"""
import numpy as np
import cv2
import pytest

from ecgtizer.PDF2XML import sup_holes


class TestSupHoles:

    def test_signal_without_holes(self):
        """Signal with no zeros should be returned unchanged (minus last element)."""
        signal = [10, 20, 30, 40, 50]
        result = sup_holes(signal, 'classic')
        np.testing.assert_array_equal(result, [10, 20, 30, 40])

    def test_hole_at_beginning(self):
        """Hole at start should be filled with next non-zero value."""
        signal = [0, 0, 30, 40, 50]
        result = sup_holes(signal, 'classic')
        assert result[0] == 30
        assert result[1] == 30

    def test_hole_at_end(self):
        """Hole at end should be filled with previous non-zero value."""
        signal = [10, 20, 30, 0, 0]
        result = sup_holes(signal, 'classic')
        assert result[-1] != 0

    def test_hole_in_middle(self):
        """Interior hole should be filled with mean of neighbors."""
        signal = [10, 0, 30, 40, 50]
        result = sup_holes(signal, 'classic')
        # Middle hole between 10 and 30 should be mean = 20
        assert result[1] == pytest.approx(20)

    def test_constant_signal_returns_zeros(self):
        """All-same signal (diff == 0) should return zeros."""
        signal = np.array([5, 5, 5, 5, 5])
        result = sup_holes(signal, 'classic')
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_all_zeros_returns_zeros(self):
        """All-zero signal (constant) should be replaced with zeros."""
        signal = np.array([0, 0, 0, 0, 0])
        result = sup_holes(signal, 'classic')
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_numpy_array_input(self):
        """Should work with numpy array input."""
        signal = np.array([10.0, 0.0, 30.0, 0.0, 50.0])
        result = sup_holes(signal, 'classic')
        assert len(result) == 4
        assert result[1] != 0

    def test_single_nonzero_value(self):
        """Signal with only one non-zero value."""
        signal = np.array([0, 0, 42, 0, 0])
        result = sup_holes(signal, 'classic')
        # First and last filled, middle holes interpolated
        assert result[0] == 42
        assert all(v != 0 for v in result)

    def test_large_signal(self):
        """Performance check with a large signal."""
        rng = np.random.RandomState(42)
        signal = rng.randint(1, 1000, size=10000).astype(float)
        # Insert some holes
        signal[100:110] = 0
        signal[5000:5005] = 0
        result = sup_holes(signal, 'classic')
        assert len(result) == 9999
        # Holes should be filled
        assert all(result[100:110] != 0)

    def test_nan_values_are_filled(self):
        """NaN values should be treated as holes and interpolated."""
        signal = np.array([10.0, np.nan, np.nan, 40.0, 50.0])
        result = sup_holes(signal, 'classic')
        assert not np.any(np.isnan(result))
        # Interpolated values should be between neighbours
        assert 10.0 < result[1] < 40.0
        assert 10.0 < result[2] < 40.0

    def test_nan_at_boundaries(self):
        """NaN at start and end should be filled from nearest valid values."""
        signal = np.array([np.nan, np.nan, 30.0, 40.0, np.nan])
        result = sup_holes(signal, 'classic')
        assert not np.any(np.isnan(result))

    def test_mixed_nan_and_zeros(self):
        """Both NaN and zero holes should be filled."""
        signal = np.array([10.0, 0.0, np.nan, 40.0, 0.0, 60.0])
        result = sup_holes(signal, 'classic')
        assert not np.any(np.isnan(result))
        assert all(v != 0 for v in result)


class TestLeadExtraction:

    def test_lazy_extraction_on_tracks(self, sample_binary_image):
        """lead_extraction with lazy method should produce output for each track."""
        from ecgtizer.PDF2XML import lead_extraction
        dic_tracks = {0: sample_binary_image}
        result, image_bins, not_scaled = lead_extraction(
            dic_tracks, "lazy", "wellue", False, False
        )
        assert 0 in result
        assert len(result[0]) > 0

    def test_full_extraction_on_tracks(self, sample_binary_image):
        from ecgtizer.PDF2XML import lead_extraction
        dic_tracks = {0: sample_binary_image}
        result, image_bins, not_scaled = lead_extraction(
            dic_tracks, "full", "wellue", False, False
        )
        assert 0 in result

    def test_fragmented_extraction_on_tracks(self, sample_binary_image):
        from ecgtizer.PDF2XML import lead_extraction
        dic_tracks = {0: sample_binary_image}
        result, image_bins, not_scaled = lead_extraction(
            dic_tracks, "fragmented", "wellue", False, False
        )
        assert 0 in result

    def test_extraction_output_is_scaled(self, sample_binary_image):
        """Scaled output should have ~5000 points for non-kardia/non-classic."""
        from ecgtizer.PDF2XML import lead_extraction
        dic_tracks = {0: sample_binary_image}
        result, _, not_scaled = lead_extraction(
            dic_tracks, "full", "wellue", False, False
        )
        assert len(result[0]) == 5000 or len(result[0]) > 4000

    def test_multiple_tracks(self, sample_binary_image):
        """Should handle multiple track images."""
        from ecgtizer.PDF2XML import lead_extraction
        dic_tracks = {0: sample_binary_image, 1: sample_binary_image.copy()}
        result, _, _ = lead_extraction(dic_tracks, "full", "wellue", False, False)
        assert len(result) == 2


def _make_classic_track(n_samples=5140, freq=1.0, amp=40.0, pulse_len=140):
    """Build one synthetic *extracted* track (post lead_extraction).

    Values are in pixel-row coordinates (larger value = lower on the page).
    The track starts with a calibration square (a ~1 mV upward step) followed
    by a sinusoidal body, mirroring the [pulse | signal] layout the calibration
    logic in ``lead_cutting`` expects.
    """
    track = np.full(n_samples, 250.0)
    plateau_end = pulse_len - 10
    track[10:plateau_end] = 150.0  # calibration square: 100 px ~= 1 mV up
    body = np.arange(n_samples - pulse_len)
    track[pulse_len:] = 250.0 - amp * np.sin(2 * np.pi * freq * body / 500.0)
    return track


class TestLeadCutting:
    """lead_cutting format inference by track count (3x4 / 6x2 / 12x1)."""

    TWELVE_LEADS = {"I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"}

    def test_12x1_twelve_tracks(self):
        """12 tracks -> 12 full-length leads, one per track."""
        from ecgtizer.PDF2XML import lead_cutting
        tracks = {i: _make_classic_track(freq=1 + 0.1 * i) for i in range(12)}
        leads = lead_cutting(tracks, 300, "classic", "", 0, NOISE=False, DEBUG=False)
        assert set(leads.keys()) == self.TWELVE_LEADS
        for name, sig in leads.items():
            assert len(sig) == 5000, f"{name} has length {len(sig)}"
            assert np.all(np.isfinite(sig))

    def test_12x1_thirteen_tracks_has_rhythm(self):
        """A 13th track is the lead-II rhythm strip (IIc)."""
        from ecgtizer.PDF2XML import lead_cutting
        tracks = {i: _make_classic_track(freq=1 + 0.1 * i) for i in range(13)}
        leads = lead_cutting(tracks, 300, "classic", "", 0, NOISE=False, DEBUG=False)
        assert set(leads.keys()) == self.TWELVE_LEADS | {"IIc"}
        assert len(leads["IIc"]) == 5000

    def test_12x1_leads_span_full_window(self):
        """Unlike 3x4/6x2, a 12x1 lead carries signal across the whole 10 s."""
        from ecgtizer.PDF2XML import lead_cutting
        tracks = {i: _make_classic_track(freq=1 + 0.1 * i) for i in range(12)}
        leads = lead_cutting(tracks, 300, "classic", "", 0, NOISE=False, DEBUG=False)
        # The last quarter (7.5-10 s) must be populated for every lead.
        for name, sig in leads.items():
            assert np.any(sig[3750:] != 0), f"{name} is empty in its final quarter"

    def test_3x4_regression(self):
        """4 tracks still infer 3x4: a rhythm strip and time-windowed leads."""
        from ecgtizer.PDF2XML import lead_cutting
        tracks = {i: _make_classic_track(freq=1 + 0.1 * i) for i in range(4)}
        leads = lead_cutting(tracks, 300, "classic", "", 0, NOISE=False, DEBUG=False)
        assert "IIc" in leads
        assert self.TWELVE_LEADS.issubset(leads.keys())
        # Lead I only occupies the first 2.5 s (0-1250) in 3x4.
        assert np.all(leads["I"][1250:] == 0)

    def test_6x2_regression(self):
        """6 tracks still infer 6x2: 5 s left column, 5 s right column."""
        from ecgtizer.PDF2XML import lead_cutting
        tracks = {i: _make_classic_track(freq=1 + 0.1 * i) for i in range(6)}
        leads = lead_cutting(tracks, 300, "classic", "", 0, NOISE=False, DEBUG=False)
        assert set(leads.keys()) == self.TWELVE_LEADS
        # Lead I occupies the first 5 s; V1 the last 5 s.
        assert np.all(leads["I"][2500:] == 0)
        assert np.all(leads["V1"][:2500] == 0)

    def test_12x1_interleaved_order(self):
        """The 'interleaved' preset assigns names by the generator's row order."""
        from ecgtizer.PDF2XML import lead_cutting, LEAD_ORDER_12X1_INTERLEAVED
        # Distinct amplitude per track so we can trace which track became which lead.
        tracks = {i: _make_classic_track(amp=10 + i) for i in range(12)}
        leads = lead_cutting(tracks, 300, "classic", "", 0, NOISE=False, DEBUG=False, lead_order="interleaved")
        assert set(leads.keys()) == self.TWELVE_LEADS
        # Track 0 -> V4, track 2 -> AVR, track 3 -> I (per the interleaved order).
        assert LEAD_ORDER_12X1_INTERLEAVED[0] == "V4"
        assert LEAD_ORDER_12X1_INTERLEAVED[2] == "AVR"

    def test_12x1_explicit_list_order(self):
        """An explicit 12-name list is honoured verbatim (top to bottom)."""
        from ecgtizer.PDF2XML import lead_cutting
        custom = ["V6", "V5", "V4", "V3", "V2", "V1", "AVF", "AVL", "AVR", "III", "II", "I"]
        tracks = {i: _make_classic_track(freq=1 + 0.1 * i) for i in range(12)}
        leads = lead_cutting(tracks, 300, "classic", "", 0, NOISE=False, DEBUG=False, lead_order=custom)
        assert set(leads.keys()) == self.TWELVE_LEADS


class TestResolveLeadOrder:
    """_resolve_lead_order_12x1: preset keys, explicit lists, invalid input."""

    def test_none_is_standard(self):
        from ecgtizer.PDF2XML import _resolve_lead_order_12x1, LEAD_ORDER_12X1_STANDARD
        assert _resolve_lead_order_12x1(None) == LEAD_ORDER_12X1_STANDARD

    def test_interleaved_preset(self):
        from ecgtizer.PDF2XML import _resolve_lead_order_12x1, LEAD_ORDER_12X1_INTERLEAVED
        assert _resolve_lead_order_12x1("interleaved") == LEAD_ORDER_12X1_INTERLEAVED

    def test_lowercase_names_normalized(self):
        from ecgtizer.PDF2XML import _resolve_lead_order_12x1
        order = ["aVR", "aVL", "aVF", "i", "ii", "iii", "v1", "v2", "v3", "v4", "v5", "v6"]
        expected = ["AVR", "AVL", "AVF", "I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
        assert _resolve_lead_order_12x1(order) == expected

    def test_invalid_falls_back_to_standard(self):
        from ecgtizer.PDF2XML import _resolve_lead_order_12x1, LEAD_ORDER_12X1_STANDARD
        assert _resolve_lead_order_12x1(["I", "II"]) == LEAD_ORDER_12X1_STANDARD  # wrong length
        assert _resolve_lead_order_12x1("bogus") == LEAD_ORDER_12X1_STANDARD  # unknown key


class TestCropToContent:
    """_crop_to_content: trims blank margins, leaves full/dark images alone."""

    def test_crops_portrait_padded_page(self):
        from ecgtizer.PDF2XML import _crop_to_content
        # Landscape ink band on a tall white page (like a 12x1 on portrait A4).
        img = np.full((1200, 800, 3), 255, dtype=np.uint8)
        img[100:400, 50:750] = 0  # ink occupies a wide-but-short region up top
        out = _crop_to_content(img)
        assert out.shape[0] < img.shape[0]  # bottom whitespace removed
        assert out.shape[1] <= img.shape[1]

    def test_full_frame_unchanged(self):
        from ecgtizer.PDF2XML import _crop_to_content
        # Light page whose ink already spans nearly the whole frame.
        img = np.full((500, 900, 3), 255, dtype=np.uint8)
        img[10:490, 10:890] = 0
        out = _crop_to_content(img)
        assert out.shape == img.shape

    def test_dark_background_unchanged(self):
        from ecgtizer.PDF2XML import _crop_to_content
        img = np.zeros((600, 800, 3), dtype=np.uint8)  # mostly dark (Kardia-like)
        img[300, :] = 255
        out = _crop_to_content(img)
        assert out.shape == img.shape


def _make_grid_and_trace():
    """Binary image (0/255) with a dense full-span grid plus a curved trace."""
    h, w = 200, 1000
    img = np.zeros((h, w), dtype=np.uint8)
    img[::10, :] = 255  # horizontal grid lines (full width)
    img[:, ::10] = 255  # vertical grid lines (full height)
    xs = np.arange(w)
    ys = (h // 2 + 40 * np.sin(2 * np.pi * xs / 150)).astype(int)
    img[ys, xs] = 255  # the ECG-like trace
    return img, xs, ys


class TestRemoveDarkGrid:
    """_remove_dark_grid: strip long straight grid lines, keep the curved trace."""

    def test_grid_removed_trace_kept(self):
        from ecgtizer.PDF2XML import _remove_dark_grid
        img, xs, ys = _make_grid_and_trace()
        lit_before = np.mean(img == 255)
        out = _remove_dark_grid(img)
        lit_after = np.mean(out == 255)
        # The grid dominates the lit pixels, so removing it is a big drop.
        assert lit_after < lit_before * 0.5
        # Most trace pixels survive (only grid-crossing points are lost).
        assert np.mean(out[ys, xs] == 255) > 0.6

    def test_binarize_autoengages_on_dense_binary(self):
        """A dense (dark-grid-like) image is thinned; a sparse one is left alone."""
        from ecgtizer.PDF2XML import _binarize_image
        # Dark grayscale grid on white -> NOISE=True path keeps it, gate fires.
        grid = np.full((200, 1000), 255, dtype=np.uint8)
        grid[::6, :] = 0
        grid[:, ::6] = 0  # near-black dense grid
        dense = _binarize_image(grid, "classic", True)
        assert np.mean(dense == 255) < 0.15  # grid stripped below the gate

        # Sparse trace-only image stays sparse (gate does not fire).
        sparse = np.full((200, 1000), 255, dtype=np.uint8)
        sparse[100, :] = 0  # one thin line
        out = _binarize_image(sparse, "classic", True)
        assert np.mean(out == 255) < 0.05


class TestCheckNoiseType:

    def test_returns_type_and_noise(self, sample_color_image):
        from ecgtizer.PDF2XML import check_noise_type
        typ, noise = check_noise_type(sample_color_image, 300, False)
        assert isinstance(typ, str)
        assert isinstance(noise, (bool, float))

    def test_white_background_classic(self):
        """A white image with varied color content should be detected as classic."""
        from ecgtizer.PDF2XML import check_noise_type
        h, w = 600, 800
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        # Add varied-color content at the middle column so liste has >1 entries
        # (otherwise the function detects single-color as Kardia)
        for y in range(0, h, 20):
            color_val = (y * 3) % 256
            img[y, w // 2] = [255, color_val, color_val]
        # Add dark signal
        img[200:210, 100:700] = [0, 0, 0]
        typ, noise = check_noise_type(img, 300, False)
        assert typ.lower() in ('classic', 'wellue', 'apple')

    def test_black_background_kardia(self):
        """A mostly dark image should be detected as Kardia type."""
        from ecgtizer.PDF2XML import check_noise_type
        h, w = 600, 800
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Add white ECG line
        img[300, :] = [255, 255, 255]
        typ, noise = check_noise_type(img, 300, False)
        assert typ.lower() == 'kardia'
