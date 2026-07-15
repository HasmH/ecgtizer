"""Integration tests for ECGtizer end-to-end workflows.

Tests complete pipelines:
- CSV → ECG dict → PDF → (read back)
- ECG dict → XML → read back → analysis
- ECG dict → completion → 12-lead output
- Signal generation → extraction round-trip
"""
import numpy as np
import os
import pytest


class TestCSVToPDFPipeline:
    """Test: load CSV data → Write_PDF → verify file exists and is valid."""

    def test_csv_to_pdf_type2(self, sample_csv_ecg, tmp_output_dir):
        from ecgtizer.XML2PDF import Write_PDF
        outpath = os.path.join(tmp_output_dir, "pipeline_6x2.pdf")
        relpath = os.path.relpath(outpath)
        Write_PDF(sample_csv_ecg, relpath, type_of_pdf="type2")
        assert os.path.exists(outpath)
        assert os.path.getsize(outpath) > 5000

    def test_csv_to_pdf_type1(self, sample_csv_ecg, tmp_output_dir):
        from ecgtizer.XML2PDF import Write_PDF
        ecg = dict(sample_csv_ecg)
        ecg['IIc'] = ecg['II'].copy()
        ecg['ref'] = np.zeros(5000)
        outpath = os.path.join(tmp_output_dir, "pipeline_3x4.pdf")
        relpath = os.path.relpath(outpath)
        Write_PDF(ecg, relpath, type_of_pdf="type1", lead_IIc=ecg['IIc'])
        assert os.path.exists(outpath)


class TestXMLRoundTrip:
    """Test: ECG dict → write_xml → parse XML → verify structure."""

    def test_write_and_read_xml(self, sample_ecg_dict, tmp_output_dir):
        import xml.etree.ElementTree as ET
        from ecgtizer.PDF2XML_mod import write_xml

        outpath = os.path.join(tmp_output_dir, "roundtrip.xml")
        table = {
            'low_freq': '0.05', 'high_freq': '150', 'BPM': '75',
            'Inter PR (ms)': '160', 'Dur.QRS (ms)': '90',
            'QT (ms)': '380', 'QTc (ms)': '410',
            'Axe P': '60', 'Axe R': '30', 'Axe T': '45',
            'Moy RR (ms)': '800', 'QTcB (ms)': '405', 'QTcF (ms)': '400',
            'Rythme': 'Sinus', 'ECG': 'Normal',
            'Age': '45', 'sex': 'F', 'other_information': ''
        }
        write_xml(sample_ecg_dict, outpath, TYPE='classic', table=table)

        # Parse and verify
        tree = ET.parse(outpath)
        root = tree.getroot()
        assert root.tag.endswith('AnnotatedECG')

        # Find components with digit data
        xml_str = ET.tostring(root, encoding='unicode')
        assert 'digits' in xml_str
        assert 'MDC_ECG_LEAD' in xml_str

    def test_xml_contains_annotations(self, sample_ecg_dict, tmp_output_dir):
        import xml.etree.ElementTree as ET
        from ecgtizer.PDF2XML_mod import write_xml

        outpath = os.path.join(tmp_output_dir, "annotations.xml")
        table = {
            'low_freq': '0.05', 'high_freq': '150', 'BPM': '80',
            'Inter PR (ms)': '180', 'Dur.QRS (ms)': '100',
            'QT (ms)': '400', 'QTc (ms)': '430',
            'Axe P': '55', 'Axe R': '25', 'Axe T': '40',
            'Moy RR (ms)': '750', 'QTcB (ms)': '425', 'QTcF (ms)': '420',
            'Rythme': 'Sinus', 'ECG': 'Normal',
            'Age': '60', 'sex': 'M', 'other_information': 'test'
        }
        write_xml(sample_ecg_dict, outpath, TYPE='classic', table=table)
        tree = ET.parse(outpath)
        xml_str = ET.tostring(tree.getroot(), encoding='unicode')
        assert 'BPM' in xml_str
        assert '80' in xml_str


class TestCompletionPipeline:
    """Test: ECG dict → completion model → 12-lead completed output."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_torch(self):
        pytest.importorskip("torch")

    def test_completion_preserves_signal_range(self, model_path, sample_csv_ecg):
        from ecgtizer.completion import completion_
        result = completion_(sample_csv_ecg, model_path, 'cpu')

        for lead_name, lead_data in result.items():
            # Completed signals should be finite
            assert np.all(np.isfinite(lead_data)), f"Lead {lead_name} contains non-finite values"

    def test_completion_all_leads_populated(self, model_path, sample_csv_ecg):
        from ecgtizer.completion import completion_
        result = completion_(sample_csv_ecg, model_path, 'cpu')

        for lead_name, lead_data in result.items():
            # No lead should be all zeros
            assert not np.all(lead_data == 0), f"Lead {lead_name} is all zeros"


class TestExtractionRoundTrip:
    """Test: synthetic signal → binary image → extraction → compare."""

    def test_full_extraction_recovers_shape(self):
        """Extract signal from a synthetic binary image and verify shape recovery."""
        from ecgtizer.extraction_functions import full_extraction

        h, w = 300, 2000
        img = np.zeros((h, w), dtype=np.uint8)

        # Draw a known signal
        expected = []
        for x in range(w):
            y = int(h / 2 + 50 * np.sin(2 * np.pi * x / 400))
            y = max(0, min(h - 1, y))
            expected.append(y)
            img[y, x] = 255
            if y + 1 < h:
                img[y + 1, x] = 255
            if y - 1 >= 0:
                img[y - 1, x] = 255

        extracted = full_extraction(img)
        expected = np.array(expected, dtype=float)

        # Correlation between extracted and expected should be high
        corr = np.corrcoef(extracted, expected)[0, 1]
        assert corr > 0.95, f"Correlation too low: {corr}"

    def test_lazy_extraction_recovers_shape(self):
        from ecgtizer.extraction_functions import lazy_extraction

        h, w = 300, 2000
        img = np.zeros((h, w), dtype=np.uint8)

        expected = []
        for x in range(w):
            y = int(h / 2 + 50 * np.sin(2 * np.pi * x / 400))
            y = max(0, min(h - 1, y))
            expected.append(y)
            img[y, x] = 255
            if y + 1 < h:
                img[y + 1, x] = 255

        extracted = np.array(lazy_extraction(img), dtype=float)
        expected = np.array(expected, dtype=float)

        corr = np.corrcoef(extracted, expected)[0, 1]
        assert corr > 0.9, f"Correlation too low: {corr}"


class TestAnalysisPipeline:
    """Test: two ECG dicts → analyse → verify metrics."""

    def test_analyse_self_comparison(self, sample_csv_ecg):
        from ecgtizer.analyses import analyse
        result = analyse(sample_csv_ecg, sample_csv_ecg)

        # Self-comparison should yield perfect correlation
        for lead, corr in result['corr'].items():
            assert corr > 0.99, f"Lead {lead} self-correlation: {corr}"

        # Self-comparison should yield near-zero MSE
        for lead, mse in result['mse'].items():
            assert mse < 0.01, f"Lead {lead} self-MSE: {mse}"

    def test_analyse_noisy_comparison(self, sample_csv_ecg):
        from ecgtizer.analyses import analyse

        # Add noise to a copy
        noisy = {}
        rng = np.random.RandomState(42)
        for lead, data in sample_csv_ecg.items():
            noisy[lead] = data + rng.randn(len(data)) * 50

        result = analyse(sample_csv_ecg, noisy)
        # Noisy comparison should still have decent correlation (signal >> noise)
        avg_corr = np.mean(list(result['corr'].values()))
        assert avg_corr > 0.5


def _make_landscape_ecg_image(n_rows, row_h=80, gap=40, width=3000):
    """Synthetic landscape ECG: ``n_rows`` waveform tracks on white paper.

    Each track has a small calibration square at the left margin followed by a
    full-width sinusoidal waveform. Used to exercise track detection and the
    12x1 pipeline without a real scan.
    """
    amp = row_h // 3
    height = gap + n_rows * (row_h + gap)
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    def paint(x, y):
        y = max(0, min(height - 1, y))
        img[max(0, y - 1):y + 2, x] = 0

    for i in range(n_rows):
        yc = gap + i * (row_h + gap) + row_h // 2
        # Calibration square: up-step (baseline -> baseline-amp -> baseline).
        for x in range(30, 130):
            paint(x, yc if (x < 50 or x > 110) else yc - amp)
        for y in range(yc - amp, yc + 1):  # vertical edges of the square
            paint(50, y)
            paint(110, y)
        # Waveform body: a baseline-dominant trace (flat with periodic
        # QRS-like spikes), so each track yields a single variance peak at its
        # baseline — as real ECGs do — rather than the two turning-point peaks
        # a pure sinusoid would produce.
        period = 100 + i * 6  # vary rate per lead so signals differ
        for x in range(160, width):
            phase = (x - 160) % period
            if phase < 20:
                y = yc - int(amp * (1 - abs(phase - 10) / 10.0))  # triangular spike up
            else:
                y = yc  # baseline
            paint(x, y)
    return img


class TestTwelveByOneFormat:
    """12x1 layout: 12 (or 13) tracks, each carrying one full 10 s lead."""

    TWELVE_LEADS = {"I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"}

    @pytest.mark.parametrize("n_rows", [4, 6, 12])
    def test_track_detection_count(self, n_rows):
        """Adaptive detection resolves sparse (4, 6) and dense (12) stacks.

        Guards against the 12x1 spacing change over-segmenting 3x4/6x2 sheets.
        """
        from ecgtizer.PDF2XML import tracks_extraction
        img = _make_landscape_ecg_image(n_rows)
        dic_tracks, _, _ = tracks_extraction(img, "classic", 300, "", NOISE=False, DEBUG=False)
        assert len(dic_tracks) == n_rows

    def test_12x1_pipeline_yields_full_leads(self):
        """End-to-end (detect -> extract -> cut) on a synthetic 12-row sheet."""
        from ecgtizer.PDF2XML import tracks_extraction, lead_extraction, lead_cutting
        img = _make_landscape_ecg_image(12)
        dic_tracks, _, _ = tracks_extraction(img, "classic", 300, "", NOISE=False, DEBUG=False)
        assert len(dic_tracks) == 12
        dic_ex, image_bin, _ = lead_extraction(dic_tracks, "full", "classic", NOISE=False)
        leads = lead_cutting(dic_ex, 300, "classic", "", 0, NOISE=False, DEBUG=False, dic_image_bin=image_bin)
        assert set(leads.keys()) == self.TWELVE_LEADS
        for name, sig in leads.items():
            assert len(sig) == 5000, f"{name} has length {len(sig)}"
            assert np.all(np.isfinite(sig)), f"{name} has non-finite values"
            assert np.any(sig != 0), f"{name} is all zeros"

    def test_ecgtizer_class_end_to_end(self, tmp_output_dir):
        """Full ECGtizer entry point: crop + noise override + interleaved order."""
        import os
        import cv2
        from ecgtizer import ECGtizer

        img = _make_landscape_ecg_image(12)
        path = os.path.join(tmp_output_dir, "synthetic_12x1.png")
        cv2.imwrite(path, img)
        # typ/noise force the deterministic path (synthetic B/W lacks the colour
        # cues auto-detection relies on); lead_order picks the interleaved layout.
        ecg = ECGtizer(
            file=path, dpi=300, extraction_method="full", typ="classic", noise=False, lead_order="interleaved"
        )
        leads = ecg.extracted_lead
        assert isinstance(leads, dict)
        assert set(leads.keys()) == self.TWELVE_LEADS
        assert all(len(v) == 5000 for v in leads.values())


class TestMultiFormatSupport:
    """Test that different ECG formats are handled correctly."""

    def test_ecg_dict_with_ref_lead(self, sample_ecg_dict):
        """Adding a ref lead should not break analysis."""
        from ecgtizer.analyses import analyse
        ecg = dict(sample_ecg_dict)
        ecg['ref'] = np.zeros(5000)
        result = analyse(ecg, ecg)
        assert 'ref' not in result['corr']

    def test_7_lead_kardia_format(self):
        """7-lead Kardia format should be plottable."""
        import matplotlib
        matplotlib.use('Agg')
        from ecgtizer.PDF2XML_mod import plot_function
        leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'ref']
        t = np.linspace(0, 10, 5000)
        ecg = {l: np.sin(2 * np.pi * t) * 500 for l in leads}
        # Should not raise
        plot_function(ecg)
