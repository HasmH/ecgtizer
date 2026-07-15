#!/usr/bin/env python3
"""Batch-digitize a folder of ECG PDFs/images with ECGtizer.

For every file in ``INPUT_DIR`` this writes a per-ECG folder under
``OUTPUT_DIR`` containing:

    1. <name>_cropped.png   the original ECG, cropped to the trace area
    2. <name>_overlay.png   the digitized trace drawn over the original
    3. <name>.xml           HL7 aECG digitized signal
    4. <name>.csv           digitized leads (one column per lead), for neurokit2

Format (3x4 / 6x2 / 12x1 / Wellue / ...) is auto-detected — you do not tell it
the layout. ECGtizer crops the page internally first, which is what lets a
landscape recording on a portrait A4 page be classified correctly.

Run
---
    python run.py                       # uses the settings below
    python run.py --input some/dir --method fragmented

Leads are assumed printed in the standard order
(I, II, III, aVR, aVL, aVF, V1-V6); change LEAD_ORDER if a device differs.

Grids (bright pink/orange, or dark gray/near-black as on some Westmead exports)
are removed automatically during digitization, so both kinds of sheet work.

Requires ECGtizer to be installed (`pip install -e .`) and poppler for PDFs.
"""
from __future__ import annotations

import argparse
import os
import traceback

import cv2
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless: save figures, don't open windows
import matplotlib.pyplot as plt  # noqa: E402

from ecgtizer import ECGtizer  # noqa: E402
from ecgtizer.PDF2XML_mod import plot_overlay  # noqa: E402

# ─── Settings (edit these) ────────────────────────────────────────────────────
INPUT_DIR = "data/data"          # folder of ECG PDFs / images
OUTPUT_DIR = "data/digitized"    # one sub-folder is created per ECG
DPI = 300                        # rasterization resolution for PDFs
METHOD = "fragmented"            # "lazy" | "full" | "fragmented"
#   fragmented -> highest detail; captures tall QRS peaks (best default)
#   full       -> similar, slightly faster
#   lazy       -> smoothest, but clips tall/steep QRS peaks (loses amplitude);
#                 use it if fragmented looks too spiky on a given sheet
NOISE = None                     # None = auto-detect (best for real clinical
#   scans). Use False only for clean, bright-grid printouts where auto-detect
#   wrongly flags noise. NOTE: neither setting removes a *dark/gray* grid — see
#   the grid caveat in the module docstring.
LEAD_ORDER = "standard"          # 12x1 row order top->bottom; "standard",
#   "interleaved", or a list of 12 names. Ignored for non-12x1 layouts.
SAMPLING_RATE = 500              # Hz — pass this to neurokit2 when analysing
IMAGE_EXTS = (".pdf", ".png", ".jpg", ".jpeg")

# Column order for the CSV (what neurokit2 will read). Internal lead keys are
# upper-case (AVR/AVL/AVF); we relabel them to the conventional aVR/aVL/aVF.
CSV_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
KEY_TO_CSV = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}


def _save_cropped(ecg: ECGtizer, is_pdf: bool, path: str) -> None:
    """Save ECGtizer's internally-cropped source image (colour-correct)."""
    img = np.asarray(ecg.image)
    if img.ndim == 3 and is_pdf:
        # PDF pages arrive as RGB; cv2 writes BGR, so convert first.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def _save_overlay(ecg: ECGtizer, path: str) -> None:
    """Save the digitized trace drawn over the (cropped) original image."""
    plt.figure(figsize=(24, 16))
    try:
        plot_overlay(ecg.dic_tracks_ex_not_scale, ecg.image, ecg.varianceh, ecg.variancev)
    finally:
        plt.savefig(path, dpi=90, bbox_inches="tight")
        plt.close()


def _save_csv(ecg: ECGtizer, path: str) -> bool:
    """Write digitized leads as a CSV (mV) with one column per lead."""
    leads = ecg.extracted_lead
    if not isinstance(leads, dict):
        return False  # single-strip formats (e.g. Wellue) have no named leads
    columns = {}
    for key, csv_name in [(k, KEY_TO_CSV.get(k, k)) for k in leads]:
        if csv_name in CSV_LEAD_ORDER:
            # Pipeline stores µV; divide by 1000 for mV (the usual ECG unit).
            columns[csv_name] = np.asarray(leads[key], dtype=float) / 1000.0
    if not columns:
        return False
    ordered = [c for c in CSV_LEAD_ORDER if c in columns]
    pd.DataFrame({c: columns[c] for c in ordered}).to_csv(path, index=False)
    return True


def process(path: str, out_root: str, method: str) -> str:
    """Digitize one file and write its artifact folder. Returns a status line."""
    name = os.path.splitext(os.path.basename(path))[0]
    is_pdf = path.lower().endswith(".pdf")
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    ecg = ECGtizer(
        file=path,
        dpi=DPI,
        extraction_method=method,
        noise=NOISE,
        lead_order=LEAD_ORDER,
        verbose=False,
    )
    if not getattr(ecg, "good", True):
        return f"SKIP  {name}: could not read file"

    _save_cropped(ecg, is_pdf, os.path.join(out_dir, f"{name}_cropped.png"))
    _save_overlay(ecg, os.path.join(out_dir, f"{name}_overlay.png"))
    ecg.save_xml(os.path.join(out_dir, f"{name}.xml"))
    wrote_csv = _save_csv(ecg, os.path.join(out_dir, f"{name}.csv"))

    leads = ecg.extracted_lead
    n_leads = len(leads) if isinstance(leads, dict) else 1
    csv_note = "csv" if wrote_csv else "no-csv(single-strip)"
    return f"OK    {name}: type={ecg.TYPE} tracks={len(ecg.dic_tracks)} leads={n_leads} [{csv_note}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch-digitize ECGs with ECGtizer.")
    parser.add_argument("--input", default=INPUT_DIR, help=f"Input folder (default: {INPUT_DIR}).")
    parser.add_argument("--output", default=OUTPUT_DIR, help=f"Output folder (default: {OUTPUT_DIR}).")
    parser.add_argument("--method", default=METHOD, choices=["lazy", "full", "fragmented"])
    args = parser.parse_args(argv)

    files = sorted(
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.lower().endswith(IMAGE_EXTS)
    )
    if not files:
        print(f"No ECG files found in {args.input}")
        return 1

    os.makedirs(args.output, exist_ok=True)
    print(f"Digitizing {len(files)} file(s) from {args.input} -> {args.output} (method={args.method})\n")
    for path in files:
        try:
            print(process(path, args.output, args.method))
        except Exception as e:  # noqa: BLE001 - keep the batch going
            print(f"FAIL  {os.path.basename(path)}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
