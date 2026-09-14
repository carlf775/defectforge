#!/usr/bin/env python3
"""Render 16-bit radiograph DICOMs as well-windowed 8-bit PNGs.

The vendor 8-bit exports clip thick sections to pure black; the DICOM keeps the
full dynamic range. Windowing here is percentile-based per image (default
0.2-99.8), optionally followed by CLAHE for local contrast in dense regions.

Usage:
    python tools/dcm_to_png.py --src "folder with .dcm" --out parts/partX \
           [--lo 0.2 --hi 99.8] [--clahe]
"""
import argparse
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


def render(path, lo_pct, hi_pct, clahe):
    ds = pydicom.dcmread(path)
    a = ds.pixel_array.astype(np.float64)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = a.max() - a
    a = np.log1p(a)  # radiograph intensity is exponential in thickness
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    if hi <= lo:
        lo, hi = a.min(), max(a.max(), a.min() + 1e-6)
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    if clahe:
        import cv2

        h, w = a.shape
        # tile grid roughly proportional to aspect ratio, ~128px tiles
        grid = (max(2, round(w / 128)), max(2, round(h / 128)))
        a = cv2.createCLAHE(clipLimit=4.0, tileGridSize=grid).apply(
            (a * 65535).astype(np.uint16)
        ) / 65535.0
    return Image.fromarray((a * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=99.8)
    ap.add_argument("--clahe", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for dcm in sorted(Path(args.src).rglob("*.dcm")):
        if dcm.stat().st_size == 0:
            print(f"SKIP (empty placeholder): {dcm.name}")
            continue
        img = render(dcm, args.lo, args.hi, args.clahe)
        dest = out / (dcm.stem + ".png")
        img.save(dest)
        print(f"{img.size[0]}x{img.size[1]}  {dest.name[:70]}")


if __name__ == "__main__":
    main()
