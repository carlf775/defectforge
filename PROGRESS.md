# Project log — DefectForge & weld-defect pipeline

Updated: 2026-08-28

## What exists

- **Live tool**: https://carlf775.github.io/defectforge/ — browser app, generates synthetic
  COCO defect datasets via Gemini image editing (Nano Banana). Real annotated defects → test
  split; good images → bases for generated train/valid. Boxes AND segmentation masks are
  measured from the pixel diff, never asked from the model. Built-in Roboflow-style polygon
  annotator ("draw them here"). CLI twin in `cli/`.
- **Datasets on this machine** (also in the GitHub release "datasets-v1"):
  - `~/Downloads/SWRD-subset/` + `.zip` (364 MB) — 600 images / 3,954 COCO polygons from the
    124 GB SWRD archive, streamed selectively (never downloaded in full) via
    `tools/swrd_extract.py`. ~100 images per defect class + 751 weld_seam regions.
    Classes: porosity, crack, inclusion, lack_of_penetration, lack_of_fusion, undercut, weld_seam.
    Cite: Zhao et al., J Nondestructive Evaluation 44:50 (2025). Source: tz-ndt.com/#/download.
  - `~/Downloads/RIAWELC/roboflow/` + `RIAWELC-roboflow.zip` (466 MB) — 24,407 224px
    radiographs, classification only (CR/PO/LP/ND), no localization. ND folder = good items.

## The plan (agreed)

Constraint: only 3–4 defect-free images of the real part, zero real defect images, no more coming.

1. **Seam finder**: train on SWRD weld_seam polygons (751 regions in the subset).
2. **Defect detector pretraining**: SWRD subset (real defects, real masks).
3. **Fine-tune**: DefectForge with SWRD as exemplars + the 3 good part images as bases
   (~150 train / 40 valid). Roboflow tiling for big panoramas.
4. **Test**: the 4th good image, reserved untouched, gets its own DefectForge run (~30 images,
   different seed) = test split with an unseen background. Raw good images = false-alarm check.
5. Honest claim: proves the pipeline + synthetic-defect detection on this part; does NOT prove
   real-defect recall — that needs physical reference specimens or production rejects.

Inference recipe for whole-part radiographs: locate seam → sliding-window tiles along it at
training scale → merge detections. Whole-image single-pass inference will miss small defects.

## Working on another machine

`git clone https://github.com/carlf775/defectforge && cd defectforge && claude` — CLAUDE.md
briefs the session automatically. Datasets: download from the datasets-v1 release. The release
also holds `swrd_ann_cache.json.gz` + `swrd_names.json.gz`: gunzip them into the directory you
run `tools/swrd_extract.py` from and the extractor skips its 20-minute index pass and resumes
instantly (it also skips any images already on disk).

## Key decisions

- Train on synthetic, test on real (or at minimum: unseen-background synthetic + real negatives).
- Rare classes first when subsetting (SWRD is 82% porosity naturally).
- Gemini is never asked for coordinates — labels come from measuring changed pixels.
