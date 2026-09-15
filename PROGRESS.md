# Project log — DefectForge & weld-defect pipeline

Updated: 2026-09-14

## Session 2026-09-14 (and 2026-08-28) — pipeline execution on Carl's parts

- **Part radiographs on this machine**: `parts/partA` (4) + `parts/partB` (3), 2048×512,
  rendered from the 16-bit DICOMs in `~/Desktop/Xray JP /` with `tools/dcm_to_png.py`
  (log transform + CLAHE — the vendor 8-bit PNGs clip thick sections to black; always
  re-render from DICOM). `parts/all/` = the 7 PNGs + `weld_zones.coco.json`
  (hand-drawn weld_seam polygons, one per image, drawn in the app's annotator).
  parts/ and data/ are gitignored — client imagery stays out of the public repo.
- **Placement zones shipped** (browser + CLI): a COCO json of zone polygons included with
  the good images aims each defect at the zone and rejects (measured) off-zone edits.
  CLI: `--regions zones.json --region-desc "the weld seam"`. Selftests cover it.
- **"Next:" guidance bar** in the app: plain-language, state-driven next-step strip.
  Carl wants all guidance purpose-first and ultra-simple — no shortcut recipes.
- **Seam finder DONE**: `runs/segment/train/runs/seam_mix_ft/weights/best.pt`.
  Recipe that worked: SWRD pretrain (yolo11n-seg, 960px, box mAP50 0.945) then fine-tune
  on 200 SWRD + the 6 part-train images repeated 30× (val = held-out part image,
  mask mAP50 0.995). Zero-shot SWRD→parts does NOT work (whole-frame boxes).
- **Defect detector pretraining DONE — use the tiled one**:
  `runs/segment/train/runs/defect_tiles/weights/best.pt` (yolo11s-seg, 640px tiles from
  `tools/tile_dataset.py`: crops centered on defects + defect-free seam negatives).
  Tile-val box mAP50 0.45 overall, porosity 0.68, undercut 0.79; weak: inclusion 0.20,
  lack_of_penetration 0.17 → generate extra synthetic examples of those two classes.
  (Whole-image variant in defect_pretrain/ scored 0.39 — panorama downscale kills small
  defects; train and infer on tiles.)
- **Two parts now, not one**: per part reserve 1 image for the unseen-background test
  split; the rest are generation bases. Part A weld = tube-to-fitting joint;
  Part B welds marked by lead Y/M letters. OPEN QUESTION: does "Soudure M" (no "ok")
  contain a real defect? If yes it must leave the good-bases and becomes real test data.
- Gemini key: free tier has ZERO quota for image models ("limit: 0") — billing on the key
  is required even for Preview 3 (~$0.20); full batch ~225 image calls ≈ $9–10.
  New-format keys start with "AQ." (not only "AIza"). Blocked here: Carl still has to
  click "Set up billing", then re-run Preview 3.

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
