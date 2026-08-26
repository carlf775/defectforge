# DefectForge

Turn a handful of annotated defect photos plus a pile of good-item photos into a full
synthetic defect-detection dataset in COCO format, ready to drop into Roboflow.

**[Open the tool →](https://USER.github.io/defectforge/)** *(runs entirely in your browser)*

```
your annotated defects  ──►  test  split   (kept real, untouched)
your good items         ──►  train / valid (Gemini paints defects onto them)
```

## How it works

1. You point it at two folders: an annotated COCO dataset of defects, and images of good items.
2. For each output image it picks a good image as the base, picks a real annotation, and crops it
   as a visual exemplar of what that defect looks like.
3. It sends base + exemplar to Gemini image editing ("Nano Banana") with an edit-only prompt:
   add one defect of this kind here, change nothing else.
4. **The annotation is measured, not guessed.** The result is diffed against the base
   (greyscale difference → morphological opening → largest connected component). Gemini is never
   asked for coordinates, because it is not reliable at reporting them. Edits that changed nothing,
   or that restyled the whole frame, are rejected and retried. The kept region yields both the
   bounding box and a real segmentation polygon (Moore boundary trace of the changed pixels), and
   `area` is the actual mask pixel count — so the dataset works for instance segmentation, not
   just detection.
5. Multiple defects per image stack: each pass edits the previous result.
6. Output is a zip with `train/`, `valid/`, `test/`, each holding its images and an
   `_annotations.coco.json` — Roboflow's own export layout, so importing preserves the splits.

Categories are unified across every split, so the synthetic classes line up with the real ones.

## Two ways to run it

### Browser (this repo's GitHub Page)

No install. A mandatory guided tour covers the workflow on first visit (re-run it any time with
the **Guided tour** button). Paste a Gemini API key, choose the two folders, tick the defect
classes, pick defect **size** (hairline → large) and **severity** (faint → severe), then
**Preview 3** — a handful of calls — before paying for a full batch. Click any result to inspect
it full size and hold space to flip against the original; drop anything unconvincing. Each
annotation also carries `attributes: {severity, width_pct, contrast}` in the COCO json.

> The key is sent from your browser directly to Google and nowhere else (the page's CSP only
> allows connections to `generativelanguage.googleapis.com`). It is kept in memory unless you
> tick *remember on this device*, which stores it in `localStorage`. Restrict the key to your
> Pages domain in Google AI Studio and never commit it.

### CLI, for large batches

```bash
pip install -r cli/requirements.txt
export GEMINI_API_KEY=...

python cli/synthgen.py --annotated ds/test --good good/ --list-classes
python cli/synthgen.py --annotated ds/test --good good/ --out synthetic \
       --train 400 --valid 100 --classes "scratch:2,dent:1" --zip
python cli/synthgen.py --selftest        # offline, checks the measurement + COCO output
```

`cli/synthgen_ui.py` is a local Gradio front end over the same engine
(`python cli/synthgen_ui.py`) if you want the UI without the browser's memory limits.

Overlays of every generated box land in `synthetic/preview/` — look at those before you upload.

## Publishing to GitHub Pages

Settings → Pages → Source: *Deploy from a branch* → `main` / `/ (root)`. There is no build step;
`index.html` is the whole app. Replace `USER` in the link above with your GitHub username.

## Limits worth knowing

- One Gemini image call per defect. 200 images at ~1.5 defects each ≈ 300 calls.
- Browser build holds every generated image in memory; past ~300 images use the CLI.
- The base image is re-encoded by the model, so a defect that blends *too* well can fall under
  the change threshold and get skipped. Skips are logged, never silently labelled.

The CLI is the plain engine — it does not yet have the size/severity presets or preview flow;
use `--hint` for the same effect in prose.
