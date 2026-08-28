# DefectForge — context for Claude

Read PROGRESS.md first: it holds the full project state, the agreed training pipeline, and
the standing decisions. Treat those decisions as settled unless the user reopens them.

Quick facts:
- index.html IS the app (no build step; GitHub Pages serves the repo root). cli/ is the
  Python twin, tools/ holds the SWRD streaming extractor.
- Deploy = commit + push to main. Verify with the in-page self-test (link near the buttons).
- Core invariant, never break it: boxes and segmentation polygons are MEASURED from the
  pixel diff between base and edit; Gemini is never asked for coordinates. Real annotated
  images always become the test split.
- Datasets live in the GitHub release "datasets-v1" (SWRD-subset.zip, RIAWELC-roboflow.zip).
- The user works in the Zen browser (Firefox-based): always keep plain file pickers
  alongside folder pickers.
- User is Danish/English bilingual — answer in the language they write in.
