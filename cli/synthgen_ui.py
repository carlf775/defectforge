#!/usr/bin/env python3
"""Browser UI for synthgen.py.   pip install gradio   &&   python synthgen_ui.py

Point it at the two folders, look at the defects it found, tick the ones you want,
generate, eyeball the boxed previews, download the Roboflow zip.
"""
import os, threading, time
from pathlib import Path

import gradio as gr

import synthgen as sg

MODELS = ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"]


def load(annotated, good):
    coco, js, exemplars, names = sg.load_coco(annotated)
    goods = sg.list_images(good)
    counts = sg.class_counts(exemplars, names)
    gallery = [(sg.exemplar_crop(e), names[e["category_id"]])
               for e in exemplars[:: max(1, len(exemplars) // 24)][:24]]
    state = {"annotated": annotated, "coco": coco, "js": js,
             "exemplars": exemplars, "names": names, "goods": goods}
    return (state,
            f"{len(exemplars)} annotated defects · {len(goods)} good images · "
            f"{len(counts)} class(es)",
            [[k, v] for k, v in counts.items()],
            gallery,
            gr.CheckboxGroup(choices=list(counts), value=list(counts), label="defects to generate"))


def generate(state, out, key, model, chosen, ntrain, nvalid, mind, maxd, workers, hint):
    if not state:
        raise gr.Error("load the folders first")
    if not key:
        raise gr.Error("Gemini API key required")
    if not chosen:
        raise gr.Error("pick at least one defect class")
    from google import genai
    names, exemplars = state["names"], state["exemplars"]
    weights = sg.parse_classes(",".join(chosen), exemplars, names)
    cfg = {"out": out, "seed": "0", "workers": int(workers), "retries": 2,
           "min_defects": int(mind), "max_defects": int(max(mind, maxd)),
           "hint": hint, "preview": True}
    edit = sg.gemini_editor(genai.Client(api_key=key), model)

    sg.copy_test(state["annotated"], state["js"], state["coco"], out)
    results, err = [], []
    total = int(ntrain) + int(nvalid)

    def run():
        try:
            for split, n in (("train", int(ntrain)), ("valid", int(nvalid))):
                if n:
                    sg.build_split(edit, cfg, split, n, state["goods"], exemplars, weights,
                                   state["coco"], names, on_result=results.append)
        except Exception as e:
            err.append(f"{type(e).__name__}: {e}")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    while t.is_alive():
        time.sleep(1)
        yield f"generating… {len(results)}/{total}", _gallery(results, names), None
    if err:
        raise gr.Error(err[0])
    n_ann = sum(len(r["boxes"]) for r in results)
    yield (f"done · {len(results)}/{total} images, {n_ann} annotations · test split copied · "
           f"written to {Path(out).resolve()}", _gallery(results, names), sg.zip_dataset(out))


def _gallery(results, names):
    return [(r["preview"], ", ".join(names[c] for c, _ in r["boxes"]))
            for r in results[-60:] if r.get("preview")]


with gr.Blocks(title="Synthetic defect dataset") as demo:
    gr.Markdown("## Synthetic defect dataset → Roboflow\n"
                "Annotated defects become the **test** split. Good images are the bases "
                "Gemini paints new defects onto for **train** / **valid**. "
                "Boxes are measured from the pixels that changed.")
    state = gr.State()
    with gr.Row():
        annotated = gr.Textbox(label="annotated dataset dir (images + COCO json)")
        good = gr.Textbox(label="good / defect-free images dir")
        out = gr.Textbox("synthetic", label="output dir")
    load_btn = gr.Button("Load dataset", variant="secondary")
    status = gr.Markdown()
    with gr.Row():
        table = gr.Dataframe(headers=["class", "annotated examples"], label="what's in the dataset",
                             interactive=False, scale=1)
        exemplars = gr.Gallery(label="the defects it will imitate", columns=6, height=260, scale=2)
    chosen = gr.CheckboxGroup(choices=[], label="defects to generate")
    with gr.Row():
        ntrain = gr.Number(200, label="train images", precision=0)
        nvalid = gr.Number(50, label="valid images", precision=0)
        mind = gr.Slider(1, 5, 1, step=1, label="min defects / image")
        maxd = gr.Slider(1, 5, 2, step=1, label="max defects / image")
        workers = gr.Slider(1, 16, 4, step=1, label="parallel requests")
    with gr.Row():
        model = gr.Dropdown(MODELS, value=MODELS[0], label="model", allow_custom_value=True)
        key = gr.Textbox(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "",
                         label="Gemini API key", type="password")
    hint = gr.Textbox(label="extra prompt hint (optional)",
                      placeholder="e.g. defects should sit on the weld seam, slightly out of focus")
    go = gr.Button("Generate", variant="primary")
    progress = gr.Markdown()
    results = gr.Gallery(label="generated (red box = annotation written to COCO)", columns=4,
                         height=460)
    zipfile = gr.File(label="Roboflow-ready zip")

    load_btn.click(load, [annotated, good], [state, status, table, exemplars, chosen])
    go.click(generate,
             [state, out, key, model, chosen, ntrain, nvalid, mind, maxd, workers, hint],
             [progress, results, zipfile])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
