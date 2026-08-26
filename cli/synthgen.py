#!/usr/bin/env python3
"""Synthetic COCO defect dataset via Gemini image editing ("nano banana").

  annotated defect dataset (COCO)  ->  copied verbatim to  out/test
  good (defect-free) images        ->  bases for generated  out/train, out/valid

For each generated image: pick a good base, pick a real annotated defect, crop it
as a visual exemplar, ask Gemini to paint that kind of defect onto the base.
The bbox is then MEASURED by diffing the edit against the base -- never guessed
from what the model claims -- so the labels match the pixels.

Output is a Roboflow-ready COCO dataset (one _annotations.coco.json per split).

  pip install google-genai pillow numpy scipy
  export GEMINI_API_KEY=...
  python synthgen.py --annotated ds/test --good good/ --list-classes
  python synthgen.py --annotated ds/test --good good/ --out synth --train 200 --valid 50 \
                     --classes "scratch:2,dent:1" --zip
  python synthgen.py --selftest      # no network: checks bbox measurement + COCO output
"""
import argparse, io, json, os, random, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CELLS = ["top-left", "top-center", "top-right",
         "middle-left", "center", "middle-right",
         "bottom-left", "bottom-center", "bottom-right"]

PROMPT = """You are producing training data for an automated visual defect-inspection system.

Image 1: a defect-free item. Image 2: a close-up crop of a real "{cat}" defect from another item.

Edit image 1 so the item has exactly one realistic "{cat}" defect of the same type, appearance and
texture as image 2, in the {region} area of the item. The defect should span roughly {size}% of the
image width.{hint}

Critical: change nothing else. Same camera angle, framing, crop, zoom, lighting, white balance,
colours, background and resolution as image 1. Every pixel outside the defect must stay identical.
Do not restyle, do not clean up, do not add text, labels, arrows or watermarks."""


# ---------- measurement ----------

def trace_poly(comp):
    """Moore boundary trace of a binary component -> [(x, y), ...], ~60 vertices."""
    ys, xs = np.nonzero(comp)
    if not len(ys):
        return []
    sy, sx = int(ys[0]), int(xs[ys == ys[0]].min())
    D = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    h, w = comp.shape
    at = lambda x, y: 0 <= x < w and 0 <= y < h and comp[y, x]
    pts, cx, cy, dr = [(sx, sy)], sx, sy, 6      # topmost-left start: nothing above it
    for _ in range(40000):
        for i in range(8):
            dd = (dr + i) % 8
            nx, ny = cx + D[dd][0], cy + D[dd][1]
            if at(nx, ny):
                cx, cy, dr = nx, ny, (dd + 6) % 8
                pts.append((cx, cy))
                break
        else:
            break                                # isolated pixel
        if (cx, cy) == (sx, sy):
            break
    step = max(1, len(pts) // 60)
    out = pts[::step]
    return out if len(out) >= 3 else pts


def diff_bbox(before, after):
    """Largest changed region -> {box, poly, mask_area, strength}, or None."""
    a = np.asarray(before.convert("L"), np.int16)
    b = np.asarray(after.convert("L"), np.int16)
    if a.shape != b.shape:
        return None
    d = np.abs(a - b).astype(np.uint8)
    d = np.asarray(Image.fromarray(d).filter(ImageFilter.MedianFilter(5)))  # kill speckle/recompression
    if d.max() < 12:
        return None
    mask = d >= max(12, 0.4 * d.max())
    lbl, n = ndimage.label(mask)
    if not n:
        return None
    biggest = 1 + int(np.argmax(ndimage.sum(mask, lbl, range(1, n + 1))))
    comp = lbl == biggest
    ys, xs = ndimage.find_objects(comp)[0]
    box = [int(xs.start), int(ys.start), int(xs.stop - xs.start), int(ys.stop - ys.start)]
    return {"box": box, "poly": trace_poly(comp), "mask_area": int(comp.sum()),
            "strength": int(d[comp].mean() / 255 * 100)}


def plausible(box, size, lo=5e-5, hi=0.25):
    """Reject global restyles (whole image changed) and single-pixel noise."""
    if box is None:
        return False
    frac = (box[2] * box[3]) / float(size[0] * size[1])
    return lo <= frac <= hi and box[2] >= 4 and box[3] >= 4


# ---------- dataset io ----------

def load_coco(d):
    """-> (coco dict, json path, exemplars, {cat_id: name})"""
    d = Path(d)
    js = d / "_annotations.coco.json"
    if not js.exists():
        cands = sorted(d.glob("*.json"))
        if not cands:
            raise SystemExit(f"no COCO json in {d}")
        js = cands[0]
    coco = json.loads(js.read_text())
    by_id = {i["id"]: i for i in coco["images"]}
    exemplars = []
    for a in coco["annotations"]:
        x, y, w, h = [int(v) for v in a["bbox"]]
        if w < 6 or h < 6:
            continue
        exemplars.append({"path": d / by_id[a["image_id"]]["file_name"],
                          "bbox": (x, y, w, h), "category_id": a["category_id"]})
    if not exemplars:
        raise SystemExit(f"no usable annotations in {js}")
    return coco, js, exemplars, {c["id"]: c["name"] for c in coco["categories"]}


def exemplar_crop(ex, pad=0.35):
    im = Image.open(ex["path"]).convert("RGB")
    x, y, w, h = ex["bbox"]
    px, py = int(w * pad), int(h * pad)
    return im.crop((max(0, x - px), max(0, y - py),
                    min(im.width, x + w + px), min(im.height, y + h + py)))


def list_images(d):
    files = sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"no images in {d}")
    return files


def class_counts(exemplars, names):
    counts = {}
    for e in exemplars:
        counts[names[e["category_id"]]] = counts.get(names[e["category_id"]], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def parse_classes(spec, exemplars, names):
    """'scratch:2,dent' -> {cat_id: weight}. Empty spec = every class, weighted by exemplar count."""
    have = {n: i for i, n in names.items() if any(e["category_id"] == i for e in exemplars)}
    if not spec:
        return {i: sum(e["category_id"] == i for e in exemplars) for i in have.values()}
    weights = {}
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        name, _, w = part.partition(":")
        if name not in have:
            raise SystemExit(f"unknown class {name!r}; available: {', '.join(have)}")
        weights[have[name]] = float(w or 1)
    return weights


# ---------- generation ----------

def gemini_editor(client, model):
    def edit(base, crop, cat, region, size_pct, hint):
        prompt = PROMPT.format(cat=cat, region=region, size=size_pct,
                               hint=f" {hint.strip()}" if hint else "")
        r = client.models.generate_content(model=model, contents=[prompt, base, crop])
        for p in r.candidates[0].content.parts:
            if getattr(p, "inline_data", None) and p.inline_data.data:
                return Image.open(io.BytesIO(p.inline_data.data)).convert("RGB")
        return None
    return edit


def make_image(edit, cfg, base_path, by_cat, weights, names, rng):
    """-> (image, [(category_id, bbox), ...])"""
    img = Image.open(base_path).convert("RGB")
    boxes = []
    cats = list(weights)
    for _ in range(rng.randint(cfg["min_defects"], cfg["max_defects"])):
        cat_id = rng.choices(cats, weights=[weights[c] for c in cats])[0]
        ex = rng.choice(by_cat[cat_id])
        crop = exemplar_crop(ex)
        region, size_pct = rng.choice(CELLS), rng.randint(4, 18)
        for attempt in range(cfg["retries"] + 1):
            try:
                out = edit(img, crop, names[cat_id], region, size_pct, cfg.get("hint", ""))
            except Exception as e:                      # rate limit / safety / transport
                print(f"  ! {type(e).__name__}: {e}", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
                continue
            if out is None:
                continue
            if out.size != img.size:
                out = out.resize(img.size, Image.LANCZOS)
            m = diff_bbox(img, out)
            if m and plausible(m["box"], img.size):
                boxes.append((cat_id, m))
                img = out                               # stack the next defect on this result
                break
    return img, boxes


def draw_preview(img, boxes, names):
    p = img.copy()
    d = ImageDraw.Draw(p)
    for cat_id, m in boxes:
        x, y, w, h = m["box"]
        if len(m["poly"]) >= 3:
            d.polygon(m["poly"], outline=(255, 0, 0), width=3)
        else:
            d.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=3)
        d.text((x + 4, max(0, y - 12)), names[cat_id], fill=(255, 0, 0))
    return p


def build_split(edit, cfg, split, count, goods, exemplars, weights, coco, names, on_result=None):
    out_dir = Path(cfg["out"]) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_dir = Path(cfg["out"]) / "preview" / split
    if cfg.get("preview", True):
        prev_dir.mkdir(parents=True, exist_ok=True)
    by_cat = {c: [e for e in exemplars if e["category_id"] == c] for c in weights}

    def task(i):
        rng = random.Random(f"{cfg['seed']}-{split}-{i}")
        img, boxes = make_image(edit, cfg, rng.choice(goods), by_cat, weights, names, rng)
        if not boxes:
            print(f"  {split} {i}: no usable defect, skipped", file=sys.stderr)
            return None
        name = f"synth_{split}_{i:05d}.png"
        img.save(out_dir / name)
        prev = str(prev_dir / name) if cfg.get("preview", True) else None
        if prev:
            draw_preview(img, boxes, names).save(prev)
        r = {"file_name": name, "size": img.size, "boxes": boxes, "preview": prev}
        if on_result:
            on_result(r)
        return r

    with ThreadPoolExecutor(cfg["workers"]) as pool:
        results = [r for r in pool.map(task, range(count)) if r]

    images, annotations, ann_id = [], [], 1
    for img_id, r in enumerate(results, 1):
        w, h = r["size"]
        images.append({"id": img_id, "file_name": r["file_name"], "width": w, "height": h,
                       "license": 0, "date_captured": ""})
        for cat_id, m in r["boxes"]:
            x, y, bw, bh = m["box"]
            seg = ([[c for pt in m["poly"] for c in pt]] if len(m["poly"]) >= 3
                   else [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]])
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": cat_id,
                                "bbox": [x, y, bw, bh], "area": m["mask_area"], "iscrowd": 0,
                                "segmentation": seg})
            ann_id += 1
    (out_dir / "_annotations.coco.json").write_text(json.dumps(
        {"info": {"description": f"synthetic {split}", "version": "1", "year": 2026},
         "licenses": [{"id": 0, "name": "synthetic", "url": ""}],
         "categories": coco["categories"], "images": images, "annotations": annotations}, indent=1))
    print(f"{split}: {len(images)} images, {len(annotations)} annotations -> {out_dir}")
    return results


def copy_test(annotated, js, coco, out):
    dst = Path(out) / "test"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for im in coco["images"]:
        src = Path(annotated) / im["file_name"]
        if src.exists():
            shutil.copy2(src, dst / im["file_name"])
            n += 1
    shutil.copy2(js, dst / "_annotations.coco.json")
    print(f"test: {n} real images copied -> {dst}")
    return n


def zip_dataset(out):
    """Roboflow accepts this zip directly and keeps the train/valid/test split."""
    out = Path(out)
    staging = out.parent / f".{out.name}_zip"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for split in ("train", "valid", "test"):
        if (out / split).exists():
            shutil.copytree(out / split, staging / split)
    path = shutil.make_archive(str(out), "zip", staging)
    shutil.rmtree(staging)
    print(f"zip -> {path}   (upload at app.roboflow.com, splits preserved)")
    return path


# ---------- self-check ----------

def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    base = Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)).filter(
        ImageFilter.GaussianBlur(2))
    after = base.copy()
    ImageDraw.Draw(after).ellipse((100, 60, 160, 110), fill=(20, 20, 20))
    m = diff_bbox(base, after)
    box = m["box"]
    assert abs(box[0] - 100) < 6 and abs(box[1] - 60) < 6, box
    assert abs(box[2] - 61) < 8 and abs(box[3] - 51) < 8, box
    assert len(m["poly"]) >= 6 and all(
        box[0] - 4 <= px <= box[0] + box[2] + 4 and box[1] - 4 <= py <= box[1] + box[3] + 4
        for px, py in m["poly"]), "polygon must hug the changed region"
    import math
    assert 0.5 * math.pi * 30 * 25 < m["mask_area"] < 1.6 * math.pi * 30 * 25, m["mask_area"]
    assert plausible(box, base.size) and diff_bbox(base, base.copy()) is None
    shift = Image.fromarray(np.clip(np.asarray(base, np.int16) + 40, 0, 255).astype(np.uint8))
    g = diff_bbox(base, shift)
    assert g is None or not plausible(g["box"], base.size), "global restyle must be rejected"

    # end-to-end with a stub editor: does the emitted COCO actually match the pixels?
    tmp = Path(tempfile.mkdtemp())
    base.save(tmp / "good.png")
    coco = {"categories": [{"id": 1, "name": "scratch", "supercategory": "defect"}]}
    exemplars = [{"path": tmp / "good.png", "bbox": (10, 10, 30, 30), "category_id": 1}]
    def stub(img, crop, cat, region, size_pct, hint):
        out = img.copy()
        ImageDraw.Draw(out).ellipse((40, 30, 90, 70), fill=(15, 15, 15))
        return out
    cfg = {"out": tmp / "ds", "seed": "0", "workers": 2, "retries": 0,
           "min_defects": 1, "max_defects": 1, "preview": True}
    build_split(stub, cfg, "train", 3, [tmp / "good.png"], exemplars, {1: 1.0}, coco,
                {1: "scratch"})
    js = json.loads((tmp / "ds" / "train" / "_annotations.coco.json").read_text())
    assert len(js["images"]) == 3 and len(js["annotations"]) == 3, js
    a = js["annotations"][0]
    assert abs(a["bbox"][0] - 40) < 6 and abs(a["bbox"][1] - 30) < 6, a["bbox"]
    assert len(a["segmentation"][0]) >= 12, "want a real polygon, not a rectangle"
    assert a["area"] < a["bbox"][2] * a["bbox"][3], "mask area must be tighter than the bbox"
    im = js["images"][0]
    on_disk = Image.open(tmp / "ds" / "train" / im["file_name"])
    assert (im["width"], im["height"]) == on_disk.size
    assert {i["id"] for i in js["images"]} == {1, 2, 3}
    shutil.rmtree(tmp)
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotated", help="dir with defect images + COCO json (becomes test split)")
    ap.add_argument("--good", help="dir with defect-free images (bases for generation)")
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--valid", type=int, default=50)
    ap.add_argument("--classes", default="", help='e.g. "scratch:2,dent:1" (default: all, natural mix)')
    ap.add_argument("--hint", default="", help="extra sentence appended to the prompt")
    ap.add_argument("--min-defects", type=int, default=1)
    ap.add_argument("--max-defects", type=int, default=2)
    ap.add_argument("--model", default="gemini-2.5-flash-image")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--seed", default="0")
    ap.add_argument("--no-preview", action="store_true", help="skip out/preview/ overlay images")
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip for Roboflow upload")
    ap.add_argument("--list-classes", action="store_true")
    ap.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.annotated:
        ap.error("--annotated is required")
    coco, js, exemplars, names = load_coco(args.annotated)
    if args.list_classes:
        for name, n in class_counts(exemplars, names).items():
            print(f"{n:5d}  {name}")
        return
    if not args.good:
        ap.error("--good is required")
    if not args.api_key:
        ap.error("set GEMINI_API_KEY or pass --api-key")

    from google import genai
    edit = gemini_editor(genai.Client(api_key=args.api_key), args.model)
    goods = list_images(args.good)
    weights = parse_classes(args.classes, exemplars, names)
    cfg = {"out": args.out, "seed": args.seed, "workers": args.workers, "retries": args.retries,
           "min_defects": args.min_defects, "max_defects": args.max_defects,
           "hint": args.hint, "preview": not args.no_preview}
    print(f"{len(exemplars)} exemplars, {len(goods)} good images, generating: "
          + ", ".join(f"{names[c]}({w:g})" for c, w in weights.items()))

    copy_test(args.annotated, js, coco, args.out)
    for split, n in (("train", args.train), ("valid", args.valid)):
        if n:
            build_split(edit, cfg, split, n, goods, exemplars, weights, coco, names)
    if args.zip:
        zip_dataset(args.out)


if __name__ == "__main__":
    main()
