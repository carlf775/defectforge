#!/usr/bin/env python3
"""Build a compact, in-repo defect library the web app can load with one click.

Picks a class-balanced handful of SWRD images (rare classes first), downscales
them, rescales their COCO annotations to match, and writes everything —
images, `_annotations.coco.json`, and a `manifest.json` file listing — under
datasets/<name>/ so GitHub Pages can serve it same-origin (no directory
listing needed; the app reads the manifest).

Usage:
    python tools/make_preloaded.py --coco data/images/_annotations.coco.json \
           --images data/images --out datasets/swrd --per-class 8 --maxdim 1100
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

EXCLUDE = {"weld_seam"}  # not a defect; still carried for completeness but not counted for balance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--maxdim", type=int, default=1100)
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--name", default="SWRD weld defects")
    args = ap.parse_args()

    d = json.load(open(args.coco))
    names = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    anns_by_img = defaultdict(list)
    for a in d["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # images grouped by the rarest defect class they contain -> pick round-robin
    defect_cats = [cid for cid, n in names.items() if n not in EXCLUDE]
    imgs_with = defaultdict(list)
    for iid, alist in anns_by_img.items():
        for cid in {a["category_id"] for a in alist if a["category_id"] in defect_cats}:
            imgs_with[cid].append(iid)
    # rarest class first
    order = sorted(defect_cats, key=lambda c: len(imgs_with[c]))
    chosen, seen = [], set()
    for cid in order:
        added = 0
        for iid in imgs_with[cid]:
            if iid in seen:
                continue
            seen.add(iid)
            chosen.append(iid)
            added += 1
            if added >= args.per_class:
                break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    new_images, new_anns, manifest_files = [], [], []
    ann_id = 1
    for new_id, iid in enumerate(chosen, 1):
        info = imgs[iid]
        src = Image.open(Path(args.images) / info["file_name"]).convert("RGB")
        w, h = src.size
        s = min(1.0, args.maxdim / max(w, h))
        nw, nh = round(w * s), round(h * s)
        if s < 1.0:
            src = src.resize((nw, nh), Image.LANCZOS)
        fname = f"swrd_{new_id:03d}.jpg"
        src.save(out / fname, quality=args.quality)
        manifest_files.append(fname)
        new_images.append({"id": new_id, "file_name": fname, "width": nw, "height": nh,
                           "license": 0, "date_captured": ""})
        for a in anns_by_img[iid]:
            x, y, bw, bh = a["bbox"]
            seg = []
            for p in a.get("segmentation") or []:
                if isinstance(p, list) and len(p) >= 6:
                    seg.append([round(v * s, 1) for v in p])
            new_anns.append({"id": ann_id, "image_id": new_id, "category_id": a["category_id"],
                             "bbox": [round(v * s, 1) for v in (x, y, bw, bh)],
                             "area": round(a.get("area", bw * bh) * s * s, 1),
                             "iscrowd": 0, "segmentation": seg})
            ann_id += 1

    coco_out = {"info": {"description": args.name}, "licenses": [{"id": 0, "name": "", "url": ""}],
                "categories": d["categories"], "images": new_images, "annotations": new_anns}
    (out / "_annotations.coco.json").write_text(json.dumps(coco_out))

    from collections import Counter
    cls_counts = Counter(names[a["category_id"]] for a in new_anns)
    (out / "manifest.json").write_text(json.dumps({
        "name": args.name,
        "description": "Radiographic weld defects (SWRD subset) — real masks.",
        "annotations": "_annotations.coco.json",
        "images": manifest_files,
        "image_count": len(new_images),
        "annotation_count": len(new_anns),
        "classes": dict(sorted(cls_counts.items(), key=lambda kv: -kv[1])),
    }, indent=1))

    total = sum((out / f).stat().st_size for f in manifest_files)
    print(f"{len(new_images)} images, {len(new_anns)} anns, {total/1e6:.1f} MB -> {out}")
    for k, v in sorted(cls_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
