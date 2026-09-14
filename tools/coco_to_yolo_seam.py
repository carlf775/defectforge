#!/usr/bin/env python3
"""Build a YOLO segmentation dataset for the seam finder from the SWRD COCO subset.

Keeps only the weld_seam class (single class 0). Every image in the subset has at
least one seam annotation, so no image is a silent negative. Split is a seeded
90/10 by image. Images are symlinked, not copied.

Usage:
    python tools/coco_to_yolo_seam.py --coco data/images/_annotations.coco.json \
           --images data/images --out train/seam_dataset
"""
import argparse
import json
import random
from pathlib import Path


def polygons(ann):
    seg = ann.get("segmentation") or []
    if isinstance(seg, dict):  # RLE — not present in this subset
        return []
    return [p for p in seg if len(p) >= 6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class-name", default="weld_seam",
                    help='single class, or comma-list e.g. "crack,porosity", or "all-except:weld_seam"')
    args = ap.parse_args()

    d = json.load(open(args.coco))
    names = {c["id"]: c["name"] for c in d["categories"]}
    if args.class_name.startswith("all-except:"):
        drop = set(args.class_name.split(":", 1)[1].split(","))
        wanted = [n for n in sorted(set(names.values())) if n not in drop]
    else:
        wanted = args.class_name.split(",")
    cls_of = {cid: wanted.index(n) for cid, n in names.items() if n in wanted}
    imgs = {i["id"]: i for i in d["images"]}

    labels = {}  # image_id -> [yolo line, ...]
    for a in d["annotations"]:
        if a["category_id"] not in cls_of:
            continue
        im = imgs[a["image_id"]]
        w, h = im["width"], im["height"]
        for poly in polygons(a):
            xs = poly[0::2]
            ys = poly[1::2]
            pts = " ".join(
                f"{min(max(x / w, 0), 1):.6f} {min(max(y / h, 0), 1):.6f}"
                for x, y in zip(xs, ys)
            )
            labels.setdefault(a["image_id"], []).append(
                f"{cls_of[a['category_id']]} {pts}")

    ids = sorted(labels)
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * args.val_frac))
    splits = {"val": ids[:n_val], "train": ids[n_val:]}

    out = Path(args.out)
    src_dir = Path(args.images).resolve()
    for split, split_ids in splits.items():
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for iid in split_ids:
            fname = imgs[iid]["file_name"]
            link = img_dir / fname
            if not link.exists():
                link.symlink_to(src_dir / fname)
            (lbl_dir / (Path(fname).stem + ".txt")).write_text(
                "\n".join(labels[iid]) + "\n"
            )

    (out / "seam.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(wanted))
    )
    print(
        f"{len(splits['train'])} train / {len(splits['val'])} val images, "
        f"{sum(len(v) for v in labels.values())} seam polygons -> {out}/seam.yaml"
    )


if __name__ == "__main__":
    main()
