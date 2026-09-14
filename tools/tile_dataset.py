#!/usr/bin/env python3
"""Build a tiled YOLO-seg dataset from a COCO set: full-resolution crops around
defects (plus defect-free seam crops as negatives) instead of whole downscaled
panoramas. Matches sliding-window inference along the seam.

Usage:
    python tools/tile_dataset.py --coco data/images/_annotations.coco.json \
           --images data/images --out train/defect_tiles --tile 640
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image

DEFECT_EXCLUDE = {"weld_seam"}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def tile_origin(cx, cy, t, w, h, rng):
    """Tile top-left so (cx, cy) lands at a jittered position inside it."""
    jx = rng.randint(-t // 4, t // 4)
    jy = rng.randint(-t // 4, t // 4)
    return clamp(cx - t // 2 + jx, 0, max(0, w - t)), clamp(cy - t // 2 + jy, 0, max(0, h - t))


def anns_in_tile(anns, x0, y0, t):
    """Annotations overlapping >=40% of their own box with the tile, clipped."""
    out = []
    for a in anns:
        x, y, w, h = a["bbox"]
        ix = max(0, min(x + w, x0 + t) - max(x, x0))
        iy = max(0, min(y + h, y0 + t) - max(y, y0))
        if w <= 0 or h <= 0 or ix * iy < 0.4 * w * h:
            continue
        polys = []
        for p in a.get("segmentation") or []:
            if not isinstance(p, list) or len(p) < 6:
                continue
            q = []
            for px, py in zip(p[0::2], p[1::2]):
                q += [clamp(px - x0, 0, t), clamp(py - y0, 0, t)]
            polys.append(q)
        if polys:
            out.append((a["category_id"], polys))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--neg-per-image", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = json.load(open(args.coco))
    names = {c["id"]: c["name"] for c in d["categories"]}
    classes = sorted(n for n in set(names.values()) if n not in DEFECT_EXCLUDE)
    cls_of = {cid: classes.index(n) for cid, n in names.items() if n in classes}
    seam_ids = {cid for cid, n in names.items() if n == "weld_seam"}
    imgs = {i["id"]: i for i in d["images"]}
    by_img = {}
    for a in d["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    rng = random.Random(args.seed)
    ids = sorted(imgs)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * args.val_frac))
    split_of = {iid: ("val" if k < n_val else "train") for k, iid in enumerate(ids)}

    out = Path(args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    t = args.tile
    counts = {"train": 0, "val": 0, "neg": 0}
    for iid in ids:
        im_info = imgs[iid]
        split = split_of[iid]
        anns = by_img.get(iid, [])
        defects = [a for a in anns if a["category_id"] in cls_of]
        seams = [a for a in anns if a["category_id"] in seam_ids]
        img = Image.open(Path(args.images) / im_info["file_name"]).convert("RGB")
        w, h = img.size
        stem = Path(im_info["file_name"]).stem

        def write_tile(x0, y0, tag, k):
            crop = img.crop((x0, y0, min(x0 + t, w), min(y0 + t, h)))
            if crop.size != (t, t):  # pad edge tiles to full size
                full = Image.new("RGB", (t, t), (0, 0, 0))
                full.paste(crop, (0, 0))
                crop = full
            name = f"{stem}_{tag}{k}"
            crop.save(out / "images" / split / f"{name}.png")
            lines = []
            for cid, polys in anns_in_tile(defects, x0, y0, t):
                for p in polys:
                    pts = " ".join(f"{v / t:.6f}" for v in p)
                    lines.append(f"{cls_of[cid]} {pts}")
            (out / "labels" / split / f"{name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            counts[split] += 1
            return bool(lines)

        for k, a in enumerate(defects):
            x, y, bw, bh = a["bbox"]
            x0, y0 = tile_origin(int(x + bw / 2), int(y + bh / 2), t, w, h, rng)
            write_tile(x0, y0, "d", k)
        # defect-free seam tiles as negatives
        made = 0
        for a in rng.sample(seams, len(seams)):
            if made >= args.neg_per_image:
                break
            x, y, bw, bh = a["bbox"]
            for _ in range(8):
                cx = rng.randint(int(x), int(x + max(1, bw)))
                cy = rng.randint(int(y), int(y + max(1, bh)))
                x0, y0 = tile_origin(cx, cy, t, w, h, rng)
                if not anns_in_tile(defects, x0, y0, t):
                    write_tile(x0, y0, "n", made)
                    counts["neg"] += 1
                    made += 1
                    break

    (out / "tiles.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(classes)))
    print(f"tiles: {counts['train']} train / {counts['val']} val "
          f"(incl. {counts['neg']} negatives) -> {out}/tiles.yaml")


if __name__ == "__main__":
    main()
