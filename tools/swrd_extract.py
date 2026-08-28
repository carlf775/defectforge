"""Pull ~100 images per defect class from the SWRD zip on Drive, streamed.
Output: ~/Downloads/SWRD-subset/ images/*.png + _annotations.coco.json (+zip)."""
import json, os, io, sys, zipfile, collections, concurrent.futures as cf
import numpy as np
from PIL import Image
from swrd_stream import HttpFile, URL

PER_CLASS = 100
CLS = {"气孔":"porosity", "裂纹":"crack", "夹渣":"inclusion",
       "未焊透":"lack_of_penetration", "未熔合":"lack_of_fusion", "咬边":"undercut"}
CLS_ALL = dict(CLS, **{"焊缝": "weld_seam"})   # seam regions ride along, don't drive selection
OUT = os.path.expanduser("~/Downloads/SWRD-subset")
os.makedirs(OUT + "/images", exist_ok=True)

names = json.load(open("swrd_names.json"))
z = zipfile.ZipFile(HttpFile(URL))
jsons = sorted(n for n in names if n.startswith("crop_weld_data/crop_weld_jsons/")
               and n.endswith(".json"))
tif_of = lambda jn: jn.replace("crop_weld_jsons", "crop_weld_images").replace(".json", ".tif")
tif_size = {i.filename: i.file_size for i in z.infolist() if i.filename.endswith(".tif")}

# pass 1: read every crop json (small), index images by class
CACHE = "swrd_ann_cache.json"
if os.path.exists(CACHE):
    ann = {k: tuple(v) for k, v in json.load(open(CACHE)).items()}
    print(f"annotation index loaded from cache: {len(ann)} images", flush=True)
else:
    print(f"reading {len(jsons)} annotation files…", flush=True)
    ann = {}
    for k, jn in enumerate(jsons):
        if k % 500 == 0: print(f"  {k}/{len(jsons)}", flush=True)
        try: d = json.loads(z.read(jn))
        except Exception: continue
        t = tif_of(jn)
        if t not in tif_size: continue
        shapes = [s for s in d.get("shapes", []) if s.get("label") in CLS_ALL
                  and len(s.get("points", [])) >= 3]
        if not any(s["label"] in CLS for s in shapes): continue   # need a real defect
        ann[jn] = (t, shapes, d.get("imageWidth"), d.get("imageHeight"))
    json.dump(ann, open(CACHE, "w"))
by_class = collections.defaultdict(list)
for jn, (t, shapes, w, h) in ann.items():
    for c in {s["label"] for s in shapes if s["label"] in CLS}: by_class[c].append(jn)

print({CLS[k]: len(v) for k, v in by_class.items()}, flush=True)

# selection: smallest tifs first, rarest classes first, dedupe
chosen = []
seen = set()
for c in sorted(by_class, key=lambda c: len(by_class[c])):
    pool = sorted((jn for jn in by_class[c] if jn not in seen),
                  key=lambda jn: tif_size[ann[jn][0]])
    take = pool[:PER_CLASS]
    chosen += take; seen.update(take)
print(f"{len(chosen)} images selected, "
      f"{sum(tif_size[ann[j][0]] for j in chosen)/1e9:.1f} GB of tif to fetch", flush=True)

import threading, time, random
_tl = threading.local()
def _zip():
    if not hasattr(_tl, "z"): _tl.z = zipfile.ZipFile(HttpFile(URL))
    return _tl.z

def grab(jn):
    t, shapes, w, h = ann[jn]
    name = os.path.basename(t).replace(".tif", ".png")
    if os.path.exists(f"{OUT}/images/{name}"):            # resume
        return (jn, name)
    raw = None
    for attempt in range(5):
        try: raw = _zip().read(t); break
        except Exception as e:
            if hasattr(_tl, "z"): del _tl.z               # fresh connection
            time.sleep(3 * (attempt + 1) + random.random() * 5)
            if attempt == 4: return (jn, f"FAIL {e}")
    if raw is None: return (jn, "FAIL no data")
    im = Image.open(io.BytesIO(raw))
    a = np.asarray(im).astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)  # contrast stretch like the paper
    a = np.clip((a - lo) / max(1, hi - lo) * 255, 0, 255).astype(np.uint8)
    name = os.path.basename(t).replace(".tif", ".png")
    Image.fromarray(a).save(f"{OUT}/images/{name}")
    return (jn, name)

results = {}
with cf.ThreadPoolExecutor(3) as pool:
    for i, (jn, r) in enumerate(pool.map(grab, chosen)):
        results[jn] = r
        print(f"  [{i+1}/{len(chosen)}] {r}", flush=True)

# COCO
cats = [{"id": i+1, "name": v, "supercategory": "defect" if v != "weld_seam" else "region"}
        for i, v in enumerate(dict.fromkeys(CLS_ALL.values()))]
cid = {c["name"]: c["id"] for c in cats}
images, annos, aid = [], [], 1
for iid, jn in enumerate([j for j in chosen if not str(results[j]).startswith("FAIL")], 1):
    t, shapes, w, h = ann[jn]
    images.append({"id": iid, "file_name": results[jn], "width": w, "height": h,
                   "license": 0, "date_captured": ""})
    for s in shapes:
        pts = [(round(x), round(y)) for x, y in s["points"]]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        area = abs(sum(xs[i]*ys[(i+1) % len(pts)] - xs[(i+1) % len(pts)]*ys[i]
                       for i in range(len(pts)))) // 2
        annos.append({"id": aid, "image_id": iid, "category_id": cid[CLS_ALL[s["label"]]],
                      "bbox": [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)],
                      "area": int(area), "iscrowd": 0,
                      "segmentation": [[c for p in pts for c in p]]})
        aid += 1
coco = {"info": {"description": "SWRD subset (streamed from the 124GB release; "
        "cite Zhao et al., J Nondestr Eval 44:50, 2025)"},
        "licenses": [{"id": 0, "name": "", "url": ""}],
        "categories": cats, "images": images, "annotations": annos}
json.dump(coco, open(f"{OUT}/images/_annotations.coco.json", "w"))
per = collections.Counter(a["category_id"] for a in annos)
print("per-class annotations:", {c["name"]: per.get(c["id"], 0) for c in cats}, flush=True)
import shutil
shutil.make_archive(OUT, "zip", OUT)
print(f"DONE: {len(images)} images, {len(annos)} annotations -> {OUT} and {OUT}.zip", flush=True)
