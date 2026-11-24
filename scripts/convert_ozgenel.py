import json, base64, zlib, io
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import shutil

ds  = Path(r"C:\Users\tjsss\OneDrive\Desktop\machine learning\crackLens\data\concrete-crack-segmentation-DatasetNinja\ds")
out = ds.parents[1] / "ozgenel_masks"          # data\ozgenel_masks, separate from the download
out.mkdir(exist_ok=True)

def decode_bitmap(b64):
    a = np.array(Image.open(io.BytesIO(zlib.decompress(base64.b64decode(b64)))))
    if a.ndim == 3:
        a = a[..., 3] if a.shape[-1] == 4 else a[..., 0]
    return a > 0

def load_mask(ann_path):
    ann = json.loads(ann_path.read_text())
    H, W = ann["size"]["height"], ann["size"]["width"]
    full = np.zeros((H, W), bool)
    for o in ann["objects"]:
        if o["geometryType"] != "bitmap":
            continue
        x, y = o["bitmap"]["origin"]
        m = decode_bitmap(o["bitmap"]["data"])
        full[y:y + m.shape[0], x:x + m.shape[1]] |= m
    return full, len(ann["objects"])

rows = []
for ap in sorted((ds / "ann").glob("*.json")):
    photo = ds / "img" / ap.name[:-5]
    try:
        mask, n_obj = load_mask(ap)
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / f"{photo.stem}.png")
        with Image.open(photo) as im:
            pw, ph = im.size
            exif = im.getexif().get(274)
        rows.append(dict(name=photo.name, photo_w=pw, photo_h=ph, size_match=(ph, pw) == mask.shape,
                         exif=exif, objects=n_obj, crack_pct=round(100 * mask.mean(), 2)))
    except Exception as e:
        print("FAILED", ap.name, repr(e))

df_oz = pd.DataFrame(rows)
print(len(df_oz), "converted |", len(list(out.glob("*.png"))), "png files")
print(df_oz.size_match.value_counts().to_dict(), df_oz.exif.value_counts(dropna=False).to_dict())
print(df_oz.groupby(["photo_w", "photo_h"]).size())
print(df_oz.crack_pct.describe(percentiles=[.05, .5, .95]))



for p in sorted((ds/'img').iterdir()):
    if p.suffix.lower() == '.jpg':
        shutil.copy2(p, out / p.name) 
        
print(len(list(out.glob("*.jpg"))), len(list(out.glob("*.png"))))