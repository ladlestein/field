"""Render labeling sheets for the hand-verified eval set.

Samples crops from the v0 harvest (excluding low_quality plays, whose
set-level labels may belong to a neighboring play), tiles them into
contact sheets at 4x with a slot ID per crop, and writes a template CSV
to be filled with human-verified labels:

  visibility: full | partial | none | junk
    (junk = not a player: ref, sideline staff, blur blob)
  number: the digits actually readable (e.g. "72", "7?" for a visible
    tens digit only). Empty for none/junk.
  color: w | b (white or blue jersey) -- resolves which team's multiset
    the number must appear in.

Usage: .venv/bin/python scripts/make_eval_sheets.py --n 96 --per-sheet 12
Outputs: data/harvest/eval/sheet_NN.jpg, template.csv
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import polars as pl

CROPS_DIR = Path("data/harvest/crops/v0")
EVAL_DIR = Path("data/harvest/eval")
SCALE = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    meta = pl.read_parquet(CROPS_DIR / "crops.parquet").filter(~pl.col("low_quality"))
    rng = random.Random(args.seed)
    crops = rng.sample(meta["crop"].to_list(), min(args.n, meta.height))

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in range(0, len(crops), args.per_sheet):
        sheet_no = s // args.per_sheet
        batch = crops[s:s + args.per_sheet]
        tiles = []
        for j, name in enumerate(batch):
            img = cv2.imread(str(CROPS_DIR / name))
            img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
            tiles.append((f"{sheet_no:02d}{chr(65 + j)}", img))
            rows.append({"slot": f"{sheet_no:02d}{chr(65 + j)}", "crop": name,
                         "visibility": "", "number": "", "color": ""})
        th = max(t.shape[0] for _, t in tiles)
        tw = max(t.shape[1] for _, t in tiles)
        cols = 4
        nrows = (len(tiles) + cols - 1) // cols
        sheet = np.full((nrows * (th + 36), cols * (tw + 12), 3), 30, np.uint8)
        for k, (slot, t) in enumerate(tiles):
            r, c = divmod(k, cols)
            y0, x0 = r * (th + 36) + 30, c * (tw + 12) + 6
            sheet[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
            cv2.putText(sheet, slot, (x0, y0 - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 255), 2)
        cv2.imwrite(str(EVAL_DIR / f"sheet_{sheet_no:02d}.jpg"), sheet)

    pl.DataFrame(rows).write_csv(EVAL_DIR / "template.csv")
    print(f"{len(crops)} crops across {sheet_no + 1} sheets -> {EVAL_DIR}")


if __name__ == "__main__":
    main()
