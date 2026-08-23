"""Render contact sheets of pseudo-labels for eyeball verification.

Every label-quality bug found so far (eval v0, pseudo v1) was invisible in
the aggregate statistics and obvious in a twelve-crop contact sheet. Label
counts and confidence histograms cannot tell you that a crop of #78 is
labelled "8" -- only looking can. So looking is part of the pipeline, not a
debugging afterthought.

Each tile shows the accepted crop with the text region that produced the
label boxed in red, captioned with the number. Sample by resolution
("single" vs "pair") to audit the two failure modes separately: single-digit
labels are where truncation and neighbour bleed land.

Usage:
  .venv/bin/python3 scripts/audit_labels.py [--game GAME_ID] [--policy v2]
      [--resolution single|pair] [--n 24] [--seed 0]
Writes sheets to the policy dir as audit_<resolution>_<k>.jpg.
"""
import argparse

import cv2
import numpy as np
import polars as pl

from game import Game, add_game_arg

TILE_H, TILE_W = 300, 210
PER_ROW = 6


def tile(crop_path, row):
    img = cv2.imread(str(crop_path))
    if img is None:
        return None
    if row.get("rx1") is not None:
        cv2.rectangle(img, (int(row["rx1"]), int(row["ry1"])),
                      (int(row["rx2"]), int(row["ry2"])),
                      (0, 0, 255), max(2, img.shape[1] // 140))
    h, w = img.shape[:2]
    s = min(TILE_W / w, TILE_H / h)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    canvas = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    canvas[:img.shape[0], :img.shape[1]] = img
    cv2.putText(canvas, str(row["number"]), (4, TILE_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_game_arg(ap)
    ap.add_argument("--policy", default="v2")
    ap.add_argument("--resolution", default=None,
                    help="filter to one resolution (single / pair)")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    game = Game(args.game)
    pdir = game.pseudo_dir(args.policy)
    labels = pl.read_parquet(pdir / "labels.parquet")
    if args.resolution and "resolution" in labels.columns:
        labels = labels.filter(pl.col("resolution") == args.resolution)
    if labels.height == 0:
        raise SystemExit("no labels match")

    n = min(args.n, labels.height)
    sample = labels.sample(n, seed=args.seed).to_dicts()
    tiles = [t for t in (tile(pdir / r["crop"], r) for r in sample)
             if t is not None]

    tag = args.resolution or "all"
    written = []
    for k, start in enumerate(range(0, len(tiles) - len(tiles) % PER_ROW,
                                    PER_ROW * 2)):
        chunk = tiles[start:start + PER_ROW * 2]
        rows = [np.hstack(chunk[i:i + PER_ROW])
                for i in range(0, len(chunk) - len(chunk) % PER_ROW, PER_ROW)]
        if not rows:
            continue
        out = pdir / f"audit_{tag}_{k}.jpg"
        cv2.imwrite(str(out), np.vstack(rows))
        written.append(out)

    print(f"{game.game_id} / {args.policy}: {labels.height} labels"
          + (f" with resolution={args.resolution}" if args.resolution else ""))
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
