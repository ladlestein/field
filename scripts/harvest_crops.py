"""Harvest player torso crops from each covered play's representative frame.

Turns the sweep's label table into training images: for every play in
data/harvest/plays.csv, run player detection on the representative pre-snap
frame and cut each detected player's upper-torso band to disk. Labels stay
play-level (the number multisets in plays.csv, joined by play_id) -- no crop
is claimed to show any particular number; that pairing is the trainer's job.

The broadcast often cuts to closeups during the pre-snap window, so the
sweep's "latest play-clock frame" is not reliably a wide formation shot.
Each play's frame is therefore chosen by detection census: run the player
detector over the play's last few aligned pre-snap frames and keep the one
with the most valid player boxes -- a formation shot wins that count over a
closeup every time. Plays with no play-clock frames at all fall back to
their other aligned frames and are flagged low_quality (those labels can
belong to a neighboring play).

Crop policy is versioned in the output path (crops/v0/...). v0 is the crude
fixed-fraction band from experiment 5 (upper 8-50% of the detection box,
full width), known to clip numbers on bent players; it costs yield, not
label correctness. Re-cutting under a better policy is cheap by design.

Usage: .venv/bin/python scripts/harvest_crops.py [--policy v0]
Outputs: data/harvest/crops/<policy>/<play_id>_<k>.jpg + crops.parquet
"""
import argparse
from pathlib import Path

import cv2
import polars as pl
from ultralytics import YOLO

from detect_field import detect_players

FRAMES_DIR = Path("data/harvest/frames")
HARVEST_DIR = Path("data/harvest")

MIN_BOX_HEIGHT = 45   # px; smaller than this and no number survives anyway
MAX_FOOT_Y = 1015     # exclude detections overlapping the score bug
BAND = (0.08, 0.50)   # v0: fixed upper-torso fraction of the box height
MAX_CANDIDATES = 6    # pre-snap frames to census per play


def torso_band(img, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h = y2 - y1
    band = img[y1 + int(BAND[0] * h):y1 + int(BAND[1] * h), x1:x2]
    return band if band.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="v0")
    args = ap.parse_args()
    if args.policy != "v0":
        raise SystemExit(f"unknown crop policy {args.policy!r} (only v0 exists)")

    out_dir = HARVEST_DIR / "crops" / args.policy
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pl.read_parquet(HARVEST_DIR / "manifest.parquet").filter(
        pl.col("play_id").is_not_null()
    )
    yolo = YOLO("yolov8s.pt")

    def valid_players(img):
        return [
            p for p in detect_players(yolo, img)
            if (p["box"][3] - p["box"][1]) >= MIN_BOX_HEIGHT
            and p["foot_point"][1] <= MAX_FOOT_Y
        ]

    rows = []
    n_crops = 0
    for play_id, group in sorted(manifest.group_by("play_id"), key=lambda g: g[0][0]):
        play_id = int(play_id[0]) if isinstance(play_id, tuple) else int(play_id)
        presnap = group.filter(pl.col("play_clock"))
        low_quality = presnap.height == 0
        pool = (group if low_quality else presnap).sort("frame_idx").tail(MAX_CANDIDATES)

        best_players, best_idx = [], None
        for idx in pool["frame_idx"].to_list():
            img = cv2.imread(str(FRAMES_DIR / f"t_{idx:05d}.jpg"))
            if img is None:
                continue
            players = valid_players(img)
            if len(players) >= len(best_players):  # >= prefers later frames on ties
                best_players, best_idx, best_img = players, idx, img
        if best_idx is None or not best_players:
            print(f"play {play_id}: no usable frame among {pool.height} candidates")
            continue

        for k, p in enumerate(best_players):
            band = torso_band(best_img, p["box"])
            if band is None:
                continue
            name = f"{play_id}_{k:02d}.jpg"
            cv2.imwrite(str(out_dir / name), band, [cv2.IMWRITE_JPEG_QUALITY, 95])
            x1, y1, x2, y2 = p["box"]
            rows.append({
                "play_id": play_id,
                "frame_idx": best_idx,
                "low_quality": low_quality,
                "crop": name,
                "det_conf": p["conf"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "foot_x": p["foot_point"][0], "foot_y": p["foot_point"][1],
            })
            n_crops += 1

    meta = pl.DataFrame(rows)
    meta.write_parquet(out_dir / "crops.parquet")
    per_play = meta.group_by("play_id").len()
    n_plays = manifest["play_id"].n_unique()
    print(f"plays processed: {per_play.height} / {n_plays}")
    print(f"  low-quality (no play-clock frame): "
          f"{meta.filter(pl.col('low_quality'))['play_id'].n_unique()}")
    print(f"crops written: {n_crops} (mean {n_crops / max(per_play.height, 1):.1f}/play)")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
