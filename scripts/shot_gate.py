"""Shot-type gate: is this frame a wide pre-snap-style shot, a closeup, or junk?

The router everything depends on (EXPERIMENTS.md architectural conclusions):
formation/personnel inference only makes sense on wide shots that show the
whole formation; jersey-number reading only pays on closeup-scale players;
replays/graphics/crowd shots should reach neither. v0 is a rule set over the
YOLO detection census — no learned classifier, so every verdict is
explainable by its features, and the thresholds are cheap to re-tune when
the contact sheet says they're wrong.

Verdicts:
  wide     many field-scale players, nobody huge — formation-analysis input
  closeup  at least one large player box — jersey-evidence input
  other    neither (crowd, graphics, booth, extreme zoom, empty field)

"wide" here means camera scale, not game phase: a wide shot during a live
play still gates as wide. Pre-snap-ness is deliberately NOT decided here —
it comes from the play clock's presence in the score bug, which is an
on-screen observable (sweep_broadcast.has_play_clock over the bug OCR
tokens) and therefore available for live games too, no nflverse data
needed. The manifest merely caches those reads for recorded games; a live
engine gets the same signal from its per-tick bug OCR.

CLI (validation): sample frames from a game's manifest, classify each,
write per-verdict contact sheets to eval/shot_gate/ for eyeballing.

Usage:
  .venv/bin/python3 scripts/shot_gate.py [--game ID] [--sample N] [--seed S]
"""
import argparse
import random

import cv2
import polars as pl

from detect_field import detect_players
from game import Game, add_game_arg

# Feature thresholds (1080p broadcast). Heights in px. Tuned against the
# feature distribution of the 125 play-clock representative frames in
# plays.csv (known formation shots): med_field_h runs 88-137 (q05-q95),
# n_field 9-34, and 10% of true wides carry a >=300px foreground body.
FIELD_SCALE = (28, 155)   # a player standing on the field in a wide shot
BIG = 300                 # a player this tall means the camera is close
MAX_FOOT_Y = 1015         # ignore detections overlapping the score bug
WIDE_MIN_PLAYERS = 9      # formations put 22 on screen; occlusion eats some
WIDE_MAX_MEDIAN_H = 150
WIDE_MAX_BIG = 2          # tolerate a couple of near-camera sideline bodies
WIDE_MIN_GRASS = 0.30     # crowd/bench shots have many bodies but no turf


def grass_fraction(img):
    """Fraction of green-turf pixels in the lower two-thirds of the frame."""
    import numpy as np

    lower = img[img.shape[0] // 3:]
    hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90)
        & (hsv[:, :, 1] >= 50) & (hsv[:, :, 2] >= 40)
    )
    return float(np.mean(mask))


def shot_features(players, img=None):
    heights = sorted(
        p["box"][3] - p["box"][1]
        for p in players if p["foot_point"][1] <= MAX_FOOT_Y
    )
    field = [h for h in heights if FIELD_SCALE[0] <= h <= FIELD_SCALE[1]]
    return {
        "n_det": len(heights),
        "n_field": len(field),
        "n_big": sum(1 for h in heights if h >= BIG),
        "med_field_h": field[len(field) // 2] if field else None,
        "max_h": heights[-1] if heights else None,
        "grass": grass_fraction(img) if img is not None else None,
    }


def classify_shot(players, img=None):
    """Rule-based verdict over the detection census. Returns (kind, features)."""
    f = shot_features(players, img)
    if (
        f["n_field"] >= WIDE_MIN_PLAYERS
        and f["n_big"] <= WIDE_MAX_BIG
        and f["med_field_h"] is not None
        and f["med_field_h"] <= WIDE_MAX_MEDIAN_H
        and (f["grass"] is None or f["grass"] >= WIDE_MIN_GRASS)
    ):
        return "wide", f
    if f["n_big"] >= 1:
        return "closeup", f
    return "other", f


# --- validation CLI ---------------------------------------------------------

THUMB_W, THUMB_H = 320, 180
SHEET_COLS = 6


def contact_sheet(entries):
    """Grid of (img, caption) thumbnails."""
    import numpy as np

    rows = (len(entries) + SHEET_COLS - 1) // SHEET_COLS
    sheet = np.zeros((rows * (THUMB_H + 18), SHEET_COLS * THUMB_W, 3), np.uint8)
    for i, (img, caption) in enumerate(entries):
        r, c = divmod(i, SHEET_COLS)
        y, x = r * (THUMB_H + 18), c * THUMB_W
        sheet[y:y + THUMB_H, x:x + THUMB_W] = cv2.resize(img, (THUMB_W, THUMB_H))
        cv2.putText(sheet, caption, (x + 4, y + THUMB_H + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return sheet


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    add_game_arg(ap)
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--seed", type=int, default=15)
    args = ap.parse_args()
    game = Game(args.game)

    manifest = pl.read_parquet(game.manifest_path)
    rng = random.Random(args.seed)
    idxs = sorted(rng.sample(manifest["frame_idx"].to_list(),
                             min(args.sample, manifest.height)))

    yolo = YOLO("yolov8s.pt")
    out_dir = game.eval_dir / "shot_gate"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_kind = {"wide": [], "closeup": [], "other": []}
    for idx in idxs:
        img = cv2.imread(str(game.frames_dir / f"t_{idx:05d}.jpg"))
        if img is None:
            continue
        kind, f = classify_shot(detect_players(yolo, img), img)
        caption = (f"t_{idx:05d} field={f['n_field']} big={f['n_big']} "
                   f"medh={f['med_field_h'] or 0:.0f} maxh={f['max_h'] or 0:.0f} "
                   f"grass={f['grass']:.2f}")
        by_kind[kind].append((img, caption))

    for kind, entries in by_kind.items():
        print(f"{kind}: {len(entries)}")
        if entries:
            path = out_dir / f"{kind}.jpg"
            cv2.imwrite(str(path), contact_sheet(entries),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"  sheet: {path}")


if __name__ == "__main__":
    main()
