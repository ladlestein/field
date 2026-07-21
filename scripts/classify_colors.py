"""Classify harvested torso crops by team jersey color.

Classical HSV color classifier -- no ML. For each crop, takes a central
sub-region (avoiding the grass/skin-heavy edges of the fixed-fraction v0
torso band), masks out pixels that look like grass or skin, and takes the
median HSV of what's left as a robust dominant-color estimate. That estimate
is scored against each of the game's two reference jersey colors (a small
built-in table keyed by game_id, since jersey colors are away/home and
season-dependent -- WAS wore white in week 15 but burgundy in week 1) and
classified as whichever team scores higher, or "other" if neither scores
above a floor (referees, sideline staff, skin/blur-dominated boxes, dark
jackets).

Usage: .venv/bin/python scripts/classify_colors.py [--game GAME_ID] [--policy v0]
Outputs: data/games/<game_id>/crops/<policy>/color.parquet
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from game import Game, add_game_arg

EVAL_LABELS_PATH = Path("eval/labels_v0.csv")

# Per-game slot -> (team, color "kind") table. The "w"/"b" slot names are
# kept fixed to match the week-15 hand-labeled eval set's convention (where
# they happen to spell out literal white/blue); for another game they're just
# stable labels for "team 1"/"team 2" -- what a slot *means* colorwise is the
# "kind" field, which KIND_SCORERS below knows how to score.
#
# 2025_01_NYG_WAS verified directly from a broadcast frame
# (data/games/2025_01_NYG_WAS/frames/t_01952.jpg): WAS (home) wore burgundy
# jerseys with white pants, NYG (road) wore blue jerseys -- NOT
# white-vs-blue as first guessed. Corrected here rather than left TODO,
# since that game's crops already exist.
GAME_COLORS = {
    "2025_15_WAS_NYG": {
        "w": {"team": "WAS", "kind": "white"},
        "b": {"team": "NYG", "kind": "blue"},
    },
    "2025_01_NYG_WAS": {
        "w": {"team": "NYG", "kind": "blue"},
        "b": {"team": "WAS", "kind": "burgundy"},
    },
}

OTHER_FLOOR = 0.18  # min winning score to avoid "other"


def circ_dist(h, center, period=180):
    """Circular distance between hue h (array) and center, on a 0..period wheel."""
    d = np.abs(h - center)
    return np.minimum(d, period - d)


def dominant_hsv(bgr):
    """Median HSV of the crop's central region, excluding grass/skin pixels.

    Returns (h, s, v, kept_frac). Falls back to the unmasked central region
    if masking leaves too few pixels (e.g. a crop that's genuinely all grass
    or all skin -- a bad detection, which should score low everywhere and
    land in "other" anyway).
    """
    h, w = bgr.shape[:2]
    y0, y1 = int(h * 0.12), int(h * 0.92)
    x0, x1 = int(w * 0.12), int(w * 0.88)
    region = bgr[y0:y1, x0:x1] if y1 > y0 and x1 > x0 else bgr
    if region.size == 0:
        region = bgr

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    grass = (H >= 28) & (H <= 95) & (S >= 50) & (V >= 35)
    skin = (circ_dist(H, 8) <= 12) & (S >= 25) & (S <= 165) & (V >= 70)
    keep = ~grass & ~skin

    if keep.sum() < max(10, 0.05 * len(H)):
        keep = np.ones_like(keep)

    Hk, Sk, Vk = H[keep], S[keep], V[keep]
    return float(np.median(Hk)), float(np.median(Sk)), float(np.median(Vk)), float(keep.mean())


def clip01(x):
    return max(0.0, min(1.0, x))


def score_white(h, s, v):
    # White jerseys are defined by *low saturation*, not raw brightness --
    # on real broadcast crops a white jersey in shadow reads as low-S,
    # medium-V, and hue is meaningless noise at low S so it isn't used here.
    # Thresholds grid-searched against eval/labels_v0.csv's 81 color-labeled
    # rows (labeled white crops: median s=30, 90th-pctile s=46; labeled blue
    # crops: s>=48 at the 10th percentile) -- 55/15 was the best cutoff/span.
    # v floor just guards against near-black "other" junk.
    return clip01((55 - s) / 15) * clip01((v - 70) / 60)


def score_blue(h, s, v):
    # Hue centered in the blue band, with enough saturation to not be a
    # washed-out white jersey or a grass/skin remnant that survived masking.
    # Grid-searched against the same eval rows (labeled blue crops: median
    # s=79, h clustered 115-123) -- hue is the load-bearing term here, so the
    # saturation floor can be permissive (5) without hurting precision.
    hue_term = clip01(1 - circ_dist(np.array([h]), 118)[0] / 15)
    return hue_term * clip01((s - 5) / 40)


def score_burgundy(h, s, v):
    # Saturated red/maroon: hue near the wheel's red end (0 or 179). No
    # value term -- an earlier version added a ceiling assuming burgundy
    # always reads dark, but spot-checking 2025_01_NYG_WAS crops (no numeric
    # eval set exists for this game yet) showed clearly-burgundy jerseys in
    # bright sun at v~170-200 getting wrongly starved into "other"; grass
    # and skin are already masked out upstream, so hue+saturation alone
    # separate burgundy well without it. NOT calibrated against hand labels
    # -- reasoned by analogy to score_blue; revisit if this game gets an
    # eval set.
    hue_term = clip01(1 - circ_dist(np.array([h]), 175)[0] / 35)
    return hue_term * clip01((s - 25) / 100)


KIND_SCORERS = {
    "white": score_white,
    "blue": score_blue,
    "burgundy": score_burgundy,
}


def classify(h, s, v, slots):
    """slots: dict of slot_label -> {"team":.., "kind":..}. Returns (color, confidence, scores dict)."""
    scores = {label: KIND_SCORERS[info["kind"]](h, s, v) for label, info in slots.items()}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_label, top_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    if top_score < OTHER_FLOOR:
        color = "other"
        confidence = clip01(1 - top_score / OTHER_FLOOR)
    else:
        color = top_label
        margin = top_score - runner_up
        confidence = clip01(margin / max(top_score, 1e-6))
    return color, confidence, scores


def validate(out, crops_dir):
    """If a hand-labeled eval set covers this crops dir, print accuracy + disagreements.

    eval/labels_v0.csv is the week-15 v0-policy hand-labeled set (96 crops,
    color column populated for the 81 non-junk rows). Silently skips if the
    label file is absent or none of its crop filenames are present here --
    i.e. this is a no-op for games/policies it doesn't cover.
    """
    if not EVAL_LABELS_PATH.exists():
        return
    labels = [r for r in csv.DictReader(open(EVAL_LABELS_PATH)) if r["color"]]
    if not labels or not (crops_dir / labels[0]["crop"]).exists():
        return

    pred = {r["crop"]: r for r in out.to_dicts()}
    correct, disagreements = 0, []
    for r in labels:
        p = pred.get(r["crop"])
        if p is None:
            continue
        if p["color"] == r["color"]:
            correct += 1
        else:
            disagreements.append((r["crop"], r["color"], p["color"], p["confidence"]))

    n = len(labels)
    print(f"\neval/labels_v0.csv accuracy: {correct}/{n} = {correct/n:.3f}")
    if disagreements:
        print("disagreements (crop, labeled, predicted, confidence):")
        for d in disagreements:
            print(f"  {d[0]:16s} labeled={d[1]:5s} predicted={d[2]:5s} conf={d[3]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    add_game_arg(ap)
    ap.add_argument("--policy", default="v0")
    args = ap.parse_args()
    game = Game(args.game)

    if args.game not in GAME_COLORS:
        raise SystemExit(
            f"no reference jersey colors for game {args.game!r}; add an entry to "
            f"GAME_COLORS in scripts/classify_colors.py"
        )
    slots = GAME_COLORS[args.game]

    crops_dir = game.crops_dir(args.policy)
    crops = pl.read_parquet(crops_dir / "crops.parquet")

    rows = []
    for crop_name, low_quality in crops.select(["crop", "low_quality"]).iter_rows():
        img = cv2.imread(str(crops_dir / crop_name))
        if img is None:
            rows.append({
                "crop": crop_name, "color": "other", "confidence": 0.0,
                "h": None, "s": None, "v": None, "kept_frac": None,
                "low_quality": low_quality,
            })
            continue
        h, s, v, kept_frac = dominant_hsv(img)
        color, confidence, scores = classify(h, s, v, slots)
        row = {
            "crop": crop_name, "color": color, "confidence": confidence,
            "h": h, "s": s, "v": v, "kept_frac": kept_frac,
            "low_quality": low_quality,
        }
        for label in slots:
            row[f"score_{label}"] = scores[label]
        rows.append(row)

    out = pl.DataFrame(rows)
    out_path = crops_dir / "color.parquet"
    out.write_parquet(out_path)

    print(f"wrote {out.height} rows to {out_path}")
    print(out["color"].value_counts().sort("color"))
    for label, info in slots.items():
        print(f"  slot {label!r} = {info['team']} ({info['kind']})")

    validate(out, crops_dir)


if __name__ == "__main__":
    main()
