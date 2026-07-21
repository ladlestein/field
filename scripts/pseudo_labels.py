"""Closeup pseudo-label harvester.

Generate per-crop jersey-number training labels by running the CRAFT+PARSeq
reading pipeline (accurate on large/closeup crops -- EXPERIMENTS.md entry 3)
over every aligned broadcast frame, keeping only reads that:

  * decode to a 1- or 2-digit string,
  * put >0.8 digit mass on every digit position,
  * read with mean confidence >0.8, and
  * name a number that actually appears in that play's known 22-player
    offense+defense number multiset (from plays.csv).

The multiset cross-check is the load-bearing guard: recognizer confidence is
untrustworthy out of distribution (it hallucinates digits from blur), so a
read is only trusted when an independent, out-of-band constraint agrees.

Wide-shot players (50-150px tall) are excluded via --min-box-height; PARSeq is
unreliable at that scale (experiments 3 and 5). This harvester deliberately
targets the closeup/medium regime where the reader is trustworthy, and uses
the roster multiset to catch the residual mistakes.

Outputs under the game dir:
  pseudo/<frame_idx>_<detection_k>.jpg   accepted torso crops (native res)
  pseudo/labels.parquet                  one row per accepted read

Usage:
  .venv/bin/python3 scripts/pseudo_labels.py [--game GAME_ID]
      [--limit N] [--min-box-height PX]
"""
import argparse
import logging
from collections import Counter

import cv2
import easyocr
import polars as pl
from ultralytics import YOLO

from game import Game, add_game_arg, easyocr_gpu
from detect_field import detect_players
from read_jerseys import torso_crop
from localize_recognize import (
    load_parseq,
    localize_text_regions,
    digit_distributions,
)

# Quiet ultralytics' per-frame prediction chatter (detect_players uses
# verbose=True); over thousands of frames it would bury the run log.
logging.getLogger("ultralytics").setLevel(logging.ERROR)
try:
    from ultralytics.utils import LOGGER as _ULTRA_LOGGER

    _ULTRA_LOGGER.setLevel(logging.ERROR)
except Exception:
    pass

DIGIT_SHARE_MIN = 0.8
CONF_MIN = 0.8


def play_number_sets(plays: pl.DataFrame) -> dict[int, set[str]]:
    """play_id -> set of jersey-number tokens (offense + defense combined)."""
    out: dict[int, set[str]] = {}
    for row in plays.iter_rows(named=True):
        nums: set[str] = set()
        for col in ("offense_numbers", "defense_numbers"):
            val = row.get(col)
            if val:
                nums.update(tok for tok in str(val).split(";") if tok)
        out[int(row["play_id"])] = nums
    return out


def evaluate_read(text, conf, dists, number_set):
    """Classify a single PARSeq read.

    Returns (status, info). status is one of:
      "format"      -- not a clean 1-2 digit read
      "digit_share" -- a digit position under the digit-share floor
      "conf"        -- mean confidence under the floor
      "multiset"    -- number not in the play's roster multiset
      "accept"      -- passed everything
    info carries (number, mean_conf, digit_share_min) when computable.
    """
    text = (text or "").strip()
    if not (1 <= len(text) <= 2 and text.isdigit()):
        return "format", None

    # Digit positions actually decoded (dists parallels the decoded string;
    # a None entry means the position put ~no mass on digits at all).
    digit_dists = dists[: len(text)]
    if len(digit_dists) < len(text) or any(d is None for d in digit_dists):
        return "digit_share", None
    shares = [d["digit_share"] for d in digit_dists]
    ds_min = min(shares)

    mean_conf = float(conf.mean().item())

    if ds_min <= DIGIT_SHARE_MIN:
        return "digit_share", (text, mean_conf, ds_min)
    if mean_conf <= CONF_MIN:
        return "conf", (text, mean_conf, ds_min)
    if text not in number_set:
        return "multiset", (text, mean_conf, ds_min)
    return "accept", (text, mean_conf, ds_min)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_game_arg(ap)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N aligned frames")
    ap.add_argument("--min-box-height", type=int, default=180,
                    help="keep detections at least this tall (px); wide-shot "
                         "players (50-150px) are excluded (default 180)")
    ap.add_argument("--after-frame", type=int, default=None,
                    help="only process frames with frame_idx > N (skip a hang)")
    ap.add_argument("--resume", action="store_true",
                    help="keep existing labels.parquet rows and continue past "
                         "the highest frame_idx recorded in it")
    args = ap.parse_args()

    game = Game(args.game)
    manifest = pl.read_parquet(game.manifest_path)
    plays = pl.read_csv(game.plays_path)
    number_sets = play_number_sets(plays)

    aligned = (
        manifest.filter(pl.col("play_id").is_not_null())
        .select("frame_idx", "play_id")
        .sort("frame_idx")
    )
    if args.limit is not None:
        aligned = aligned.head(args.limit)

    out_dir = game.dir / "pseudo"
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_rows = []
    after_frame = args.after_frame
    if args.resume and (out_dir / "labels.parquet").exists():
        existing = pl.read_parquet(out_dir / "labels.parquet")
        resume_rows = existing.to_dicts()
        resumed_from = int(existing["frame_idx"].max())
        after_frame = max(after_frame or 0, resumed_from)
        print(f"resuming: {len(resume_rows)} existing labels, "
              f"continuing after frame {after_frame}")
    if after_frame is not None:
        aligned = aligned.filter(pl.col("frame_idx") > after_frame)

    yolo = YOLO("yolov8s.pt")
    reader = easyocr.Reader(["en"], gpu=easyocr_gpu(), verbose=False)
    parseq, preprocess = load_parseq()

    rows = list(resume_rows)
    n_frames = 0
    n_closeup = 0
    n_raw_reads = 0
    rej = Counter()  # format / digit_share / conf / multiset
    accepted_numbers = Counter()
    MAX_REGIONS_PER_CROP = 8  # a jersey torso has <=2-3 text regions; more
    # means the "player" is a text-dense graphic/overlay false positive

    def flush():
        if not rows:
            return
        pl.DataFrame(rows).write_parquet(out_dir / "labels.parquet")

    total = aligned.height
    for prog, rec in enumerate(aligned.iter_rows(named=True), start=1):
        frame_idx = int(rec["frame_idx"])
        play_id = int(rec["play_id"])
        number_set = number_sets.get(play_id, set())

        frame_path = game.frames_dir / f"t_{frame_idx:05d}.jpg"
        # Heartbeat: overwritten every frame, so a hang is diagnosable to the
        # exact frame (and skippable with --after-frame).
        (out_dir / "progress.txt").write_text(f"{frame_idx}\n")
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"  [skip] missing frame {frame_path}")
            continue
        n_frames += 1

        players = detect_players(yolo, img)
        # Deterministic order left-to-right so detection_k is reproducible.
        players = sorted(players, key=lambda p: p["foot_point"][0])
        closeups = [p for p in players
                    if (p["box"][3] - p["box"][1]) >= args.min_box_height]

        for k, p in enumerate(closeups):
            n_closeup += 1
            box = p["box"]
            box_height = float(box[3] - box[1])
            crop = torso_crop(img, box)
            if crop is None:
                continue
            regions = localize_text_regions(reader, crop)
            if len(regions) > MAX_REGIONS_PER_CROP:
                continue
            crop_saved = False
            crop_name = f"{frame_idx}_{k}.jpg"
            for (rx1, ry1, rx2, ry2) in regions:
                region = crop[ry1:ry2, rx1:rx2]
                if region.size == 0:
                    continue
                n_raw_reads += 1
                text, conf, dists = digit_distributions(parseq, preprocess, region)
                status, info = evaluate_read(text, conf, dists, number_set)
                if status != "accept":
                    rej[status] += 1
                    continue

                number, mean_conf, ds_min = info
                if not crop_saved:
                    cv2.imwrite(str(out_dir / crop_name), crop)
                    crop_saved = True
                x1, y1, x2, y2 = [int(v) for v in box]
                tens = int(number[0]) if len(number) == 2 else None
                units = int(number[-1])
                rows.append({
                    "crop": crop_name,
                    "frame_idx": frame_idx,
                    "play_id": play_id,
                    "number": number,
                    "tens": tens,
                    "units": units,
                    "read_conf": mean_conf,
                    "digit_share_min": ds_min,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "box_height": box_height,
                })
                accepted_numbers[number] += 1

        if prog % 100 == 0:
            flush()
            print(f"  ... {prog}/{total} frames | accepted={len(rows)} "
                  f"| closeup_det={n_closeup} raw_reads={n_raw_reads}",
                  flush=True)

    flush()

    # ---- stats ----
    print("\n=== pseudo-label harvest ===")
    print(f"game:                    {game.game_id}")
    print(f"min box height:          {args.min_box_height}px")
    print(f"aligned frames selected: {total}"
          + (f" (--limit {args.limit})" if args.limit is not None else ""))
    print(f"frames processed:        {n_frames}")
    print(f"closeup-scale dets:      {n_closeup}")
    print(f"raw PARSeq reads:        {n_raw_reads}")
    print(f"rejected (format 1-2dig):{rej['format']}")
    print(f"rejected (digit_share):  {rej['digit_share']}")
    print(f"rejected (conf):         {rej['conf']}")
    print(f"rejected (multiset):     {rej['multiset']}")
    print(f"accepted labels:         {len(rows)}")
    print(f"distinct numbers:        {len(accepted_numbers)}")
    if accepted_numbers:
        print("\nlabels per number:")
        for num, cnt in sorted(accepted_numbers.items(),
                               key=lambda t: (-t[1], int(t[0]))):
            bar = "#" * min(cnt, 60)
            print(f"  {num:>2}: {cnt:>4}  {bar}")
    print(f"\noutput: {out_dir}/labels.parquet + crops", flush=True)


if __name__ == "__main__":
    main()
