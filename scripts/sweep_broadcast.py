"""Sweep the full broadcast: align every bug-bearing frame to its nflverse play.

Walks 1fps frames (data/harvest/frames/t_NNNNN.jpg, extracted via ffmpeg),
OCRs the score bug on each, aligns to play-by-play, and writes a manifest.
Then collapses frames to one representative pre-snap frame per play (the
latest aligned frame whose bug still shows a play clock -- the play clock
vanishes at the snap, so that frame is closest to the set formation) and
reports coverage stats against the game's real snap count.

Usage:
  .venv/bin/python scripts/sweep_broadcast.py [--game GAME_ID] [--limit N]
      [--start-idx N] [--tag NAME]
Outputs (per game, under data/games/<game_id>/):
  manifest.parquet   one row per swept frame
  plays.csv          one row per covered play (representative frame)
--tag suffixes the output filenames (manifest_<tag>.parquet) so partial or
experimental sweeps can't clobber the real ones.
"""
import argparse
import re
import time

import cv2
import easyocr
import polars as pl

from game import Game, add_game_arg, easyocr_gpu
from scorebug_align import group_lines, match_play, parse_bug
# Tight crop around the FOX bug (both bars); smaller region + 2x scale keeps
# per-frame OCR fast enough to sweep ~7400 frames.
ROI = (450, 845, 1500, 1030)  # x1, y1, x2, y2
OCR_SCALE = 2
OCR_ALLOWLIST = "0123456789:&STNDRHOALGT OWNBAL"


def ocr_roi_tokens(reader, img):
    x1, y1, x2, y2 = ROI
    strip = cv2.resize(img[y1:y2, x1:x2], None, fx=OCR_SCALE, fy=OCR_SCALE,
                       interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(strip, allowlist=OCR_ALLOWLIST)
    tokens = []
    for quad, text, conf in results:
        xs = [p[0] / OCR_SCALE + x1 for p in quad]
        ys = [p[1] / OCR_SCALE + y1 for p in quad]
        tokens.append({
            "text": text.strip().upper().replace("I", "1"),
            "x": min(xs), "y": min(ys),
            "cx": sum(xs) / 4, "cy": sum(ys) / 4,
            "conf": conf,
        })
    return tokens


def has_play_clock(tokens):
    """A 1-2 digit token sitting to the right on the down-&-distance line."""
    for line in group_lines(tokens):
        joined = " ".join(t["text"] for t in line)
        m = re.search(r"\b[1234](?:ST|ND|RD|TH)\s*&\s*(?:\d{1,2}|GOAL)\b", joined)
        if not m:
            continue
        tail = line[-1]["text"]
        tail_digits = re.sub(r"\D", "", tail)
        if tail is not line[0]["text"] and not tail.endswith(m.group(0).split()[-1]) \
                and 1 <= len(tail_digits) <= 2:
            return True
        # Play clock often OCRs fused with a leading colon-artifact ("813",
        # "{22"); accept a standalone short-digit token at the line's end.
        if re.fullmatch(r"[:{'.8]?\d{1,2}", tail) and not joined.endswith(m.group(0)):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    add_game_arg(ap)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-idx", type=int, default=1)
    ap.add_argument("--tag", default=None,
                    help="suffix output filenames instead of overwriting the real ones")
    args = ap.parse_args()
    game = Game(args.game)
    suffix = f"_{args.tag}" if args.tag else ""
    if (args.limit or args.start_idx > 1) and not args.tag:
        raise SystemExit("partial sweeps must use --tag so they can't clobber the full outputs")

    frames = sorted(game.frames_dir.glob("t_*.jpg"))
    frames = [f for f in frames if int(f.stem.split("_")[1]) >= args.start_idx]
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        raise SystemExit(f"no frames found in {game.frames_dir}")

    pbp = game.pbp()
    part = game.participation()
    reader = easyocr.Reader(["en"], gpu=easyocr_gpu(), verbose=False)

    rows = []
    t0 = time.time()
    for n, fpath in enumerate(frames):
        idx = int(fpath.stem.split("_")[1])
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        tokens = ocr_roi_tokens(reader, img)
        state = parse_bug(tokens)
        play, note = (None, "no bug state") if state["clock"] is None else \
            match_play(pbp, state)
        rows.append({
            "frame_idx": idx,
            "t_sec": idx - 1,
            "n_tokens": len(tokens),
            "clock": state["clock"],
            "qtr": state["qtr"],
            "down": state["down"],
            "ydstogo": state["ydstogo"],
            "play_clock": has_play_clock(tokens),
            "play_id": None if play is None else int(play["play_id"]),
            "play_time": None if play is None else play["time"],
            "note": note if play is None else note or None,
        })
        if (n + 1) % 100 == 0:
            rate = (n + 1) / (time.time() - t0)
            print(f"{n+1}/{len(frames)} frames, {rate:.2f} fps, "
                  f"{sum(1 for r in rows if r['play_id'] is not None)} aligned",
                  flush=True)

    manifest = pl.DataFrame(rows)
    game.dir.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(game.dir / f"manifest{suffix}.parquet")

    # Representative pre-snap frame per play: latest aligned frame that still
    # shows a play clock; fall back to latest aligned frame at all.
    aligned = manifest.filter(pl.col("play_id").is_not_null())
    reps = (
        aligned.sort(["play_id", "play_clock", "frame_idx"])
        .group_by("play_id", maintain_order=True)
        .agg(
            pl.col("frame_idx").last().alias("rep_frame_idx"),
            pl.col("play_clock").last().alias("rep_has_play_clock"),
            pl.len().alias("n_frames"),
        )
    )
    numbers = part.select(
        pl.col("play_id").cast(pl.Int64),
        "offense_numbers", "defense_numbers",
    )
    reps = reps.join(numbers, on="play_id", how="left").sort("play_id")
    reps.write_csv(game.dir / f"plays{suffix}.csv")

    part_ids = set(part["play_id"].cast(pl.Int64).to_list())
    in_part = reps.filter(pl.col("play_id").is_in(list(part_ids)))
    print(f"\nframes swept: {manifest.height}")
    print(f"frames with bug state: {manifest.filter(pl.col('clock').is_not_null()).height}")
    print(f"frames aligned: {aligned.height}")
    print(f"participation plays covered: {in_part.height} / {part.height} "
          f"(+{reps.height - in_part.height} aligned to no-play rows)")
    print(f"  with play-clock representative: {in_part.filter(pl.col('rep_has_play_clock')).height}")


if __name__ == "__main__":
    main()
