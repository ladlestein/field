"""Closeup pseudo-label harvester.

Generate per-crop jersey-number training labels by running the CRAFT+PARSeq
reading pipeline (accurate on large/closeup crops -- EXPERIMENTS.md entry 3)
over every aligned broadcast frame, keeping only reads that:

  * sit on the primary player, not a neighbour intruding at the crop's edge,
  * decode to a 1- or 2-digit string,
  * put >0.8 digit mass on every digit position,
  * read with mean confidence >0.8,
  * survive the truncation test if they are a single digit, and
  * name a number that actually appears in that play's known 22-player
    offense+defense number multiset (from plays.csv).

The multiset cross-check is the load-bearing guard: recognizer confidence is
untrustworthy out of distribution (it hallucinates digits from blur), so a
read is only trusted when an independent, out-of-band constraint agrees.

The first and fifth conditions were added after auditing v1 of this harvest
(EXPERIMENTS.md entry 9): the multiset guard is strong for two-digit reads
but nearly toothless for single digits, which let two failure modes through
at ~20-25% of the single-digit labels. See on_primary_player() and
resolve_digits() for what each one tests and why.

A label must answer "what number is THIS player wearing?", not "is some
legible on-roster number visible somewhere in this picture?" -- the crop is
the model's input and the player box defines whose number is being asked
about. That is why a read's geometry matters as much as its content.

Wide-shot players (50-150px tall) are excluded via --min-box-height; PARSeq is
unreliable at that scale (experiments 3 and 5). This harvester deliberately
targets the closeup/medium regime where the reader is trustworthy, and uses
the roster multiset to catch the residual mistakes.

Outputs under the game dir:
  pseudo/<policy>/<frame_idx>_<detection_k>.jpg  accepted torso crops
  pseudo/<policy>/labels.parquet                 one row per accepted read

The policy directory is versioned for the same reason crop policies are: the
accept rules decide what a label means, so a run under new rules is a new
dataset, not an update to the old one. v1 is the pre-audit harvest.

Usage:
  .venv/bin/python3 scripts/pseudo_labels.py [--game GAME_ID]
      [--policy NAME] [--limit N] [--min-box-height PX]
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

# --- geometry guard (see on_primary_player) ---
CENTER_BAND = (0.12, 0.88)   # allowed horizontal centre of a region

# --- truncation test (see resolve_digits) ---
EXPAND_WIDTHS = 1.0     # how far to widen a single digit, in region widths
MIN_EXPAND_ROOM = 0.8   # region-widths of clear space needed to test a side

# CRAFT input cap. torso_crop() upscales 4x, so a closeup torso can exceed
# 2000px on a side -- pure interpolation, carrying no information the native
# pixels lack, and the oversized tensors are what aborts Metal without a
# traceback (EXPERIMENTS.md entry 8).
MAX_CRAFT_DIM = 1600


def craft_regions(reader, crop):
    """localize_text_regions with the oversized-crop cap applied."""
    h, w = crop.shape[:2]
    scale = min(1.0, MAX_CRAFT_DIM / max(h, w))
    if scale == 1.0:
        return localize_text_regions(reader, crop)
    small = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    inv = 1.0 / scale
    out = []
    for x1, y1, x2, y2 in localize_text_regions(reader, small):
        out.append((max(0, int(x1 * inv)), max(0, int(y1 * inv)),
                    min(w, int(x2 * inv)), min(h, int(y2 * inv))))
    return out


def on_primary_player(crop, region):
    """Does this text region belong to the player the crop is about?

    A torso crop is cut from one player's detection box, but in traffic a
    neighbour overlaps that box and a sliver of *their* number lands inside
    it -- typically clipped against the crop's left or right edge. Labelling
    the crop with that number teaches the model the wrong association: the
    picture is mostly one player and the label names another.

    A number worn by the primary player sits well inside their silhouette:
    chest and back numbers near the middle, shoulder numbers off-centre but
    still comfortably within the box. An intruding number is pinned against a
    side. So the test is purely where the region's centre falls.

    Note what this deliberately does *not* test: whether the region touches
    the crop's edge. An earlier version rejected those, and it threw away
    good labels -- on a tight detection box a large chest number legitimately
    runs to the boundary. Auditing the rejects showed the centre test alone
    catches the intruders and keeps those.
    """
    w = crop.shape[1]
    x1, _, x2, _ = region
    cx = (x1 + x2) / 2 / w
    return CENTER_BAND[0] <= cx <= CENTER_BAND[1]


def read_patch(model, preprocess, patch):
    """(text, mean_conf, min digit_share) for an image patch."""
    if patch is None or patch.size == 0:
        return "", 0.0, None
    text, conf, dists = digit_distributions(model, preprocess, patch)
    text = (text or "").strip()
    mean_conf = float(conf.mean().item())
    used = dists[: len(text)]
    if not text or len(used) < len(text) or any(d is None for d in used):
        return text, mean_conf, None
    return text, mean_conf, min(d["digit_share"] for d in used)


def clean_pair(text, conf, ds):
    """A confident, fully-digit two-digit read."""
    return (len(text) == 2 and text.isdigit() and conf > CONF_MIN
            and ds is not None and ds > DIGIT_SHARE_MIN)


def resolve_digits(model, preprocess, crop, region, digit):
    """Is a single-digit read the whole number, or half of a two-digit one?

    CRAFT sometimes boxes only one digit of a two-digit number. PARSeq then
    reads that digit cleanly and confidently, and the roster multiset barely
    filters it: a stray digit only has to match one of the ~3 single-digit
    numbers on the field, and those belong to quarterbacks, receivers and
    defensive backs -- exactly the players who dominate closeups. A crop of
    #78 comes out labelled "8".

    Rather than guess from geometry, test it. Widen the region by one digit
    width to the left, re-read; then to the right, re-read. A genuine single
    digit stays single on both sides, because there is only jersey fabric
    beside it. A truncated pair grows into a two-digit number on one side.

    The widened read is used only to *reject*, never to relabel. Recovering
    the two-digit number is tempting -- it turns a discard into a label -- but
    an audit of recovered numbers found roughly a third of them wrong: the
    widened patch is mostly fabric, and PARSeq will hallucinate a plausible
    second digit from a fold or seam ('1' especially). The roster multiset
    does not catch it, for the same reason it does not catch the truncation
    itself. So a digit-pair signal here means "this read is not trustworthy",
    and nothing more.

    Returns (number, reason); number is None when the read must be dropped.
    """
    h, w = crop.shape[:2]
    x1, y1, x2, y2 = region
    rw = x2 - x1
    if rw <= 0 or x1 / rw < MIN_EXPAND_ROOM or (w - x2) / rw < MIN_EXPAND_ROOM:
        # No room to look beside the digit, so a neighbouring digit cut off by
        # the crop edge can't be ruled out. Drop rather than guess.
        return None, "untestable"

    pad = int(EXPAND_WIDTHS * rw)
    for side, patch in (
        ("left", crop[y1:y2, max(0, x1 - pad):x2]),
        ("right", crop[y1:y2, x1:min(w, x2 + pad)]),
    ):
        text, conf, ds = read_patch(model, preprocess, patch)
        if not clean_pair(text, conf, ds):
            continue
        # The original digit must survive in the position the widening added
        # to; otherwise the wider patch is reading something else entirely.
        if (side == "left" and text.endswith(digit)) or \
           (side == "right" and text.startswith(digit)):
            return None, "truncated_pair"
    return digit, "single"


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


def evaluate_read(text, mean_conf, ds_min):
    """Quality gate on a single PARSeq read, before geometry or roster checks.

    Returns (status, info). status is one of:
      "format"      -- not a clean 1-2 digit read
      "digit_share" -- a digit position under the digit-share floor
      "conf"        -- mean confidence under the floor
      "ok"          -- passed the quality floors
    info carries (number, mean_conf, digit_share_min) when computable.
    """
    text = (text or "").strip()
    if not (1 <= len(text) <= 2 and text.isdigit()):
        return "format", None
    if ds_min is None:
        return "digit_share", None
    if ds_min <= DIGIT_SHARE_MIN:
        return "digit_share", (text, mean_conf, ds_min)
    if mean_conf <= CONF_MIN:
        return "conf", (text, mean_conf, ds_min)
    return "ok", (text, mean_conf, ds_min)


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
    ap.add_argument("--policy", default="v2",
                    help="output under pseudo/<policy>/; versioned because the "
                         "accept rules change what a label means (default v2)")
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

    out_dir = game.pseudo_dir(args.policy)
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
            regions = craft_regions(reader, crop)
            if len(regions) > MAX_REGIONS_PER_CROP:
                continue
            crop_name = f"{frame_idx}_{k}.jpg"
            crop_rows = []
            for (rx1, ry1, rx2, ry2) in regions:
                if not on_primary_player(crop, (rx1, ry1, rx2, ry2)):
                    rej["off_player"] += 1
                    continue
                region = crop[ry1:ry2, rx1:rx2]
                if region.size == 0:
                    continue
                n_raw_reads += 1
                text, mean_conf, ds_min = read_patch(parseq, preprocess, region)
                status, info = evaluate_read(text, mean_conf, ds_min)
                if status != "ok":
                    rej[status] += 1
                    continue

                number, mean_conf, ds_min = info
                if len(number) == 1:
                    number, reason = resolve_digits(
                        parseq, preprocess, crop, (rx1, ry1, rx2, ry2), number)
                    if number is None:
                        rej[reason] += 1
                        continue
                else:
                    reason = "pair"

                if number not in number_set:
                    rej["multiset"] += 1
                    continue

                x1, y1, x2, y2 = [int(v) for v in box]
                crop_rows.append({
                    "crop": crop_name,
                    "frame_idx": frame_idx,
                    "play_id": play_id,
                    "number": number,
                    "tens": int(number[0]) if len(number) == 2 else None,
                    "units": int(number[-1]),
                    "read_conf": mean_conf,
                    "digit_share_min": ds_min,
                    "resolution": reason,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "box_height": box_height,
                    "rx1": rx1, "ry1": ry1, "rx2": rx2, "ry2": ry2,
                    "crop_w": crop.shape[1], "crop_h": crop.shape[0],
                })

            # One crop, one answer. Two surviving numbers means we cannot say
            # which belongs to the player the crop is about, so neither is a
            # usable label.
            if len({r["number"] for r in crop_rows}) > 1:
                rej["crop_conflict"] += len(crop_rows)
                continue
            if crop_rows:
                cv2.imwrite(str(out_dir / crop_name), crop)
                rows.extend(crop_rows)
                for r in crop_rows:
                    accepted_numbers[r["number"]] += 1

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
    print(f"rejected (off player):   {rej['off_player']}")
    print(f"rejected (format 1-2dig):{rej['format']}")
    print(f"rejected (digit_share):  {rej['digit_share']}")
    print(f"rejected (conf):         {rej['conf']}")
    print(f"rejected (untestable 1d):{rej['untestable']}")
    print(f"rejected (truncated 1d): {rej['truncated_pair']}")
    print(f"rejected (multiset):     {rej['multiset']}")
    print(f"rejected (crop conflict):{rej['crop_conflict']}")
    print(f"accepted labels:         {len(rows)}")
    n_single = sum(1 for r in rows if r.get("resolution") == "single")
    print(f"  single-digit: {n_single}   two-digit: {len(rows) - n_single}")
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
