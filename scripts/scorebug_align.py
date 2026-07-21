"""Align a broadcast frame to its nflverse play via score-bug OCR.

The FOX score bug is fixed-position rendered text: game clock and quarter in
the bottom bar, down & distance (and play clock) in a smaller bar above it.
OCR those, then join against play-by-play (quarter + down/distance filter,
nearest game clock at or before the frame's clock, since a pre-snap frame
shows more time remaining than the recorded snap time) to get the play_id,
which keys into the participation file's full 11-v-11 lists.

Usage: .venv/bin/python scripts/scorebug_align.py [--game GAME_ID] frame.jpg [more frames...]
"""
import argparse
import re

import cv2
import easyocr
import polars as pl

from game import Game, add_game_arg, easyocr_gpu

BUG_TOP = 820  # score bug occupies the bottom strip of the 1080p frame

DOWN_WORDS = {"1ST": 1, "2ND": 2, "3RD": 3, "4TH": 4}


OCR_SCALE = 3
# The bug only ever shows digits, the clock colon, down/quarter words, GOAL,
# and team letters; restricting the charset stops '2:14' reading as '2814'.
OCR_ALLOWLIST = "0123456789:&STNDRHOALGT OWNBAL"


def ocr_bug_tokens(reader, img):
    strip = cv2.resize(img[BUG_TOP:, :], None, fx=OCR_SCALE, fy=OCR_SCALE,
                       interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(strip, allowlist=OCR_ALLOWLIST)
    tokens = []
    for quad, text, conf in results:
        xs = [p[0] / OCR_SCALE for p in quad]
        ys = [p[1] / OCR_SCALE for p in quad]
        # The bug's condensed font reads 1 as I sometimes; canonicalize.
        text = text.strip().upper().replace("I", "1")
        tokens.append({
            "text": text,
            "x": min(xs), "y": min(ys) + BUG_TOP,
            "cx": sum(xs) / 4, "cy": sum(ys) / 4 + BUG_TOP,
            "conf": conf,
        })
    return tokens


def group_lines(tokens, y_tol=14):
    """Group tokens into visual lines by y proximity, each sorted by x."""
    lines = []
    for t in sorted(tokens, key=lambda t: t["cy"]):
        for line in lines:
            if abs(line[-1]["cy"] - t["cy"]) <= y_tol:
                line.append(t)
                break
        else:
            lines.append([t])
    return [sorted(line, key=lambda t: t["x"]) for line in lines]


def parse_bug(tokens):
    """Extract clock, quarter, down, distance from OCR tokens."""
    state = {"clock": None, "qtr": None, "down": None, "ydstogo": None, "raw": tokens}

    # Down & distance: "<down> & <n|GOAL>" on one visual line. Remember that
    # line's tokens: the play clock lives there too, and under a minute it
    # looks exactly like a sub-minute game clock, so position is the only
    # thing separating them.
    dd_token_ids = set()
    for line in group_lines(tokens):
        joined = " ".join(t["text"] for t in line)
        m = re.search(r"\b([1234])(?:ST|ND|RD|TH)\s*&\s*(\d{1,2}|GOAL)\b", joined)
        if m:
            state["down"] = int(m.group(1))
            state["ydstogo"] = 0 if m.group(2) == "GOAL" else int(m.group(2))
            dd_token_ids = {id(t) for t in line}
            break

    # Quarter: a standalone down-word (or OT) that is not part of the
    # down-&-distance line. It anchors where the game clock must sit.
    qtr_tok = None
    for t in tokens:
        if id(t) in dd_token_ids:
            continue
        if re.fullmatch(r"[1234](?:ST|ND|RD|TH)|OT", t["text"]):
            if qtr_tok is None or t["cy"] > qtr_tok["cy"]:
                qtr_tok = t
    if qtr_tok is not None:
        state["qtr"] = 5 if qtr_tok["text"] == "OT" else int(qtr_tok["text"][0])

    # Game clock candidates: directly above the quarter token in the main bar
    # (when we found one), never from the down-&-distance line.
    if qtr_tok is not None:
        pool = [
            t for t in tokens
            if t["cy"] < qtr_tok["cy"] and abs(t["cx"] - qtr_tok["cx"]) < 80
        ]
    else:
        pool = [t for t in tokens if id(t) not in dd_token_ids]

    # Clean M:SS first. The bug's small colon often OCRs as an 8 ("2814" for
    # 2:14) or "{"/"."; under a minute the minutes digit disappears entirely
    # (":32", sometimes with tenths), leaving punct-or-8 + seconds.
    for t in pool:
        if re.fullmatch(r"\d{1,2}:\d{2}", t["text"]):
            state["clock"] = t["text"]
            break
    if state["clock"] is None:
        for t in pool:
            m = re.fullmatch(r"(\d{1,2})8(\d{2})", t["text"])
            if m and int(m.group(1)) <= 15 and int(m.group(2)) <= 59:
                state["clock"] = f"{m.group(1)}:{m.group(2)}"
                break
    if state["clock"] is None:
        for t in pool:
            m = re.fullmatch(r"[:{.,'8]\s?(\d{1,2})(?:[.]\d)?", t["text"])
            if m and int(m.group(1)) <= 59:
                state["clock"] = f"0:{int(m.group(1)):02d}"
                break
    return state


def clock_seconds(clock):
    mm, ss = clock.split(":")
    return int(mm) * 60 + int(ss)


def match_play(pbp, state, max_lead=45):
    """Plays matching quarter (and down/distance when parsed), whose snap
    clock is at or just below the frame clock."""
    if state["clock"] is None or state["qtr"] is None:
        return None, "need at least clock + quarter"
    frame_s = clock_seconds(state["clock"])
    cand = pbp.filter(
        (pl.col("qtr") == state["qtr"]) & pl.col("time").is_not_null()
    ).with_columns(
        pl.col("time").str.split(":").list.get(0).cast(pl.Int32).mul(60)
        .add(pl.col("time").str.split(":").list.get(1).cast(pl.Int32))
        .alias("time_s")
    )
    if state["down"] is not None:
        cand = cand.filter(
            (pl.col("down") == state["down"]) & (pl.col("ydstogo") == state["ydstogo"])
        )
    nearby = cand.with_columns((pl.col("time_s") - frame_s).abs().alias("dt")).sort("dt")
    cand = cand.filter(
        (pl.col("time_s") <= frame_s) & (pl.col("time_s") >= frame_s - max_lead)
    ).sort((frame_s - pl.col("time_s")))
    if cand.height == 0:
        closest = [
            f"Q{r['qtr']} {r['time']} {r['down']}&{r['ydstogo']}"
            for r in nearby.head(3).iter_rows(named=True)
        ]
        return None, f"no play matches; closest by clock: {closest}"
    note = "" if cand.height == 1 else f"({cand.height} candidates, took nearest clock)"
    return cand.row(0, named=True), note


def show_participation(part, play_id):
    row = part.filter(pl.col("play_id") == play_id)
    if row.height == 0:
        print("  no participation row (no_play?)")
        return
    r = row.row(0, named=True)
    for side in ("offense", "defense"):
        names = r[f"{side}_names"].split(";")
        nums = r[f"{side}_numbers"].split(";")
        poss = r[f"{side}_positions"].split(";")
        joined = ", ".join(
            f"#{n} {p} {nm}" for n, p, nm in sorted(
                zip(nums, poss, names), key=lambda t: int(t[0])
            )
        )
        print(f"  {side}: {joined}")


def main():
    ap = argparse.ArgumentParser()
    add_game_arg(ap)
    ap.add_argument("frames", nargs="+")
    args = ap.parse_args()
    game = Game(args.game)
    pbp = game.pbp()
    part = game.participation()
    frames = args.frames
    reader = easyocr.Reader(["en"], gpu=easyocr_gpu(), verbose=False)

    for fpath in frames:
        img = cv2.imread(fpath)
        print(f"\n=== {fpath}")
        if img is None:
            print("  unreadable")
            continue
        tokens = ocr_bug_tokens(reader, img)
        state = parse_bug(tokens)
        print(f"  bug: clock={state['clock']} qtr={state['qtr']} "
              f"down={state['down']} ydstogo={state['ydstogo']}")
        print(f"  tokens: {[t['text'] for t in tokens]}")
        play, note = match_play(pbp, state)
        if play is None:
            print(f"  alignment failed: {note}")
            continue
        print(f"  play_id={play['play_id']:.0f} Q{play['qtr']} {play['time']} "
              f"{play['down']}&{play['ydstogo']} at {play['yrdln']} {note}")
        print(f"  desc: {play['desc'][:110]}")
        show_participation(part, play["play_id"])


if __name__ == "__main__":
    main()
