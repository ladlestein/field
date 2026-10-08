"""Pre-snap formation geometry on gated wide frames (rung 1).

From one wide pre-snap frame: estimate the line of scrimmage, split the
detections into offense and defense, and make three graded calls —
formation family (shotgun / under center / pistol), backfield count
(FTN's n_offense_backfield, which excludes the QB), and the defensive
shell (0/1/2-high).

Geometry, deliberately minimal (no homography yet): the painted yard
lines give an image-space axis family; perpendicular to their median
direction is "downfield" for projection purposes, and the median spacing
between adjacent painted lines is 5 real yards, which converts pixel
offsets to yards near the formation. Everything downstream is 1-D
arithmetic on two projections (downfield = depth, along-yard-line =
lateral).

Design choices that lean on known failure modes rather than fighting
them: an under-center QB usually merges with the center into one YOLO
blob (EXPERIMENTS.md entry 1), so "no central player deeper than ~2.5
yards" IS the under-center signal. Offense/defense: the defense always
owns the deepest player (safeties 10+ yards; the deepest offensive
player, a shotgun QB or deep back, sits ~7). No team colors needed.

Thresholds here were set from measured per-play feature distributions
against nflverse/FTN truth (EXPERIMENTS.md entry 12), not guessed.

Usage:
  .venv/bin/python3 scripts/presnap.py [--game ID] [--dump]   # eval
--dump writes per-play features + calls + truth to eval/presnap/
along with annotated overlays for every disagreement.
"""
import argparse
import math

import cv2
import numpy as np
import polars as pl

from detect_field import detect_field_lines, detect_players
from game import Game, add_game_arg
from shot_gate import FIELD_SCALE, MAX_FOOT_Y, classify_shot, contact_sheet

# Depth thresholds are yards behind the OL row's FEET (not the ball) —
# linemen set their feet ~1.5-2yd behind the ball, so these run ~1.8yd
# shallower than playbook numbers. Measured shallowest-central-back
# medians (EXPERIMENTS.md entry 12): pistol QB ~2.9, shotgun QB ~3.4,
# under-center RB (the QB is blob-merged and invisible) ~4.8. The classes
# genuinely overlap at this resolution; boundaries split the medians.
PISTOL_MAX_DEPTH = 3.1
SHOTGUN_MAX_DEPTH = 4.4   # deeper shallowest-back than this = UC's RB
BACKFIELD_MIN_DEPTH = 2.0
BACKFIELD_MAX_DEPTH = 8.0  # beyond this is the umpire or junk, not a back
QB_SEARCH_LAT = 4.0       # lateral band searched for the QB/backfield stack
BACKFIELD_LAT = 8.0       # lateral band for backfield membership
SHELL_MIN_DEPTH = 9.0     # defenders deeper than this (from OL row) are "high"
LOS_WINDOW_YD = 4.0       # density window that locates the LOS


def cross2(a, b):
    """z-component of the 2-D cross product (numpy 2.x dropped 2-D cross)."""
    return float(a[0] * b[1] - a[1] * b[0])


def fit_yard_lines(yard_segments):
    """Group Hough segments into distinct painted yard lines, each fitted as
    (point, unit direction) by PCA over its segments' endpoints.

    Perspective makes the lines converge, so no single image axis measures
    depth. Instead each painted line is an iso-depth contour: a point's
    field-depth is interpolated between the two lines bracketing it, which
    is locally exact under perspective and makes the yard scale local for
    free.
    """
    lines = []  # each: {"pts": [...], "p": mean, "u": unit dir (dy>0)}
    for x1, y1, x2, y2 in yard_segments:
        v = np.array([x2 - x1, y2 - y1], float)
        n = np.linalg.norm(v)
        if n < 1:
            continue
        u = v / n if v[1] > 0 else -v / n
        mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2], float)
        for ln in lines:
            if abs(cross2(u, ln["u"])) > 0.07:   # ~4 degrees
                continue
            if abs(cross2(mid - ln["p"], ln["u"])) > 18:
                continue
            ln["pts"] += [np.array([x1, y1], float), np.array([x2, y2], float)]
            break
        else:
            lines.append({"pts": [np.array([x1, y1], float),
                                  np.array([x2, y2], float)],
                          "p": mid, "u": u})
    fitted = []
    for ln in lines:
        pts = np.array(ln["pts"])
        p = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - p)
        u = vt[0] / np.linalg.norm(vt[0])
        if u[1] < 0:
            u = -u
        extent = float(np.ptp(pts @ u))
        fitted.append({"p": p, "u": u, "n_seg": len(pts) // 2,
                       "extent": extent})
    # Painted field NUMBERS fake their way in here: at tight zooms their
    # 2yd-tall digit strokes pass Hough's minimum length and their slant
    # tilts the whole geometry. A real yard line spans a large stretch of
    # the frame; number strokes don't. (No angle-consistency filter:
    # perspective legitimately fans the lines' image angles 10-30 degrees
    # apart — handling that is the point of per-line interpolation.)
    return [ln for ln in fitted if ln["extent"] >= 250]


def signed_dists(lines, pt):
    """Signed perpendicular distance (px) from pt to each fitted line."""
    return np.array([cross2(pt - ln["p"], ln["u"]) for ln in lines])


def depth_mapping(lines, anchor):
    """Order the fitted lines by their distance at the anchor point and
    assign each a field-depth in yards (adjacent painted lines are 5yd
    apart; a Hough-missed line shows up as a gap snapping to 10/15yd).
    Returns (ordered lines, yard positions) or None."""
    if len(lines) < 3:
        return None
    d = signed_dists(lines, anchor)
    order = np.argsort(d)
    lines = [lines[i] for i in order]
    d = d[order]
    gaps = np.diff(d)
    if (gaps < 25).any():          # two "lines" closer than ~1yd: bad fit
        lines = [ln for i, ln in enumerate(lines)
                 if i == 0 or gaps[i - 1] >= 25]
        d = signed_dists(lines, anchor)
        gaps = np.diff(d)
        if len(lines) < 3:
            return None
    base = float(np.median(gaps[gaps <= np.min(gaps) * 1.4]))
    yards = [0.0]
    for g in gaps:
        yards.append(yards[-1] + 5.0 * max(1, round(g / base)))
    return lines, np.array(yards)


def field_depth(lines, yards, pt):
    """Field-depth of an image point in yards, by interpolating between the
    two bracketing painted lines (local px/yd comes out as a byproduct)."""
    d = signed_dists(lines, pt)
    i = int(np.searchsorted(d, 0.0))
    if i == 0:
        a, b = 0, 1
    elif i >= len(d):
        a, b = len(d) - 2, len(d) - 1
    else:
        a, b = i - 1, i
    ppy = (d[b] - d[a]) / (yards[b] - yards[a])
    depth = yards[a] + (0.0 - d[a]) / ppy
    return depth, ppy


def field_region(grass_mask):
    """The playing surface as one filled region: largest grass component,
    closed and hole-filled, so painted logos, numbers and the white LOS
    area count as field. A per-pixel turf test drops every player who
    happens to stand on the midfield logo."""
    closed = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE,
                              np.ones((45, 45), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if n < 2:
        return closed
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == biggest).astype(np.uint8) * 255
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(region)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def on_field(region, pt):
    y, x = int(pt[1]), int(pt[0])
    h, w = region.shape
    return 0 <= y < h and 0 <= x < w and region[y, x] > 0


def analyze_presnap(img, players=None, yolo=None):
    """All rung-1 measurements for one frame. Returns None when the frame
    lacks usable geometry (too few yard lines / players)."""
    if players is None:
        players = detect_players(yolo, img)
    grass_mask, _, yard_segments, _, _ = detect_field_lines(img)

    anchor = np.array([960.0, 620.0])
    mapping = depth_mapping(fit_yard_lines(yard_segments), anchor)
    if mapping is None:
        return None
    lines, yards = mapping

    # Player filter. On the field (largest grass component, holes filled,
    # so painted logos count — a per-pixel turf test drops every player
    # standing on the midfield logo), above the score bug, and of
    # human height for the LOCAL yard scale: a fixed pixel band here
    # silently dropped the standing QB at tight zooms, which read as
    # "no central back" and miscalled shotgun as under center.
    region = field_region(grass_mask)
    feet, depths, ppys, lats = [], [], [], []
    for p in players:
        f = np.array(p["foot_point"])
        if f[1] > MAX_FOOT_Y or not on_field(region, f):
            continue
        dep, ppy_f = field_depth(lines, yards, f)
        if ppy_f <= 5:
            continue
        h_yd = (p["box"][3] - p["box"][1]) / ppy_f
        if not (0.6 <= h_yd <= 3.0):
            continue
        feet.append(f)
        depths.append(dep)
        ppys.append(ppy_f)
        # Lateral: along the yard-line direction nearest the player,
        # scaled by his local yard scale. Only used relatively.
        u = lines[int(np.argmin(np.abs(signed_dists(lines, f))))]["u"]
        lats.append(float((f - anchor) @ u) / ppy_f)
    if len(feet) < 10:
        return None
    depth_f = np.array(depths)   # field yards along the yard-line grid
    lat_yd = np.array(lats)
    ppy = float(np.median(ppys))

    # LOS: densest LOS_WINDOW_YD band of players along field depth.
    order = np.sort(depth_f)
    best_n, best_center = -1, None
    for i in range(len(order)):
        n = int(np.searchsorted(order, order[i] + LOS_WINDOW_YD) - i)
        if n > best_n:
            best_n, best_center = n, order[i] + LOS_WINDOW_YD / 2
    los = best_center
    depth_yd = depth_f - los

    # Which side is the defense? Not "the deepest player" — the umpire
    # stands ~12yd deep in the OFFENSIVE backfield, deeper than any
    # shotgun QB. But the defense fields several deep bodies (safeties,
    # off-coverage corners) where the offense has at most that one
    # umpire, so count players beyond 6yd per side; tie falls back to
    # the deeper maximum.
    neg, pos = depth_yd[depth_yd < 0], depth_yd[depth_yd >= 0]
    if len(neg) == 0 or len(pos) == 0:
        return None
    deep_neg, deep_pos = int((neg < -6).sum()), int((pos > 6).sum())
    if deep_neg != deep_pos:
        def_negative = deep_neg > deep_pos
    else:
        def_negative = abs(neg.min()) > pos.max()
    off_sign = 1.0 if def_negative else -1.0
    off_mask = depth_yd * off_sign >= 0
    off_depth = depth_yd[off_mask] * off_sign    # yards behind LOS, positive
    off_lat = lat_yd[off_mask]
    def_depth = depth_yd[~off_mask] * -off_sign  # yards beyond LOS, positive
    if len(off_depth) == 0 or len(def_depth) == 0:
        return None
    # OL center: densest 8yd lateral window among near-LOS offensive players.
    near = off_lat[np.abs(off_depth) <= 2.5]
    if len(near) == 0:
        return None
    order = np.sort(near)
    best_n, ol_center = -1, None
    for i in range(len(order)):
        n = int(np.searchsorted(order, order[i] + 8.0) - i)
        if n > best_n:
            best_n, ol_center = n, order[i] + 4.0

    # Re-anchor depths to the OL row itself. The density-window LOS sits
    # ~2yd on the defense's side of the ball (the window holds more
    # defenders than blockers), which showed up as every play's OL at a
    # phantom 1.6-2.0yd "depth". Football measures backfield depth from
    # the line's feet anyway.
    row_mask = (off_depth <= 3.5) & (np.abs(off_lat - ol_center) <= 6.0)
    if not row_mask.any():
        return None
    ol_row = float(np.median(off_depth[row_mask]))
    off_depth = off_depth - ol_row
    def_depth = def_depth + ol_row
    # Sharpen the lateral center to the row's median: the density window
    # only localizes it to a couple of yards.
    ol_center = float(np.median(off_lat[row_mask]))

    # "The QB" is the MOST CENTRAL deep player, selected, not gated: a QB
    # stands directly behind the C while wings and H-backs at the same
    # shallow depths sit 2.5-4yd off-center, so they lose the centrality
    # contest to a visible QB instead of masquerading as him — and a
    # hard lateral cutoff can't zero out on ol_center noise. Players
    # within 0.8yd lateral of the winner (a stacked pistol/I backfield)
    # resolve to the shallowest of the stack.
    cmask = ((np.abs(off_lat - ol_center) <= QB_SEARCH_LAT)
             & (off_depth > BACKFIELD_MIN_DEPTH)
             & (off_depth <= BACKFIELD_MAX_DEPTH))
    if cmask.any():
        coff = np.abs(off_lat - ol_center)[cmask]
        cdep = off_depth[cmask]
        stack = coff <= coff.min() + 0.8
        qb_depth = float(cdep[stack].min())
    else:
        qb_depth = 0.0

    # Formation family from that player's depth (OL-row-relative):
    # an under-center QB merges with the C (entry 1) and is invisible,
    # so the most-central back is then the RB at ~4.5-6. Pistol QB ~2.5-3,
    # shotgun QB ~3.2-4. Known confusion, accepted for now: an I-form
    # fullback (under center) stands where a pistol QB stands; depth
    # geometry alone cannot split that pair at this resolution.
    if qb_depth == 0.0 or qb_depth > SHOTGUN_MAX_DEPTH:
        formation = "UNDER CENTER"
    elif qb_depth <= PISTOL_MAX_DEPTH:
        formation = "PISTOL"
    else:
        formation = "SHOTGUN"

    in_backfield = (
        (off_depth > BACKFIELD_MIN_DEPTH) & (off_depth <= BACKFIELD_MAX_DEPTH)
        & (np.abs(off_lat - ol_center) <= BACKFIELD_LAT)
    )
    n_backfield = int(in_backfield.sum())
    if formation != "UNDER CENTER" and n_backfield > 0:
        n_backfield -= 1  # FTN's count excludes the QB

    n_high = int((def_depth >= SHELL_MIN_DEPTH).sum())
    shell = {0: "0-high", 1: "1-high"}.get(n_high, "2-high")

    return {
        "los": float(los), "ppy": ppy, "lines": lines, "yards": yards,
        "off_sign": off_sign,
        "n_off": len(off_depth), "n_def": len(def_depth),
        "qb_depth": round(qb_depth, 2), "formation": formation,
        "n_backfield": n_backfield, "n_high": n_high, "shell": shell,
        "ol_center": float(ol_center),
        "feet": feet, "depth_yd": depth_yd, "lat_yd": lat_yd,
    }


def draw_overlay(img, r):
    """Annotated copy of the frame: fitted yard lines, LOS, feet by side."""
    out = img.copy()
    lines, yards = r["lines"], r["yards"]
    for ln in lines:
        p1 = (ln["p"] - ln["u"] * 1500).astype(int)
        p2 = (ln["p"] + ln["u"] * 1500).astype(int)
        cv2.line(out, tuple(p1), tuple(p2), (180, 180, 180), 1)
    # LOS: interpolate position/direction between its bracketing lines.
    i = int(np.clip(np.searchsorted(yards, r["los"]), 1, len(lines) - 1))
    a, b = lines[i - 1], lines[i]
    t = (r["los"] - yards[i - 1]) / (yards[i] - yards[i - 1])
    p = a["p"] + (b["p"] - a["p"]) * t
    u = a["u"] + (b["u"] - a["u"]) * t
    u /= np.linalg.norm(u)
    cv2.line(out, tuple((p - u * 1500).astype(int)),
             tuple((p + u * 1500).astype(int)), (0, 255, 255), 3)
    for f, d in zip(r["feet"], r["depth_yd"]):
        signed = d * r["off_sign"]  # positive = offense side of the LOS
        if abs(signed) <= 1.0:
            color = (0, 200, 0)       # line scrum
        elif signed > 0:
            color = (255, 160, 0)     # offense backfield (blue-ish)
        else:
            color = (0, 0, 255)       # defense (red)
        cv2.circle(out, tuple(f.astype(int)), 7, color, -1)
        cv2.putText(out, f"{d:+.1f}", tuple((f + 10).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    label = (f"{r['formation']} qb={r['qb_depth']} bf={r['n_backfield']} "
             f"{r['shell']} ppy={r['ppy']:.1f}")
    cv2.putText(out, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (0, 255, 255), 3, cv2.LINE_AA)
    return out


# --- eval -------------------------------------------------------------------

def pick_frames(game, yolo, per_play=2):
    """Up to per_play latest wide-gated play-clock frames per aligned play.

    One frame carries LOS/center/detection noise that the formation
    itself doesn't have; the caller votes across frames. Latest frames
    first (closest to snap = most settled)."""
    m = pl.read_parquet(game.manifest_path).filter(
        pl.col("play_id").is_not_null() & pl.col("play_clock")
    ).sort("frame_idx", descending=True)
    chosen = {}
    for r in m.iter_rows(named=True):
        pid = int(r["play_id"])
        if len(chosen.get(pid, [])) >= per_play:
            continue
        img = cv2.imread(str(game.frames_dir / f"t_{r['frame_idx']:05d}.jpg"))
        if img is None:
            continue
        players = detect_players(yolo, img)
        if classify_shot(players, img)[0] == "wide":
            chosen.setdefault(pid, []).append((r["frame_idx"], img, players))
    return chosen


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    add_game_arg(ap)
    ap.add_argument("--dump", action="store_true",
                    help="write features CSV + disagreement overlays")
    args = ap.parse_args()
    game = Game(args.game)
    yolo = YOLO("yolov8s.pt")

    truth = game.participation().select(
        pl.col("play_id").cast(pl.Int64), "offense_formation"
    ).join(
        game.ftn().select(
            pl.col("nflverse_play_id").cast(pl.Int64).alias("play_id"),
            "qb_location", "n_offense_backfield",
        ),
        on="play_id",
    ).filter(pl.col("offense_formation").is_not_null())

    frames = pick_frames(game, yolo)
    out_dir = game.eval_dir / "presnap"
    out_dir.mkdir(parents=True, exist_ok=True)

    def mode(vals):
        return max(set(vals), key=vals.count)

    rows, overlays = [], []
    for t in truth.iter_rows(named=True):
        pid = int(t["play_id"])
        if pid not in frames:
            continue
        results = [(idx, img, analyze_presnap(img, players=players))
                   for idx, img, players in frames[pid]]
        results = [(i, im, r) for i, im, r in results if r is not None]
        if not results:
            rows.append({"play_id": pid, "ok": False,
                         "truth_formation": t["offense_formation"]})
            continue
        # Vote across the play's frames: the formation doesn't change
        # frame to frame, the measurement noise does.
        idx, img, r0 = results[0]  # latest = closest to snap
        row = {
            "play_id": pid, "frame_idx": idx, "ok": True,
            "n_frames": len(results),
            "qb_depth": float(np.median([r["qb_depth"] for _, _, r in results])),
            "formation": mode([r["formation"] for _, _, r in results]),
            "n_backfield": mode([r["n_backfield"] for _, _, r in results]),
            "shell": mode([r["shell"] for _, _, r in results]),
            "n_off": r0["n_off"], "n_def": r0["n_def"],
            "ppy": round(r0["ppy"], 1),
            "truth_formation": t["offense_formation"],
            "truth_backfield": t["n_offense_backfield"],
        }
        rows.append(row)
        if args.dump and (row["formation"] != row["truth_formation"]
                          or row["n_backfield"] != row["truth_backfield"]):
            cap = (f"p{pid} t_{idx:05d} {row['formation']} (true "
                   f"{t['offense_formation']}) bf={row['n_backfield']} "
                   f"(true {t['n_offense_backfield']}) qb={row['qb_depth']:.1f}")
            overlays.append((draw_overlay(img, r0), cap))

    df = pl.DataFrame([r for r in rows if r.get("ok")])
    print(f"plays with truth + wide frame: {len(rows)}, analyzable: {df.height}")
    if df.height:
        form_acc = df.filter(pl.col("formation") == pl.col("truth_formation"))
        bf_acc = df.filter(pl.col("n_backfield") == pl.col("truth_backfield"))
        print(f"formation accuracy: {form_acc.height}/{df.height} "
              f"({form_acc.height / df.height:.1%})")
        print(df.group_by("truth_formation", "formation").len()
                .sort("truth_formation", "formation"))
        print(f"backfield accuracy: {bf_acc.height}/{df.height} "
              f"({bf_acc.height / df.height:.1%})")
        print("qb_depth by true formation:")
        print(df.group_by("truth_formation").agg(
            pl.col("qb_depth").quantile(0.1).alias("q10"),
            pl.col("qb_depth").median().alias("med"),
            pl.col("qb_depth").quantile(0.9).alias("q90"),
        ).sort("truth_formation"))
        print(df.group_by("shell").len().sort("shell"))
    if args.dump:
        pl.DataFrame(rows).write_csv(out_dir / "features.csv")
        if overlays:
            cv2.imwrite(str(out_dir / "disagreements.jpg"),
                        contact_sheet(overlays), [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"disagreement overlays: {out_dir / 'disagreements.jpg'} "
                  f"({len(overlays)})")


if __name__ == "__main__":
    main()
