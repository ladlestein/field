"""Experiment: multi-frame super-resolution on set players in a wide formation shot.

Pre-snap, set players are motionless for seconds while the camera sits on
sticks, so consecutive broadcast frames are many slightly-shifted samples of
the same jersey. Registering each player's crops to sub-pixel precision on an
upscaled grid and averaging should recover detail no single frame contains --
unless h264 compression already destroyed it identically across frames, which
is the question this experiment answers.

Pipeline: find the static set window via motion energy -> detect players on
the reference frame -> per player, register every frame's crop to the
reference with ECC on a 4x grid -> quality-gate by correlation -> average ->
compare CRAFT+PARSeq reads on stacked vs single-frame crops.

Usage: .venv/bin/python scripts/sr_stack.py
"""
from pathlib import Path

import cv2
import easyocr

from game import easyocr_gpu
import numpy as np
from ultralytics import YOLO

from localize_recognize import digit_distributions, fmt_dist, load_parseq, localize_text_regions
from read_jerseys import detect_players

FRAMES_DIR = Path("data/sr_experiment/frames")
OUT_DIR = Path("data/sr_experiment/out")
UPSCALE = 4
MIN_BOX_HEIGHT = 45  # px; skip tiny far-field/sideline detections
CROP_PAD = 0.25  # fraction of box size added on each side
MIN_KEPT_FRAMES = 20


def find_set_window(frame_paths):
    """Longest run of low inter-frame motion = the set window."""
    prev = None
    diffs = []
    for p in frame_paths:
        g = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (480, 270))
        if prev is not None:
            diffs.append(np.mean(cv2.absdiff(g, prev)))
        prev = g
    diffs = np.array(diffs)
    thresh = np.percentile(diffs, 25) * 1.5

    best_start, best_len, run_start = 0, 0, None
    for i, low in enumerate(diffs < thresh):
        if low and run_start is None:
            run_start = i
        elif not low and run_start is not None:
            if i - run_start > best_len:
                best_start, best_len = run_start, i - run_start
            run_start = None
    if run_start is not None and len(diffs) - run_start > best_len:
        best_start, best_len = run_start, len(diffs) - run_start
    return best_start, best_start + best_len, diffs


def padded_box(box, shape):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = w * CROP_PAD, h * CROP_PAD
    return (
        int(max(0, x1 - px)), int(max(0, y1 - py)),
        int(min(shape[1], x2 + px)), int(min(shape[0], y2 + py)),
    )


def register_and_stack(crops, ref_idx):
    """ECC translation-only registration of every crop to the reference, on a
    4x-upscaled grid for sub-pixel accuracy; average the aligned survivors."""
    ref_up = cv2.resize(crops[ref_idx], None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    ref_gray = cv2.cvtColor(ref_up, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    aligned, rhos = [], []
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    for i, crop in enumerate(crops):
        if crop.shape != crops[ref_idx].shape:
            continue
        up = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            rho, warp = cv2.findTransformECC(ref_gray, gray, warp, cv2.MOTION_TRANSLATION, criteria)
        except cv2.error:
            continue
        w = cv2.warpAffine(
            up.astype(np.float32), warp, (ref_up.shape[1], ref_up.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE,
        )
        aligned.append(w)
        rhos.append(rho)

    if len(aligned) < MIN_KEPT_FRAMES:
        return None, None, len(aligned)

    # Gate to the best-correlated 70% -- drops twitches and the snap itself.
    rhos = np.array(rhos)
    keep = rhos >= np.quantile(rhos, 0.3)
    stack = np.mean([a for a, k in zip(aligned, keep) if k], axis=0)
    return np.clip(stack, 0, 255).astype(np.uint8), ref_up, int(keep.sum())


def unsharp(img, sigma=2.0, amount=0.6):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def recognize(reader, parseq, preprocess, img, label):
    regions = localize_text_regions(reader, img)
    print(f"    [{label}] regions={len(regions)}")
    for j, (x1, y1, x2, y2) in enumerate(regions):
        text, conf, dists = digit_distributions(parseq, preprocess, img[y1:y2, x1:x2])
        print(f"      t{j} ({x2-x1}x{y2-y1}px) read={text!r} conf={conf.mean().item():.2f}")
        for pos, d in enumerate(dists):
            print(f"         pos{pos}: {fmt_dist(d)}")


def main():
    frame_paths = sorted(FRAMES_DIR.glob("f_*.png"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start, end, diffs = find_set_window(frame_paths)
    print(f"frames: {len(frame_paths)}, set window: [{start}, {end}) "
          f"({(end - start) / 29.97:.1f}s), motion 25th pct: {np.percentile(diffs, 25):.2f}")
    window_paths = frame_paths[start:end]
    ref_idx = len(window_paths) // 2

    ref_img = cv2.imread(str(window_paths[ref_idx]))
    yolo = YOLO("yolov8s.pt")
    players = [
        p for p in detect_players(yolo, ref_img)
        if (p["box"][3] - p["box"][1]) >= MIN_BOX_HEIGHT
    ]
    players.sort(key=lambda p: p["foot_point"][0])
    print(f"players kept (height >= {MIN_BOX_HEIGHT}px): {len(players)}")

    annotated = ref_img.copy()
    for i, p in enumerate(players):
        x1, y1, x2, y2 = [int(v) for v in p["box"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, str(i), (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(str(OUT_DIR / "reference_players.jpg"), annotated)

    # One decode pass over the window; collect every player's crop per frame.
    boxes = [padded_box(p["box"], ref_img.shape) for p in players]
    crops = [[] for _ in players]
    for path in window_paths:
        frame = cv2.imread(str(path))
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            crops[i].append(frame[y1:y2, x1:x2])

    reader = easyocr.Reader(["en"], gpu=easyocr_gpu(), verbose=False)
    parseq, preprocess = load_parseq()

    for i, player_crops in enumerate(crops):
        stack, ref_up, kept = register_and_stack(player_crops, ref_idx)
        h, w = player_crops[ref_idx].shape[:2]
        print(f"\np{i:02d} native={w}x{h}px frames_kept={kept}")
        if stack is None:
            print("    skipped: too few registered frames")
            continue
        sharp = unsharp(stack)
        cv2.imwrite(str(OUT_DIR / f"p{i:02d}_single.png"), ref_up)
        cv2.imwrite(str(OUT_DIR / f"p{i:02d}_stack.png"), stack)
        cv2.imwrite(str(OUT_DIR / f"p{i:02d}_sharp.png"), sharp)
        recognize(reader, parseq, preprocess, ref_up, "single 4x")
        recognize(reader, parseq, preprocess, stack, "stacked")
        recognize(reader, parseq, preprocess, sharp, "stacked+unsharp")


if __name__ == "__main__":
    main()
