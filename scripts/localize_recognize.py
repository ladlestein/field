"""Experiment: two-stage jersey number reading on a broadcast frame.

Stage 1 (localize): EasyOCR's CRAFT detector finds tight text regions inside
each player's torso crop. Stage 2 (recognize): PARSeq reads each tight region;
its per-position softmax is renormalized over digits to yield an honest
probability distribution rather than a single string.

Usage: .venv/bin/python scripts/localize_recognize.py data/samples_broadcast/b_30.jpg
"""
import sys
from pathlib import Path

import cv2
import easyocr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from read_jerseys import detect_players, torso_crop

DIGIT_INDICES = list(range(1, 11))  # PARSeq charset: index 1 -> '0' ... 10 -> '9'


def load_parseq():
    model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True).eval()
    preprocess = transforms.Compose([
        transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    return model, preprocess


def localize_text_regions(reader, crop, pad_frac=0.15):
    """CRAFT detection only; returns padded (x1, y1, x2, y2) regions."""
    horizontal, free = reader.detect(crop)
    regions = []
    for x_min, x_max, y_min, y_max in horizontal[0]:
        w, h = x_max - x_min, y_max - y_min
        px, py = int(w * pad_frac), int(h * pad_frac)
        x1, y1 = max(0, x_min - px), max(0, y_min - py)
        x2, y2 = min(crop.shape[1], x_max + px), min(crop.shape[0], y_max + py)
        if x2 - x1 > 4 and y2 - y1 > 4:
            regions.append((x1, y1, x2, y2))
    for quad in free[0]:
        pts = np.array(quad)
        x1, y1 = pts.min(axis=0).astype(int)
        x2, y2 = pts.max(axis=0).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(crop.shape[1], x2), min(crop.shape[0], y2)
        if x2 - x1 > 4 and y2 - y1 > 4:
            regions.append((x1, y1, x2, y2))
    return regions


def digit_distributions(model, preprocess, region_bgr):
    """Run PARSeq; return (greedy_string, greedy_conf, per-position digit dists).

    Each dist is the softmax over PARSeq's full charset at that decode position,
    renormalized over the 10 digit classes.
    """
    pil = Image.fromarray(cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        logits = model(preprocess(pil).unsqueeze(0))
        probs = logits.softmax(-1)
    labels, confidences = model.tokenizer.decode(probs)
    text, conf = labels[0], confidences[0]

    dists = []
    seq = probs[0]  # (L, C)
    for pos in range(seq.shape[0]):
        p = seq[pos]
        if p.argmax().item() == 0:  # [E] end-of-sequence: stop
            break
        digit_mass = p[DIGIT_INDICES]
        total = digit_mass.sum().item()
        if total < 1e-6:
            dists.append(None)  # position is confidently non-digit
            continue
        dist = (digit_mass / digit_mass.sum()).tolist()
        dists.append({"digit_share": total, "dist": dist})
    return text, conf, dists


def fmt_dist(d):
    if d is None:
        return "non-digit"
    top = sorted(enumerate(d["dist"]), key=lambda t: -t[1])[:3]
    body = " ".join(f"{dig}:{p:.2f}" for dig, p in top)
    return f"[{body}] (digit_share={d['digit_share']:.2f})"


def main():
    frame_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/samples_broadcast/b_30.jpg")
    img = cv2.imread(str(frame_path))
    if img is None:
        raise SystemExit(f"could not read {frame_path}")

    yolo = YOLO("yolov8s.pt")
    players = detect_players(yolo, img)
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    parseq, preprocess = load_parseq()

    out_dir = Path("data/output/digit_crops")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"players detected: {len(players)}")
    for i, p in enumerate(sorted(players, key=lambda p: p["foot_point"][0])):
        crop = torso_crop(img, p["box"])
        if crop is None:
            continue
        regions = localize_text_regions(reader, crop)
        if not regions:
            continue
        print(f"p{i:02d} det_conf={p['conf']:.2f} regions={len(regions)}")
        for j, (x1, y1, x2, y2) in enumerate(regions):
            region = crop[y1:y2, x1:x2]
            cv2.imwrite(str(out_dir / f"{frame_path.stem}_p{i:02d}_t{j}.jpg"), region)
            text, conf, dists = digit_distributions(parseq, preprocess, region)
            print(f"  t{j} ({x2-x1}x{y2-y1}px) read={text!r} conf={conf.mean().item():.2f}")
            for pos, d in enumerate(dists):
                print(f"     pos{pos}: {fmt_dist(d)}")


if __name__ == "__main__":
    main()
