"""Experiment: read jersey numbers from detected players in a broadcast frame.

Baseline test of off-the-shelf OCR (EasyOCR, digits-only) on upscaled
torso crops, before deciding whether a custom digit classifier is needed.

Usage: .venv/bin/python scripts/read_jerseys.py data/samples_broadcast/b_30.jpg
"""
import sys
from pathlib import Path

import cv2
import easyocr

from game import easyocr_gpu
import numpy as np
from ultralytics import YOLO

from detect_field import dedupe_boxes

UPSCALE = 4


def detect_players(model, img):
    results = model.predict(img, classes=[0], conf=0.08, verbose=False)
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append({"box": (x1, y1, x2, y2), "foot_point": ((x1 + x2) / 2, y2), "conf": float(box.conf[0])})
    return dedupe_boxes(boxes)


def torso_crop(img, box):
    """Upper-torso band of the player box, where the number sits whether the
    player faces toward or away from the camera."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h = y2 - y1
    ty1 = y1 + int(0.15 * h)
    ty2 = y1 + int(0.60 * h)
    crop = img[max(0, ty1):ty2, max(0, x1):x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)


def main():
    frame_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/samples_broadcast/b_30.jpg")
    img = cv2.imread(str(frame_path))
    if img is None:
        raise SystemExit(f"could not read {frame_path}")

    model = YOLO("yolov8s.pt")
    players = detect_players(model, img)
    print(f"players detected: {len(players)}")

    reader = easyocr.Reader(["en"], gpu=easyocr_gpu(), verbose=False)

    out_dir = Path("data/output/jersey_crops")
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated = img.copy()
    for i, p in enumerate(sorted(players, key=lambda p: p["foot_point"][0])):
        crop = torso_crop(img, p["box"])
        if crop is None:
            continue
        cv2.imwrite(str(out_dir / f"{frame_path.stem}_p{i:02d}.jpg"), crop)

        readings = reader.readtext(crop, allowlist="0123456789")
        x1, y1, x2, y2 = [int(v) for v in p["box"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_parts = [f"{text}({conf:.2f})" for _, text, conf in readings]
        label = " ".join(label_parts) if label_parts else "-"
        cv2.putText(annotated, f"p{i:02d}:{label}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        print(f"  p{i:02d} box=({x1},{y1},{x2},{y2}) det_conf={p['conf']:.2f} ocr={label}")

    out_path = Path("data/output") / f"{frame_path.stem}_jerseys.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"annotated: {out_path}")


if __name__ == "__main__":
    main()
