"""First milestone: find field lines and player locations in a single pre-snap frame.

Usage: .venv/bin/python scripts/detect_field.py data/samples/f_01.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def detect_field_lines(img):
    """Classical CV: isolate white field markings, split into near-horizontal
    (sidelines/hash-adjacent) and near-vertical (yard lines) segments via Hough transform."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Grass mask: broad green/turf hue range, used to restrict line search to the field.
    grass_mask = cv2.inRange(hsv, (25, 30, 30), (95, 255, 255))
    grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    # Line mask: bright, low-saturation pixels (white paint) within the field.
    line_mask = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255))
    line_mask = cv2.bitwise_and(line_mask, line_mask, mask=grass_mask)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    edges = cv2.Canny(line_mask, 50, 150)
    segments = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80, minLineLength=120, maxLineGap=15
    )

    yard_lines, sidelines, other = [], [], []
    if segments is not None:
        for seg in segments.reshape(-1, 4):
            x1, y1, x2, y2 = seg
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angle = abs(angle) % 180
            if angle > 90:
                angle = 180 - angle
            if angle > 55:  # steep -> yard line (crosses field top-to-bottom)
                yard_lines.append(seg)
            elif angle < 20:  # shallow -> sideline / boundary
                sidelines.append(seg)
            else:
                other.append(seg)

    return grass_mask, line_mask, yard_lines, sidelines, other


def _box_area(b):
    x1, y1, x2, y2 = b["box"]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(b1, b2):
    x1 = max(b1["box"][0], b2["box"][0])
    y1 = max(b1["box"][1], b2["box"][1])
    x2 = min(b1["box"][2], b2["box"][2])
    y2 = min(b1["box"][3], b2["box"][3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def dedupe_boxes(boxes, containment_thresh=0.7):
    """Drop boxes that are mostly contained within a higher-confidence box.

    Standard NMS compares intersection-over-union, which misses duplicates where
    one box is a tight, accurate read of a player and the other is a looser
    partial box covering the same player plus some slop -- their IoU is low
    (the union is inflated by the slop) even though one box is basically a
    subset of the other. Intersection-over-smaller-area catches that case.
    """
    boxes_sorted = sorted(boxes, key=lambda b: b["conf"], reverse=True)
    kept = []
    for b in boxes_sorted:
        smaller = _box_area(b)
        is_dup = any(
            min(smaller, _box_area(k)) > 0
            and _intersection_area(b, k) / min(smaller, _box_area(k)) > containment_thresh
            for k in kept
        )
        if not is_dup:
            kept.append(b)
    return kept


def detect_players(model, img):
    from game import torch_device
    results = model.predict(img, classes=[0], conf=0.08, verbose=True,
                            device=torch_device())  # class 0 = person
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, y2  # foot position, not box center, for field location
        boxes.append({"box": (x1, y1, x2, y2), "foot_point": (cx, cy), "conf": float(box.conf[0])})
    return dedupe_boxes(boxes)


def annotate(img, yard_lines, sidelines, other, players):
    out = img.copy()
    for x1, y1, x2, y2 in other:
        cv2.line(out, (x1, y1), (x2, y2), (128, 128, 128), 1)
    for x1, y1, x2, y2 in yard_lines:
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red
    for x1, y1, x2, y2 in sidelines:
        cv2.line(out, (x1, y1), (x2, y2), (255, 0, 255), 2)  # magenta
    for p in players:
        x1, y1, x2, y2 = [int(v) for v in p["box"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = [int(v) for v in p["foot_point"]]
        cv2.circle(out, (cx, cy), 4, (0, 255, 255), -1)
    return out


def main():
    frame_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/samples/f_01.jpg")
    img = cv2.imread(str(frame_path))
    if img is None:
        raise SystemExit(f"could not read {frame_path}")

    grass_mask, line_mask, yard_lines, sidelines, other = detect_field_lines(img)

    model = YOLO("yolov8s.pt")
    players = detect_players(model, img)

    out = annotate(img, yard_lines, sidelines, other, players)

    out_dir = Path("data/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = frame_path.stem
    cv2.imwrite(str(out_dir / f"{stem}_annotated.jpg"), out)
    cv2.imwrite(str(out_dir / f"{stem}_line_mask.jpg"), line_mask)

    print(f"frame: {frame_path}")
    print(f"yard-line segments: {len(yard_lines)}, sideline segments: {len(sidelines)}, other: {len(other)}")
    print(f"players detected: {len(players)}")
    for p in sorted(players, key=lambda p: p["foot_point"][0]):
        print(f"  foot=({p['foot_point'][0]:.0f},{p['foot_point'][1]:.0f}) conf={p['conf']:.2f}")


if __name__ == "__main__":
    main()
