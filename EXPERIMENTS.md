# Experiment Inventory

Working log of experiments toward the project goal: identify, from a live
broadcast stream, which players are on the field. Experiments are listed in
chronological order. Source videos live in `data/` (gitignored):

- `commanders_giants_week_15_2025_full.mp4` — full FOX broadcast, 1080p 29.97fps,
  the production target.
- `commanders_giants_week_15_2025_all-22-compressed.mp4` /
  `..._leaner.mp4` (in `~/Downloads`) — all-22 coaching film, 1080p 59.94fps.
  Each play appears twice: first with the LOS vertical, then a replay with the
  LOS horizontal (fixed camera direction, so orientation relative to each team
  flips by quarter/possession).
- `commanders_giants_week_15_2025_condensed.mp4` — condensed broadcast cut.

Sample frames used below: `data/samples/f_NN.jpg` (all-22, 1 fps from t=600s;
f_11–f_21 are pre-snap on one play, f_21 just before the snap) and
`data/samples_broadcast/b_NN.jpg` (broadcast, 1 fps from t=1200s; b_10 is a
wide pre-snap formation shot, b_30 a tight down-the-line closeup).

---

## 1. Field lines + player detection, all-22 pre-snap frame

**Script:** `scripts/detect_field.py`
**Run:** `.venv/bin/python3 scripts/detect_field.py data/samples/f_21.jpg`
**Output:** `data/output/<stem>_annotated.jpg`, `<stem>_line_mask.jpg`

Classical CV for field geometry (HSV grass/white-paint masking → Canny →
probabilistic Hough; segment angle splits yard lines from sidelines) plus
YOLOv8s person detection for player locations, with foot-point estimation
(bottom-center of box).

**Results (f_21):** yard lines found reliably (as multiple unmerged segments
per painted line); top sideline found; end-zone helmet logo produces false
sideline segments. Players: 17 of 22 detected at conf=0.08 after
intersection-over-smaller-area dedup (`dedupe_boxes`) — standard IoU NMS
misses the loose-box-over-tight-box duplicate pattern. yolov8n at conf=0.35
found only 3; model size + threshold mattered enormously.

**Known limitations:** C/RG/RT read as one blob (also the two DTs across from
them); a crop-and-3x-upscale probe showed resolution fixes the DTs but *not*
the interlocked trio — that needs tracking through the snap, instance
segmentation, or football-specific fine-tuning. Line segments need merging
into single per-line detections; LOS derivation not yet attempted.

---

## 2. Jersey reading baseline: EasyOCR on whole torso crops

**Script:** `scripts/read_jerseys.py`
**Run:** `.venv/bin/python3 scripts/read_jerseys.py data/samples_broadcast/b_30.jpg`

Single-stage baseline: detect players, take a heuristic torso band from each
box, hand the whole band to EasyOCR. Kept as the comparison point that
isolated *localization* (not recognition) as the binding constraint.

**Results (b_30, tight closeup):** near-total failure — no correct two-digit
reads even on numbers a human reads trivially (84, 85, 78 clearly visible).

---

## 3. Two-stage reading: CRAFT localization → PARSeq recognition

**Script:** `scripts/localize_recognize.py`
**Run:** `.venv/bin/python3 scripts/localize_recognize.py data/samples_broadcast/b_30.jpg`
**Output:** tight digit crops in `data/output/digit_crops/`

Stage 1: EasyOCR's CRAFT detector finds text regions inside each torso crop.
Stage 2: PARSeq (torch.hub `baudm/parseq`, pretrained) reads each region; its
per-position softmax is renormalized over the 10 digit classes to produce an
honest probability distribution (`digit_share` = how much mass the position
puts on digits at all — used to flag non-digit regions like nameplates).

**Results (b_30, tight closeup):** 3 exact reads at conf 1.00 (84, 85, 78);
one partial read whose distribution correctly split mass across both digits
of the true number (72 → pos0 7:0.67 / 2:0.29); nameplates mostly flagged
non-digit (digit_share ≈ 0), but one nameplate false-positived as "9" with
digit_share 0.44 — digit-mass alone is an insufficient region filter; needs
geometric priors (nameplates are wide/short and sit above the number) and,
later, roster priors.

**Results (b_10, wide formation shot):** CRAFT finds zero regions at native
resolution; at 4x bicubic upscale, 3 of ~36 players yield regions, all
visually unreadable mush. PARSeq confidently hallucinates on such input
(a "5" at 0.93 from an illegible blob) — its confidence is untrustworthy this
far out of distribution. Never ingest wide-shot reads into a belief state
without independent support.

---

## 4. Multi-frame super-resolution on set players (negative result)

**Script:** `scripts/sr_stack.py`
**Run:** `.venv/bin/python3 scripts/sr_stack.py`
(expects frames in `data/sr_experiment/frames/`, extracted via
`ffmpeg -ss 1205 -t 9 -i data/commanders_giants_week_15_2025_full.mp4 data/sr_experiment/frames/f_%04d.png`)
**Output:** per-player single/stacked/sharpened crops + `reference_players.jpg`
in `data/sr_experiment/out/`

Hypothesis: pre-snap set players are motionless (rule-enforced) under a
locked camera, so registering many frames to sub-pixel precision on a 4x grid
and averaging recovers detail no single frame has. Pipeline: motion-energy
scan finds the static set window → YOLO on reference frame → per player, ECC
translation-only registration of every frame's crop → gate to best-correlated
70% → mean stack (→ optional unsharp).

**Results:** stack ≈ single frame, pixel for pixel. Formation players: zero
text regions either way. Diagnostic measured why: inter-frame sub-pixel shift
std is only 0.06–0.13px (max 0.38px) — the locked camera gives almost no
sample-phase diversity, and h264 P-frames copy blocks from predecessors, so
the frames are nearly N copies of the same data, not N independent samples.
Both failure modes anticipated in the design discussion turned out true
simultaneously.

**Incidental findings:** the one flawless read ("20" at 0.95+) was the chain
crew's yard-marker sign, not a jersey — frames contain many legible non-jersey
digits (markers, painted field numbers, score bug), so reads must be gated to
player-torso regions. Unsharp masking consistently *destroyed* CRAFT
localization (regions found in stacked crops vanished in sharpened ones).

**Verdict:** for this source (locked pre-snap camera, ~7 Mbps h264),
multi-frame SR is a dead end. Might still apply to handheld/jittery shots
(real phase diversity), but those shots make registration hardest.

---

## 5. Anatomical torso-band probe: is the information even there? (yes)

**Script:** none committed — run ad hoc; method fully described below.
**Frame:** `data/sr_experiment/frames/f_0217.png` (reference frame of the
experiment-4 set window).
**Artifacts:** `data/sr_experiment/out/torso_contact_sheet.jpg`,
`band_pNN.png` (6x-upscaled bands, one per player).

Motivated by the observation that a human watching the wide shot *can* read
some jersey numbers, contradicting the pipeline's zero yield. Skips generic
text detection entirely: YOLO detections filtered to on-field players
(box height ≥ 45px, foot point within the field area) → 15 players; for each,
fixed torso bands (8–50% and 25–70% of box height, at full and middle-60%
width) upscaled 6x bicubic. Two evaluations: (a) human inspection of a
contact sheet of the bands; (b) PARSeq directly on each band variant, taking
the variant with highest mean digit mass.

**Human reads (ground truth by eye):** ~half the on-field players have
readable or near-readable numbers: a clear burgundy 9 (p08), a clear trailing
0 (p14), a probable 55 (p03, white-on-blue), a legible shoulder/TV number
(p01, 16 or 18). Two players' numbers were *cut in half by the fixed band*
(p05, p11) because bent torsos shift the number out of any fixed fraction of
the box.

**PARSeq on the same bands:** missed every number the eye reads most easily
(p08's 9 → "5"; p14's 0 → "53"; p03's 55 → nothing), partially agreed on one
(p01 → "1"), and hallucinated where nothing is visible — digit_share 0.91 on
a blank white torso (p10 → "9") and a three-digit "144" (p02) despite jersey
numbers having at most two digits.

**Conclusions:**
- **The wide shot does carry identity information** for a meaningful fraction
  of players. This amends experiment 4's verdict: the SR negative result
  showed stacking adds nothing, not that nothing is there. Single frames
  already contain human-readable digits our pipeline fails to extract.
- The failure is now precisely located, and it's *both* stages: CRAFT never
  localizes these regions at this scale, and PARSeq misreads or hallucinates
  on them even when banding hands it roughly the right pixels — it's trained
  on crisp scene text, not 12px cloth-warped digits.
- **digit_share is not a validity signal at this scale** — it measures
  "vaguely glyph-shaped," scoring 0.91 on a blank torso and 0.02 on a
  human-readable 55. Do not use it to gate belief-state ingestion here.
- Fixed-fraction bands are inadequate localization: bent torsos move the
  number out of band. Pose keypoints (shoulders/hips → torso quad + rotation
  normalization, shoulder points → TV-number anchors) are the right localizer.
- Next build implied by all of the above: pose-guided localization + a small
  purpose-trained digit classifier (two 10-way heads + "not visible," trained
  on synthetically degraded jersey digits), evaluated against a
  replay-derived labeled set (replays of the same play in the same file give
  ground truth for wide-shot crops with no cross-video alignment).

---

## Architectural conclusions so far

- **No single frame answers "who's on the field."** Wide formation shots
  contribute positions/formation plus partial identity evidence (experiment 5:
  roughly half the players' numbers are human-readable there); tight closeups
  and replays contribute the strongest identity reads; a persistent belief
  state fuses evidence across shot types, updated Bayesianly
  (roster/tendency/situation priors × visual-evidence likelihoods →
  posteriors that carry forward).
- **Shot-type classification is the router** everything depends on: classify
  each frame (wide formation / closeup / replay / junk), then apply the
  technique that shot supports. Not yet built; likely next.
- **At closeup scale, localization is the bottleneck** — PARSeq reads nearly
  perfectly when pointed at the right pixels (experiment 3). **At wide-shot
  scale, both stages fail** (experiment 5): CRAFT localizes nothing and
  PARSeq misreads human-readable digits. Closing the wide-shot gap needs
  pose-guided localization plus a recognizer trained for low-res jersey
  digits, not better generic OCR.
- **Guard the belief state.** Recognizer confidence is miscalibrated on
  out-of-distribution input; non-jersey digits abound. Evidence quality
  gating matters as much as evidence collection.
