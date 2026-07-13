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

## 6. Score-bug → nflverse play alignment (first end-to-end agreement)

**Script:** `scripts/scorebug_align.py`
**Run:** `.venv/bin/python3 scripts/scorebug_align.py data/samples_broadcast/b_10.jpg [...]`
**Data:** `data/nflverse/` (gitignored): `pbp_participation_2025.parquet`,
`play_by_play_2025.parquet`, `roster_weekly_2025.parquet`,
`ftn_charting_2025.parquet`, plus `FINDINGS.md` with schema notes and both
week-15 rosters. Key facts: 2025 participation data exists publicly (the
post-2022 gap ended) and covers every real snap of `2025_15_WAS_NYG`; the
participation file embeds names, positions, and jersey numbers per play,
index-aligned, so no roster join is needed for number labels.

Method: OCR the FOX score bug (bottom strip, 3x upscaled, charset-restricted;
the condensed font reads 1 as I and the clock colon as 8 — both handled by
canonicalization/fallback). Parse game clock, quarter, down & distance. Join
to play-by-play: filter quarter + down/distance, take the play whose snap
clock is nearest at-or-below the frame clock (pre-snap frames show more time
remaining than the recorded snap time). The play_id keys into participation
for the full 11-v-11 lists.

**Results:** b_01 and b_10 both align to play 828 (Q1 2:14, 1st & 10 at the
NYG 9, Dart incomplete deep to Slayton). Cross-check against experiment 3's
independent visual reads from the b_30 closeup: every read number (84, 85,
78, partial 72) appears in the play's participation list (Johnson, Manhertz,
Thomas, Eluemunor), and the "GO…" nameplate fragment matches #97 Goldman on
defense. Two fully independent paths — pixels vs. bug-OCR+database — agree.

**Scope of the claim:** a working demonstration on one play, not a
validation. Agreement is set-membership ("84 is on the field"), not
per-detection assignment. Caveat: b_30 is ~20s after the incompletion and may
show the next snap; reads check out because personnel carried over.

**Instructive failure:** b_20 (post-play frame) doesn't align — the clock is
past the snap time and the bug still shows the stale pre-play down &
distance. The play clock's presence in the bug is a clean pre-snap/post-play
discriminator (b_01/b_10 have one, b_20 doesn't); between-play frames should
inherit game state via temporal continuity rather than fresh alignment.

**Full-game sweep results** (`scripts/sweep_broadcast.py`, 7,392 frames at
1 fps): first pass covered 143/164 participation plays; every quarter's
parsing died at exactly the 1:00 mark because FOX drops the minutes digit
under a minute (":32"), which also collides with the play-clock format —
resolved positionally (game clock sits above the quarter token; play clock
lives on the down-&-distance line). With the sub-minute fix: **151/164
participation plays covered** (125 with a play-clock pre-snap
representative; ~96% of scrimmage plays). Remaining misses: 7 kickoffs
(different bug layout, deprioritized) and 6 scattered plays where the bug
was likely obscured through the pre-snap window. Outputs:
`data/harvest/manifest.parquet` (per frame), `plays.csv` (per play with
number multisets) — the durable label source for number-reader training.

**v0 crop harvest** (`scripts/harvest_crops.py`): first pass used the
sweep's representative frames directly and 21 plays yielded zero crops —
inspection showed their "latest play-clock frame" was a closeup or sideline
shot (FOX cuts away during pre-snap windows; a play clock in the bug does
not imply a formation shot). Fixed by detection census: for each play, run
the player detector over its last few aligned pre-snap frames and keep the
frame with the most valid player boxes — a wide shot always outscores a
closeup, no learned classifier needed. Result: **157/161 plays, 3,273
torso crops** (mean 20.8/play), labels play-level via `crops.parquet` →
`plays.csv` number multisets; 35 plays flagged `low_quality` (no play-clock
frame; labels may belong to a neighboring play). Random-sample QA: ~25-35%
of crops show readable/partial digits; junk (refs, sideline staff, blur) is
present but harmless under set-level labels — the pairing step routes it to
"no readable number."

---

## 7. Eval set v0 (Claude-labeled, multiset-cross-checked)

**Script:** `scripts/make_eval_sheets.py` (renders labeling contact sheets)
**Labels:** `eval/labels_v0.csv` (committed — labeling effort is not
re-derivable, unlike the rest of the `data/` tree). Columns: slot, crop,
visibility (full/partial/none/junk), number (with `?` for unreadable digit
positions), color (w/b jersey), sure (y/n), play_id.

96 crops sampled from the v0 harvest (excluding low_quality plays), labeled
by Claude by eye at 4x — not yet human-reviewed — with every digit-bearing
label cross-checked against its play's 22-player number multiset from
participation data. Human review of the full reads and (especially) the
"none" calls would strengthen it: the cross-check can catch wrong digits
but not missed numbers.

**Base rates for v0 wide-shot crops** (the measuring stick for any future
recognizer): 13/96 full numbers readable (14%; 11 confident), 29 partial
(30%), 39 none visible (41%), 15 junk/non-players (16%).

**Labeling-process findings:** the multiset cross-check caught 3 human
labeling errors out of 33 initial digit reads — an over-read ("50" that was
a lone 0 next to a fold shadow; the multiset contained 0 and not 50), a
digit confusion (22 vs 24 at 8x — genuinely ambiguous, downgraded to
partial), and one outright hallucination of a digit from a fabric fold (the
same failure mode experiment 5 convicted PARSeq of — humans do it too).
Verification against out-of-band constraints is not optional at this
resolution, for models or for people. After correction: 32/32 compatible.

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
