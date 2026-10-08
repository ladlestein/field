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

## 8. Closeup pseudo-label harvest + color classifier (+ MPS migration)

**Scripts:** `scripts/pseudo_labels.py`, `scripts/classify_colors.py`
**Run:** `.venv/bin/python3 scripts/pseudo_labels.py [--game ID] [--resume]
[--after-frame N]`; `classify_colors.py [--game ID]`

**Pseudo-labels** (the "teacher" path: CRAFT+PARSeq reads closeup-scale
detections ≥180px tall on every aligned frame; a read survives only as a
clean 1-2 digit decode with digit_share and confidence >0.8 AND membership
in the play's participation number multiset). Yields: week 15 — 1,128
labels, 53 distinct numbers; week 1 — 2,213 labels, 61 distinct. Combined:
**3,341 real per-crop training labels, 70/100 numbers covered, median 34
examples per number; 30 numbers have zero examples** — the measured gap
that defines synthetic data's (targeted, deferred) role. Rejection profile:
of clean, confident digit reads, **~45% failed the multiset check** (week
15: 929/2,057) — the roster constraint is load-bearing; without it the
training set would be poisoned at scale. The multiset check is color-blind
(cross-team number collisions), which is harmless for digit labels.

**Color classifier**: classical HSV (median dominant color of the central
crop region, grass/skin masked), 95.1% against eval-v0 color labels; found
one eval labeling error (fixed: 01D w→b). Discovered week 1 WAS wore
burgundy at home — the untuned burgundy detector puts 44% of week-1 crops
in "other"; needs a week-1 eval set to tune against.

**MPS migration**: all models (YOLO, CRAFT, PARSeq) now run on the Apple
GPU via `game.torch_device()` (`FIELD_DEVICE` env overrides). Speedups:
YOLO 7.4x, CRAFT 6.9x, PARSeq 1.75x; harvester end-to-end ~7x. War
stories, so they don't get re-learned: (1) detached/nohup'd processes run
at background QoS and the GPU scheduler throttles them hard — promote with
`taskpolicy -B -p PID`; (2) one MPS hang (transient, did not reproduce)
and one reproducible silent process death on a specific frame (t_04884, an
extreme sideline closeup whose frame-filling detections make giant CRAFT
input tensors — Metal aborts without a Python traceback). Mitigations now
in the harvester: `--resume` (continue past the last recorded frame),
per-frame heartbeat file (pins any hang to an exact frame), `--after-frame`
(skip a killer input), and a regions-per-crop cap. Still TODO: cap crop
size fed to CRAFT (downscale >800px crops) to avoid the pathological
shapes entirely.

> **Superseded by experiment 9.** The label counts above are real but the
> set was contaminated: ~20-25% of its single-digit labels are wrong. Use
> `pseudo/v2/`, not `pseudo/v1/`. The CRAFT size cap is also done there.

---

## 9. Auditing the pseudo-labels: where a roster cross-check stops working

**Scripts:** `scripts/pseudo_labels.py` (v2 accept rules),
`scripts/audit_labels.py` (new — contact sheets for eyeball verification)
**Run:** `.venv/bin/python3 scripts/pseudo_labels.py --game ID --policy v2`

Before training anything on the entry-8 harvest, sampled its crops and
looked at them. **The two-digit labels were clean (12/12 correct). The
single-digit labels — 944 of 3,341, 28% of the set — were ~20-25% wrong**,
in 36 sampled crops. Two failure modes, both invisible in the aggregate
statistics:

- **Truncation.** CRAFT boxes one digit of a two-digit number; PARSeq reads
  it cleanly and confidently. A crop of #34 is labelled "4", #78 → "8",
  #84 → "4".
- **Neighbour bleed.** In traffic another player overlaps the detection box
  and a sliver of *their* number lands inside the crop. One crop is
  dominated by a lineman wearing 74 with a teammate's "6" in the corner; it
  was labelled "6".

**Why the roster multiset didn't catch them.** It is a strong guard for
two-digit reads — a wrong read must match one of ~16 two-digit numbers out
of 90, so it fails ~4 times in 5. A single-digit read only has to match one
of the ~3 single-digit numbers on the field out of 10. Worse, single digits
in the modern NFL belong to quarterbacks, receivers and defensive backs —
exactly the players who dominate closeups — so the coincidence lands often.
The guard was never uniformly strong; its strength scales with how much the
read constrains.

**v2 accept rules.** Per text region: centre of the region must fall in the
middle 12-88% of the crop (run *before* the recognizer, so it is free);
quality floors as before; then for a single digit, widen the region one
digit-width left, re-read, then right, re-read — a digit pair appearing on
either side means drop; then the multiset; then, if a crop still carries two
different numbers, drop the crop. Region geometry is now stored in
`labels.parquet`, so the set is auditable and re-filterable without
re-running the models. CRAFT input is capped at 1600px (entry 8's TODO).

**Two guard designs that looked right and were wrong** — both caught only by
rendering the rejects and looking:

- *Rejecting regions that touch the crop's side edge* killed 15.7% of
  two-digit reads. Two of twelve sampled rejects were large, correct chest
  numbers that simply ran to the boundary of a tight detection box.
  Centrality alone catches the intruders and keeps these. Dropped.
- *Recovering* the full number from the widening test (lone "8" widens to
  "78" → relabel 78) is very appealing: it converts discards into labels,
  and it passed the multiset check. **About a third of the recoveries were
  wrong.** The widened patch is mostly fabric and PARSeq hallucinates a
  plausible second digit from a fold or seam ("1" especially) — the entry-3
  and entry-5 finding again, and the multiset misses it for the same reason
  it misses the truncation. The widening test is now **reject-only**.
- A third guard was considered and rejected on the evidence: regions in the
  top 15% of the crop looked like junk (a sideline down marker read as "3"),
  but rendering all 147 of them showed they are overwhelmingly **shoulder
  numbers** — legitimate, and a signal worth keeping.

**Results** (both games, 6,211 aligned frames, 29,775 closeup detections,
14,404 raw reads; 1h51m + 2h59m on the GPU, no Metal aborts — the size cap
holds). Rejections: format 8,272, multiset 1,689, off-player 1,463,
untestable single 567, digit_share 507, truncated single 387, confidence
243, crop conflict 54.

|                     | v1     | v2     |
|---------------------|--------|--------|
| labels              | 3,341  | 2,685  |
| crops               | 3,239  | 2,653  |
| single-digit share  | 28.3%  | 12.1%  |
| distinct numbers    | 70     | 69     |
| median labels/number| 34     | 32     |

80% retention, and coverage is essentially unchanged — what went was
concentrated in the contaminated slice. The most-frequent labels are now
72/78/55/74/18 rather than v1's 5/3/8/72/2: the single-digit inflation was
the bug showing up in the histogram all along, if anyone had asked why
skill-position numbers dominated a set built mostly from line play.

**Residual error** (audited): two-digit labels 12/12 correct; single-digit
~9/12, the remainder being non-jersey digits (sideline markers) and
non-player detections (a coach on the sideline). Roughly 3% overall, down
from roughly 7%.

**Method note.** Both label-quality bugs found so far (eval v0, pseudo v1)
were invisible in counts and confidence histograms and obvious in a
twelve-crop contact sheet. `audit_labels.py` exists so that looking is a
pipeline step rather than a debugging afterthought.

---

## 10. Live viewer v0: the game loop exists (server/app.py + viewer/)

**Question.** Can the score-bug → alignment stack run against a playing
video in real time, displayed beside it — the spine every later predictor
plugs into?

**Setup.** One aiohttp process serves the viewer page, the video file
(range requests, so `<video>` seeks), and a WebSocket. The browser's
video element owns the playback clock and reports it over the socket;
a worker thread predicts on the *newest* reported time only — no queue,
so a slow tick costs coverage, never freshness. Per tick: grab the frame
with cv2 (seek by msec), OCR the bug, `parse_bug` + `match_play`, ship
bug state + aligned play + participation + personnel code to the page.

**Result.** Works end to end on the week-15 broadcast: ~220–350 ms per
tick on MPS (seek + OCR + align), comfortably inside the 1 Hz cadence,
correct clock/quarter/down readings on pre-snap wide shots, correct play
alignment with the full 22 displayed as the play unfolds.

**Negative result worth keeping: two OCR paths had silently diverged.**
`scorebug_align.ocr_bug_tokens` (full-width strip below y=820, 3×) fails
to read the game clock on frames where `sweep_broadcast.ocr_roi_tokens`
(tight ROI (450,845)–(1500,1030), 2×) reads it fine — same recognizer,
same frame, different crop/scale, different tokens ("11:32" → "12132"
in the full strip). The manifest was validated with the ROI path, so the
viewer uses it; first suspicion (CPU-vs-MPS nondeterminism) was checked
and cleared — tokens are identical across devices. Lesson: EasyOCR reads
are a function of crop context, not just the pixels of the text itself;
there should eventually be one blessed bug-OCR function, not two.

**Also learned** (user-observed, Tunsil injury): participation positions
are *roster* positions, not alignment — after Tunsil left, WAS's real
line showed as 3 G + 1 T with 3 TEs on the field (Coleman, rostered G,
played T). The panel is labeled "roster positions" accordingly; any
future formation predictor must not treat roster position as where a
player lines up.

---

## 11. Shot-type gate v0: the router exists (scripts/shot_gate.py)

**Question.** Can a cheap rule set over the YOLO detection census route
frames (wide / closeup / other) reliably enough to feed formation
analysis only real wide shots?

**Setup.** Features per frame: count of "field-scale" detections
(height 28–155px, feet above the score bug), count of "big" ones
(≥300px), median field-scale height, and grass fraction (HSV turf mask
over the lower ⅔). Verdict rules over those; ~6ms YOLO + mask per frame
on MPS, so the gate is free at any live cadence. Ground truth for
tuning: the feature distribution of the 125 play-clock representative
frames in plays.csv (known formation shots); validation by contact
sheet, per the house rule.

**First thresholds were wrong in both directions** (guessed, not
measured): median-height cap 110px rejected half of true wides (the
real q05–q95 is 88–137), and requiring *zero* big detections misrouted
punt/PAT formations to closeup — ~10% of true wides carry 1–2 near-camera
sideline bodies. Measuring the known-good distribution before setting
thresholds fixed both in one pass. Final: n_field ≥ 9, med_h ≤ 150,
n_big ≤ 2, grass ≥ 0.30.

**Results** (this game). Random 120-frame sample: 35 wide / 47 closeup
/ 38 other, with zero junk in the wide sheet (two borderline
between-play transition shots at wide scale; harmless — formation
analysis finds no formation there). Play-level recall: 520 of 1,151
aligned play-clock frames gate wide, covering 103/125 plays (82.4%).
**The 22 missed plays are the broadcast's fault, not the gate's**: a
contact sheet of their pre-snap frames shows every one is a closeup —
QB walking to the line, coaches, sideline — the wide camera simply
never ran before those snaps at 1fps sampling. 82% is the source's
ceiling here.

**Design note.** The gate classifies camera *scale* only; pre-snap-ness
stays with the play clock (manifest / live bug OCR), and play-type
routing (scrimmage vs kick) stays with alignment. Keeping those
concerns out of the gate keeps its failure modes legible.

---

## 12. Pre-snap formation geometry v0 (scripts/presnap.py)

**Question.** From one gated wide pre-snap frame: LOS, offense/defense
split, and three graded calls — formation family (shotgun / under
center / pistol), backfield count (FTN's `n_offense_backfield`,
QB excluded), defensive shell (0/1/2-high).

**What works (the durable machinery).** Painted yard lines fitted as
individual (point, direction) lines; a player's field-depth is
interpolated between the two lines bracketing his feet (5yd apart on
the ground), which handles perspective exactly and yields a local
px/yard scale as a byproduct. LOS from the densest 4yd player band,
re-anchored to the OL row's median depth; offense = the side whose
deep bodies are fewer (safeties outnumber the one deep umpire);
"the QB" = the most-central back, selected not gated. Ground truth
joined from participation (`offense_formation`) + FTN
(`n_offense_backfield`, `qb_location`).

**Six wrong designs this entry paid for** (each caught by rendering
overlays, not by the aggregate numbers):

1. *Single-axis depth projection* — perpendicular-in-image is not
   perpendicular-on-ground; wide-split WRs picked up ±40yd phantom
   depth. Replaced by per-line interpolation.
2. *Per-pixel grass gate* — dropped every player standing on the giant
   midfield logo. Replaced by field-region (largest grass component,
   holes filled).
3. *Fixed pixel height band* (inherited from the shot gate) — at tight
   zooms (70+ px/yd) the standing QB exceeds 155px and vanished,
   reading as "no central back" = phantom under-center. Replaced by
   height relative to local yard scale (0.6-3.0 player-heights).
4. *Angle-consistency filter on yard lines* — perspective legitimately
   fans their image angles 10-30°; the filter nuked every frame.
   Extent alone (≥250px) separates real lines from painted digit
   strokes.
5. *Blob-height under-center detector* — hypothesis: UC's QB+C merge
   into one anomalously tall YOLO box. Measured: distributions
   identical across classes (median ratio 1.15 everywhere). Dead.
6. *Multi-frame voting across the play-clock window* — per-class depth
   medians collapsed (3.53/3.66/3.76) because early-window frames
   measure a formation still assembling, not the same formation with
   noise. Freshness-over-coverage applies within the pre-snap window;
   voting over the last ~2s keeps the separation.

**Results** (88 analyzable plays of 93 with truth + wide frame):

- Formation 59/88 (67%) — **below the always-shotgun baseline
  (70.5%)**, stated plainly. Unlike the baseline it recalls half the
  under-center plays and both other calls ride on the same geometry,
  but as a pure single-frame classifier it is at the noise floor:
  measured class medians (depth of most-central back behind the OL
  row's feet) are pistol 2.9 / shotgun 3.7 / UC-RB ~4.3+, within
  ±0.5yd noise of each other. I-form FB vs pistol QB is geometrically
  degenerate at this resolution (accepted confusion).
- Backfield count 49/88 exact (56%), 80/88 within ±1 (91%).
- Shell: no public label; distribution (2-high 51, 1-high 33, 0-high 4)
  is plausible for a snowy December game. Grade later against
  `defense_coverage_type` correlation.

**Paths forward, in expected-value order:** (a) detect FOX's virtual
blue LOS line by color — removes the OL-row anchoring noise entirely;
(b) feet estimation better than box-bottom (bent postures bias depth);
(c) fuse situation priors (down/distance/clock strongly predict
formation) — this is the project's Bayesian architecture anyway, and
vision-vs-prior disagreement is itself a signal; (d) eventually a small
learned classifier on the backfield image patch, trained league-wide.

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
- **A cross-check's strength scales with how much the evidence constrains**
  (experiment 9). The roster multiset is strong against a two-digit read and
  weak against a single digit; the same check, the same data, a fourfold
  difference in filtering power. When the belief state ingests evidence,
  what it must weigh is not "did an independent source agree" but "how
  surprised would that source have been to agree by chance."
