# field

Real-time NFL broadcast analysis: identify which players are on the field
from a live TV stream. Long-term architecture: a persistent Bayesian belief
state over the 22 on-field players, updated per-frame from shot-type-routed
visual evidence (wide shots → positions/formation; closeups/replays → jersey
numbers) combined with out-of-band priors (rosters, participation data,
tendencies).

Read `EXPERIMENTS.md` before proposing approaches — it inventories every
experiment to date with results and, importantly, the negative results and
overturned assumptions. Add an entry there for every experiment; record
failures as carefully as successes.

## Environment

- Python venv at `.venv/` (Python 3.14). Run everything as
  `.venv/bin/python3 scripts/<script>.py` from the repo root.
- Dataframes: **polars**, not pandas (pandas/pyarrow are not installed).
- ML: ultralytics YOLOv8 (weights auto-download to repo root, gitignored),
  EasyOCR, PARSeq via torch.hub. All CPU; no CUDA on this machine.
- `data/` is gitignored and holds everything bulky: source videos, extracted
  frames, experiment outputs, and nflverse parquet files
  (`data/nflverse/FINDINGS.md` documents their schemas and the week-15
  rosters).

## Test game

Washington Commanders @ New York Giants, 2025 week 15
(`GAME_ID = "2025_15_WAS_NYG"`). Videos in `data/`:
`..._full.mp4` is the FOX broadcast (1080p, 29.97fps, the production target);
`..._all-22-compressed.mp4` is coaching film (each play shown twice: LOS
vertical, then LOS horizontal); `..._condensed.mp4` is a short cut.

## Pipeline scripts (in dependency order)

Everything derived from one broadcast lives under
`data/games/<game_id>/` (frames/, manifest.parquet, plays.csv,
crops/<policy>/, eval/). `scripts/game.py` owns the layout and nflverse
table access; pipeline scripts take `--game` (default `2025_15_WAS_NYG`).

1. `scripts/extract_frames.py` — all frame extraction goes through this;
   don't hand-roll ffmpeg commands. Default mode writes 1 fps jpgs named
   `t_NNNNN.jpg` where index N covers video seconds [N-1, N) — downstream
   code relies on `t_sec = index - 1`. `--start/--duration --native-fps`
   writes every frame of a window as lossless png (used for multi-frame
   work, where jpg re-compression would contaminate the input).
2. `scripts/detect_field.py` — field lines (Hough) + player boxes (YOLOv8s,
   conf 0.08 + containment dedup) on a single frame.
3. `scripts/scorebug_align.py` — OCR the FOX score bug, align a frame to its
   nflverse play, print the 22-player participation lists. Importable parts
   (`parse_bug`, `match_play`, `group_lines`) are reused by the sweep.
4. `scripts/sweep_broadcast.py` — run alignment over all extracted 1 fps
   frames; writes the game's `manifest.parquet` (per frame) and `plays.csv`
   (one representative pre-snap frame per play, with jersey-number
   multisets). The manifest is the durable label source for training; crops
   are cheap derived artifacts, safe to re-cut. Partial runs (`--limit` /
   `--start-idx`) require `--tag` so they can't clobber the full outputs.
5. `scripts/harvest_crops.py` — cut per-player torso crops from each
   covered play's best pre-snap frame (chosen by detection census over the
   play's aligned frames, since play-clock presence doesn't imply a wide
   shot). Writes the game's `crops/<policy>/` + `crops.parquet`; labels
   stay play-level in `plays.csv`. Crop policy is versioned; v0 is the
   crude fixed torso band. Do NOT re-cut a policy dir that an eval set
   references — crop filenames encode detection order, which is not stable
   across sources; use a new policy name instead.
6. `scripts/pseudo_labels.py` — the teacher path: reads closeup-scale
   detections on every aligned frame with CRAFT+PARSeq and keeps only reads
   that survive geometry, quality, truncation and roster-multiset checks,
   yielding per-crop number labels with no hand-labeling. Writes
   `pseudo/<policy>/` + `labels.parquet`; policy is versioned like crop
   policy, because the accept rules decide what a label *means* — v1 is the
   pre-audit harvest, v2 adds the guards from EXPERIMENTS.md entry 9.
7. `scripts/classify_colors.py` — per-crop team color (HSV, grass/skin
   masked) so a play's number multiset can be split by team. Writes
   `color.parquet`. The burgundy scorer is untuned (week 1 WAS wore
   burgundy at home; needs a week-1 eval set).
8. `scripts/make_eval_sheets.py` — labeling contact sheets for eval sets;
   labels live in repo-tracked `eval/` (hand-labeled work is not
   re-derivable, unlike everything in `data/`).
9. `scripts/read_jerseys.py`, `scripts/localize_recognize.py`,
   `scripts/sr_stack.py` — experiment scripts (see EXPERIMENTS.md entries
   2-4); superseded in parts but kept as baselines.

## Live viewer

`server/app.py` (engine) + `viewer/index.html` (page) — the game loop: one
aiohttp process serving the page, the video (with range support), and a
WebSocket. The browser's `<video>` owns the playback clock; the engine
predicts on the newest reported time only (freshness over coverage — a
prediction after the snap is worthless). The server adds `scripts/` to
sys.path to reuse the pipeline modules. Run with
`.venv/bin/python3 server/app.py`, open
http://127.0.0.1:8899/. Bug OCR must use `sweep_broadcast.ocr_roi_tokens`
(the manifest-validated path), NOT `scorebug_align.ocr_bug_tokens` — same
recognizer, different crop, different reads (EXPERIMENTS.md entry 10).
Participation positions shown are roster positions, not alignment.

## Hard-won facts (don't re-learn these)

- Broadcast OCR: the FOX bug's condensed font reads 1 as I and the clock
  colon as 8; both are handled in `parse_bug`. The play clock's presence
  distinguishes pre-snap frames from post-play frames with stale down &
  distance.
- Recognizer confidence (PARSeq, EasyOCR) is untrustworthy on low-res
  input — it hallucinates digits from blur with high confidence. Never
  ingest a read without independent support (experiments 3 and 5).
- Multi-frame super-resolution on the locked pre-snap camera is a dead end
  (experiment 4): ~0.1px inter-frame shift diversity + h264 block copying.
- Frames contain many legible non-jersey digits (yard markers, chain-crew
  signs, field numbers); digit reads must be gated to player torsos.
- The roster-multiset cross-check is strong for two-digit reads and weak for
  single-digit ones: a stray digit only has to match one of the ~3
  single-digit numbers on the field, and those belong to QBs, WRs and DBs —
  exactly the players who dominate closeups. Single-digit reads need their
  own evidence (experiment 9).
- A crop's label must answer "what number is *this* player wearing?", not
  "is some on-roster number visible in this picture?" Neighbouring players
  intrude at the edges of a detection box; a read's geometry matters as much
  as its content.
- Verify a label set by looking at sampled crops before training on it. Both
  label-quality bugs found so far (eval v0, pseudo v1) were invisible in the
  aggregate statistics and obvious in a 12-crop contact sheet.

## Conventions

- Commits: imperative subject, body explains the *why* and records results
  if the change embodies an experiment. Don't commit `data/` or weights.
- Pushing: `git -c credential.helper='!gh auth git-credential' push`
  (gh CLI holds the GitHub token; plain `git push` has no credentials).
- The user gates commits and pushes — propose, don't auto-commit.
