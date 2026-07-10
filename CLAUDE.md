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
   frames; writes `data/harvest/manifest.parquet` (per frame) and
   `data/harvest/plays.csv` (one representative pre-snap frame per play,
   with jersey-number multisets). This manifest is the durable label source
   for training; crops are cheap derived artifacts, safe to re-cut.
5. `scripts/read_jerseys.py`, `scripts/localize_recognize.py`,
   `scripts/sr_stack.py` — experiment scripts (see EXPERIMENTS.md entries
   2-4); superseded in parts but kept as baselines.

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

## Conventions

- Commits: imperative subject, body explains the *why* and records results
  if the change embodies an experiment. Don't commit `data/` or weights.
- Pushing: `git -c credential.helper='!gh auth git-credential' push`
  (gh CLI holds the GitHub token; plain `git push` has no credentials).
- The user gates commits and pushes — propose, don't auto-commit.
