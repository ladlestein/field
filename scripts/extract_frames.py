"""Extract frames from a game video via ffmpeg.

Two modes, matching the two ways this project consumes video:

  1 fps sweep frames (default): one jpg per second of video, named t_NNNNN.jpg
  where index N covers video time [N-1, N) seconds -- the convention
  sweep_broadcast.py expects (t_sec = index - 1).

  Full-rate windows (--start/--duration with --native-fps): every frame of a
  short clip as png (lossless, so later processing isn't stacked on a second
  round of jpg compression), named f_NNNN.png. Used for e.g. the multi-frame
  SR experiment.

Usage:
  .venv/bin/python scripts/extract_frames.py data/commanders_giants_week_15_2025_full.mp4 data/harvest/frames
  .venv/bin/python scripts/extract_frames.py data/..._full.mp4 data/sr_experiment/frames \
      --start 1205 --duration 9 --native-fps
"""
import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--fps", type=float, default=1.0, help="sampling rate (default 1)")
    ap.add_argument("--native-fps", action="store_true",
                    help="keep every frame (ignores --fps); writes lossless png")
    ap.add_argument("--start", type=float, default=None, help="start time in seconds")
    ap.add_argument("--duration", type=float, default=None, help="clip length in seconds")
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"no such video: {args.video}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-v", "error"]
    if args.start is not None:
        cmd += ["-ss", str(args.start)]
    if args.duration is not None:
        cmd += ["-t", str(args.duration)]
    cmd += ["-i", str(args.video)]
    if args.native_fps:
        pattern = args.out_dir / "f_%04d.png"
    else:
        cmd += ["-vf", f"fps={args.fps}", "-q:v", "2"]
        pattern = args.out_dir / "t_%05d.jpg"
    cmd.append(str(pattern))

    subprocess.run(cmd, check=True)
    n = len(list(args.out_dir.glob(pattern.name.replace("%04d", "*").replace("%05d", "*"))))
    print(f"{n} frames in {args.out_dir}")


if __name__ == "__main__":
    main()
