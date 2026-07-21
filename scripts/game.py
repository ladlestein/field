"""Per-game paths and nflverse table access.

One game = one nflverse game_id (e.g. "2025_15_WAS_NYG") and one directory
under data/games/<game_id>/ holding everything derived from that broadcast:

    frames/             1 fps jpgs from extract_frames.py
    manifest.parquet    per-frame sweep output
    plays.csv           per-play representative frames + number multisets
    crops/<policy>/     harvested torso crops + crops.parquet
    eval/               labeling sheets

nflverse parquets are season-level and shared across games in data/nflverse/;
the season is parsed from the game_id.
"""
import argparse
from pathlib import Path

import polars as pl

NFLVERSE = Path("data/nflverse")
DEFAULT_GAME = "2025_15_WAS_NYG"


class Game:
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.season = int(game_id.split("_")[0])
        self.dir = Path("data/games") / game_id
        self.frames_dir = self.dir / "frames"
        self.manifest_path = self.dir / "manifest.parquet"
        self.plays_path = self.dir / "plays.csv"
        self.eval_dir = self.dir / "eval"

    def crops_dir(self, policy: str) -> Path:
        return self.dir / "crops" / policy

    def pbp(self) -> pl.DataFrame:
        return pl.read_parquet(
            NFLVERSE / f"play_by_play_{self.season}.parquet"
        ).filter(pl.col("game_id") == self.game_id)

    def participation(self) -> pl.DataFrame:
        return pl.read_parquet(
            NFLVERSE / f"pbp_participation_{self.season}.parquet"
        ).filter(pl.col("nflverse_game_id") == self.game_id)


def add_game_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--game", default=DEFAULT_GAME,
                    help=f"nflverse game_id (default {DEFAULT_GAME})")


def torch_device() -> str:
    """Apple-silicon GPU (mps) when available; FIELD_DEVICE env overrides."""
    import os

    import torch

    override = os.environ.get("FIELD_DEVICE")
    if override:
        return override
    return "mps" if torch.backends.mps.is_available() else "cpu"


def easyocr_gpu():
    """easyocr's Reader takes False for CPU or a torch device string."""
    dev = torch_device()
    return dev if dev != "cpu" else False
