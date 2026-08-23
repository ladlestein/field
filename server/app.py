"""Live viewer: play a recorded broadcast beside real-time predictions.

One process serves the whole thing over HTTP on localhost:
  /        the viewer page (scripts/viewer.html)
  /video   the broadcast file, with range support so <video> can seek
  /ws      WebSocket carrying clock updates in and predictions out

The browser's <video> element owns the playback clock; its transport
controls are the one set of start/stop buttons. The page reports the
current playback time over the WebSocket, and the prediction worker
always works on the *newest* reported time — never a queue — so a slow
inference tick costs coverage, not freshness (a prediction that lands
after the snap is worthless).

v0 predictions: score-bug OCR (parse_bug), nflverse play alignment
(match_play), and the aligned play's participation lists + personnel
grouping. The participation panel is what alignment unlocks, not a
visual prediction — the page labels it as such. Vision-side player ID
and formation predictors plug into predict_one() as they mature.

Usage: .venv/bin/python3 server/app.py [--game GAME_ID] [--video PATH]
"""
import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

from aiohttp import WSMsgType, web

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from game import Game, add_game_arg, easyocr_gpu
from scorebug_align import match_play, parse_bug
# The sweep's tight-ROI OCR, not scorebug_align's full-width strip: same
# recognizer, but the crop/scale the whole manifest was validated with.
# The full-strip variant misses the clock on frames the ROI reads fine.
from sweep_broadcast import ocr_roi_tokens

DEFAULT_VIDEO = "data/commanders_giants_week_15_2025_full.mp4"
VIEWER_HTML = REPO_ROOT / "viewer" / "index.html"

# Re-predict when the reported playback time has moved this far from the
# last predicted time (forward or backward, so seeks re-trigger).
PREDICT_STEP_S = 1.0


class Predictor:
    """Blocking model state, used only from the worker thread."""

    def __init__(self, game: Game, video_path: Path):
        import cv2
        import easyocr

        self.cv2 = cv2
        self.reader = easyocr.Reader(["en"], gpu=easyocr_gpu(), verbose=False)
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open video: {video_path}")
        self.pbp = game.pbp()
        self.part = game.participation()

    def frame_at(self, t: float):
        self.cap.set(self.cv2.CAP_PROP_POS_MSEC, max(t, 0.0) * 1000)
        ok, img = self.cap.read()
        return img if ok else None

    def participation_for(self, play_id) -> dict | None:
        import polars as pl

        row = self.part.filter(pl.col("play_id") == play_id)
        if row.height == 0:
            return None
        r = row.row(0, named=True)
        out = {}
        for side in ("offense", "defense"):
            players = sorted(
                zip(
                    r[f"{side}_numbers"].split(";"),
                    r[f"{side}_positions"].split(";"),
                    r[f"{side}_names"].split(";"),
                ),
                key=lambda p: int(p[0]),
            )
            out[side] = [{"num": n, "pos": p, "name": nm} for n, p, nm in players]
        return out

    def predict_one(self, t: float) -> dict:
        """One inference tick at playback time t. Blocking; worker thread only."""
        started = time.time()
        msg: dict = {"type": "prediction", "t": t}

        img = self.frame_at(t)
        if img is None:
            msg["error"] = "no frame at this time"
            return msg

        tokens = ocr_roi_tokens(self.reader, img)
        state = parse_bug(tokens)
        msg["bug"] = {k: state[k] for k in ("clock", "qtr", "down", "ydstogo")}
        msg["tokens"] = [tk["text"] for tk in tokens]

        play, note = match_play(self.pbp, state)
        msg["note"] = note
        if play is not None:
            msg["play"] = {
                "play_id": int(play["play_id"]),
                "qtr": play["qtr"],
                "time": play["time"],
                "down": play["down"],
                "ydstogo": play["ydstogo"],
                "yrdln": play["yrdln"],
                "posteam": play["posteam"],
                "defteam": play["defteam"],
                "desc": play["desc"],
            }
            part = self.participation_for(play["play_id"])
            if part is not None:
                msg["participation"] = part
                msg["personnel"] = personnel_group(part["offense"])

        msg["latency_ms"] = int((time.time() - started) * 1000)
        return msg


def personnel_group(offense: list[dict]) -> str | None:
    """Offense personnel as the usual 2-digit code: #RB then #TE."""
    counts = {"RB": 0, "TE": 0}
    for p in offense:
        pos = p["pos"]
        if pos in ("RB", "FB", "HB"):
            counts["RB"] += 1
        elif pos == "TE":
            counts["TE"] += 1
    return f"{counts['RB']}{counts['TE']}"


class Engine:
    """Bridges the asyncio side (websockets) and the worker thread (models).

    The worker loop wakes whenever the target time moves, predicts at the
    newest target, and drops everything in between.
    """

    def __init__(self, game: Game, video_path: Path):
        self.game = game
        self.video_path = video_path
        self.clients: set[web.WebSocketResponse] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.wake = threading.Event()
        self.lock = threading.Lock()
        self.target_t: float | None = None
        self.playing = False
        self.predict_on = True
        self.last_predicted_t: float | None = None

    def set_clock(self, t: float, playing: bool):
        with self.lock:
            self.target_t = t
            self.playing = playing
        self.wake.set()

    def set_predict(self, on: bool):
        with self.lock:
            self.predict_on = on
            self.last_predicted_t = None  # re-predict current spot on re-enable
        self.wake.set()

    def worker(self):
        predictor = Predictor(self.game, self.video_path)
        self._broadcast({"type": "status", "ready": True})
        while True:
            self.wake.wait()
            self.wake.clear()
            while True:
                # Predict while paused too: a seek should refresh the panel.
                # The PREDICT_STEP_S dedup keeps a paused video from
                # re-predicting the same spot in a busy loop.
                with self.lock:
                    t, on = self.target_t, self.predict_on
                    last = self.last_predicted_t
                if (
                    t is None
                    or not on
                    or (last is not None and abs(t - last) < PREDICT_STEP_S)
                ):
                    break
                with self.lock:
                    self.last_predicted_t = t
                self._broadcast(predictor.predict_one(t))
                # loop: target may have moved while we were predicting

    def _broadcast(self, msg: dict):
        if self.loop is None:
            return
        data = json.dumps(msg, default=str)
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(self._send(ws, data), self.loop)

    async def _send(self, ws: web.WebSocketResponse, data: str):
        try:
            await ws.send_str(data)
        except ConnectionResetError:
            self.clients.discard(ws)


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    engine: Engine = request.app["engine"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    engine.clients.add(ws)
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            if data.get("type") == "clock":
                engine.set_clock(float(data["t"]), bool(data["playing"]))
            elif data.get("type") == "predict":
                engine.set_predict(bool(data["on"]))
    finally:
        engine.clients.discard(ws)
    return ws


async def index(request: web.Request) -> web.Response:
    html = VIEWER_HTML.read_text()
    info = {"game_id": request.app["engine"].game.game_id,
            "video": request.app["engine"].video_path.name}
    html = html.replace("__GAME_INFO__", json.dumps(info))
    return web.Response(text=html, content_type="text/html")


async def video(request: web.Request) -> web.FileResponse:
    return web.FileResponse(request.app["engine"].video_path)


def main():
    ap = argparse.ArgumentParser()
    add_game_arg(ap)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"video not found: {video_path}")

    engine = Engine(Game(args.game), video_path)
    app = web.Application()
    app["engine"] = engine
    app.add_routes([
        web.get("/", index),
        web.get("/video", video),
        web.get("/ws", ws_handler),
    ])

    async def on_startup(app):
        engine.loop = asyncio.get_running_loop()
        threading.Thread(target=engine.worker, daemon=True).start()

    app.on_startup.append(on_startup)
    print(f"viewer: http://{args.host}:{args.port}/  "
          f"(game {engine.game.game_id}, video {video_path.name})")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
