#!/usr/bin/env python3
"""
Minimal API server for the eco‑evo dashboard.

Step 2: WebSocket endpoint that understands simple JSON commands.

Supported messages from the frontend (text frames, JSON encoded):

- {"type": "ping"}
    -> server responds once with {"type": "pong"}

- {"type": "run_segment", "start_day": <int>, "length": <int>}
    -> server streams a short fake metrics time‑series:
       {"type": "metrics", "day": d, "N": ..., "M": ...}
       and finally
       {"type": "segment_done", "start_day": ..., "end_day": ...}

Later we will replace the fake metrics with real data from Simulation.
Run:
    python api_server.py
"""

from __future__ import annotations

from typing import Any, Optional
import asyncio
import json
from pathlib import Path
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Make src importable (Config, Simulation).
sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import Config
from sim import Simulation


app = FastAPI(title="EcoEvo exp2-2 API")

# Allow the static dashboard (served from http://localhost:8000) to talk to us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Simple health check."""
    return {"status": "ok"}


_SIMULATION: Optional[Simulation] = None


def get_simulation() -> Simulation:
    """
    Lazily construct a Simulation instance using config.yaml.

    For now we keep a single long-lived Simulation per API process.
    """

    global _SIMULATION
    if _SIMULATION is None:
        base_dir = Path(__file__).parent
        cfg_path = base_dir / "config.yaml"
        config = Config.from_yaml(cfg_path)
        # For interactive runs via the dashboard, disable auto post-processing
        # plots and videos to keep things responsive.
        config.plot_after_run = False
        config.make_video = False
        _SIMULATION = Simulation(config)
    return _SIMULATION


async def _handle_ping(ws: WebSocket) -> None:
  """Reply to a ping with a pong."""
  await ws.send_text(json.dumps({"type": "pong"}))


async def _handle_run_segment(ws: WebSocket, start_day: int, length: int) -> None:
  """
  Advance the real Simulation by `length` days, streaming metrics.
  """

  if length <= 0:
    return

  sim = get_simulation()

  # If the requested start_day is behind the current sim.day, we cannot
  # go backwards; for now we just ignore start_day and continue forward.
  # Later we can wire this to checkpoint loading.
  target_end_day = sim.day + max(0, length)

  while sim.day < target_end_day:
    # Run one simulation day; if it returns False, the population is extinct
    # or the simulation decided to stop early.
    should_continue = await asyncio.to_thread(sim.step_day)
    latest = sim.metrics.get_latest()
    if latest is not None:
      payload: dict[str, Any] = {
        "type": "metrics",
        "day": latest.t,
        "N": latest.N,
        "M": latest.M,
        "eta": latest.eta_window,
      }
      await ws.send_text(json.dumps(payload))

    if not should_continue:
      break

  end_msg = {
    "type": "segment_done",
    "start_day": start_day,
    "end_day": sim.day,
  }
  await ws.send_text(json.dumps(end_msg))


@app.websocket("/ws")
async def websocket_main(ws: WebSocket) -> None:
  """
  Main WebSocket handler.

  Expects each text frame to be a JSON object with a "type" field.
  """

  await ws.accept()
  try:
    while True:
      raw = await ws.receive_text()
      try:
        msg = json.loads(raw)
      except json.JSONDecodeError:
        # Ignore non‑JSON frames for now.
        continue

      msg_type = msg.get("type")
      if msg_type == "ping":
        await _handle_ping(ws)
      elif msg_type == "run_segment":
        start_day = int(msg.get("start_day", 0) or 0)
        length = int(msg.get("length", 10) or 10)
        await _handle_run_segment(ws, start_day=start_day, length=length)
      else:
        # Unknown type: for now just echo it back so the frontend can see it.
        await ws.send_text(raw)
  except WebSocketDisconnect:
    # Client disconnected; just exit the loop.
    return


def main() -> None:
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
