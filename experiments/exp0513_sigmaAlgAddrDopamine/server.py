"""Local dashboard server for exp0513."""

from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import urlparse
import webbrowser

from src.config import ExperimentConfig, config_from_user_dict, load_config_from_yaml
from src.data import available_task_names
from src.assignment import (
    build_dopamine_assignment,
    build_forward_edge_records,
    build_graph_payload,
    resolve_dopamine_m,
)
from src.model import SelfModulatedMLP
from src.train import load_experiment_checkpoint, make_run_dir, run_experiment


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


class DashboardState:
    """Single-run dashboard state manager."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock = threading.Lock()
        self.active_run: dict[str, Any] | None = None
        self.worker_thread: threading.Thread | None = None
        self.stop_event: threading.Event | None = None

    def load_default_config(self) -> dict[str, Any]:
        config = load_config_from_yaml(DEFAULT_CONFIG_PATH)
        return {
            "config": config.to_user_dict(),
            "task_names": available_task_names(),
            "graph_payload": self._build_preview_payload(config)["graph_payload"],
        }

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            payload = {
                "active_run": deepcopy(self.active_run),
                "has_active_run": self.active_run is not None,
                "is_running": self.worker_thread is not None and self.worker_thread.is_alive(),
            }
        return payload

    def request_stop(self) -> dict[str, Any]:
        with self.lock:
            if self.worker_thread is None or not self.worker_thread.is_alive() or self.stop_event is None:
                raise RuntimeError("No active training run to stop.")
            self.stop_event.set()
            if self.active_run is not None:
                self.active_run["status"] = "stopping"
        return self.get_state()

    def start_run(self, user_payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.worker_thread is not None and self.worker_thread.is_alive():
                raise RuntimeError("A training run is already in progress.")

        config = config_from_user_dict(user_payload, base_dir=self.root)
        preview = self._build_preview_payload(config)
        run_dir = make_run_dir(self.root / "runs", config.run_name)
        live_status_path = run_dir / "live_viewer.json"

        stop_event = threading.Event()
        initial_state = {
            "run_name": config.run_name,
            "task_name": config.task_name,
            "status": "starting",
            "epoch": 0,
            "epochs_total": config.epochs,
            "local_epoch": 0,
            "local_epochs_total": config.epochs,
            "global_epoch": preview["global_epoch_completed_before"],
            "global_epoch_start": preview["global_epoch_completed_before"] + 1,
            "global_epoch_end": preview["global_epoch_completed_before"] + config.epochs,
            "lambda": config.lambda_value,
            "train_loss": None,
            "val_loss": None,
            "best_val_loss": None,
            "seed": config.seed,
            "m": preview["dopamine_m"],
            "N": preview["edge_count"],
            "dopamine_m": preview["dopamine_m"],
            "recommended_dopamine_m": preview["recommended_dopamine_m"],
            "coverage_c": preview["coverage_c"],
            "resume_from": config.resume_from,
            "run_dir": str(run_dir),
            "updated_at": None,
            "graph_payload": preview["graph_payload"],
        }

        with self.lock:
            self.active_run = initial_state
            self.stop_event = stop_event

        worker = threading.Thread(
            target=self._run_training,
            args=(config, run_dir, live_status_path, stop_event),
            daemon=True,
            name="exp0513-dashboard-worker",
        )

        with self.lock:
            self.worker_thread = worker
        worker.start()
        return self.get_state()

    def _build_preview_payload(self, config: ExperimentConfig) -> dict[str, Any]:
        model = SelfModulatedMLP()
        edge_records = build_forward_edge_records(model)

        if config.resume_from:
            checkpoint = load_experiment_checkpoint(config.resume_from)
            graph_payload = dict(checkpoint.get("graph_payload") or {})
            return {
                "graph_payload": graph_payload,
                "dopamine_m": int(checkpoint["effective_dopamine_m"]),
                "recommended_dopamine_m": int(checkpoint.get("recommended_dopamine_m", checkpoint["effective_dopamine_m"])),
                "coverage_c": int(checkpoint["coverage_c"]),
                "edge_count": int(checkpoint["edge_count"]),
                "global_epoch_completed_before": int(checkpoint.get("global_epoch_completed", 0)),
            }

        dopamine_m, recommended_dopamine_m = resolve_dopamine_m(
            coverage_c=config.coverage_c,
            hidden_pool_size=model.hidden_pool_size(),
            dopamine_m_override=config.dopamine_m_override,
        )
        _, _, assignment_metadata = build_dopamine_assignment(
            edge_records=edge_records,
            hidden_node_ids=model.hidden_node_ids(),
            coverage_c=config.coverage_c,
            dopamine_m=dopamine_m,
            seed=config.seed,
        )
        return {
            "graph_payload": build_graph_payload(model, assignment_metadata),
            "dopamine_m": dopamine_m,
            "recommended_dopamine_m": recommended_dopamine_m,
            "coverage_c": config.coverage_c,
            "edge_count": len(edge_records),
            "global_epoch_completed_before": 0,
        }

    def _update_live_state(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.active_run = deepcopy(payload)

    def _run_training(
        self,
        config: ExperimentConfig,
        run_dir: Path,
        live_status_path: Path,
        stop_event: threading.Event,
    ) -> None:
        try:
            run_experiment(
                config=config,
                run_dir=run_dir,
                live_status_path=live_status_path,
                stop_requested=stop_event.is_set,
                status_callback=self._update_live_state,
            )
        except Exception as exc:  # pragma: no cover - exercised via manual dashboard use
            with self.lock:
                current = deepcopy(self.active_run) or {}
                current["status"] = "failed"
                current["error"] = str(exc)
                self.active_run = current
        finally:
            with self.lock:
                self.worker_thread = None
                self.stop_event = None


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    """Serve the dashboard page plus a minimal JSON API."""

    server_version = "exp0513-dashboard/0.1"

    @property
    def dashboard_state(self) -> DashboardState:
        return self.server.dashboard_state  # type: ignore[attr-defined]

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/viz/")
            self.end_headers()
            return
        if parsed.path == "/api/default-config":
            self._send_json(self.dashboard_state.load_default_config())
            return
        if parsed.path == "/api/state":
            self._send_json(self.dashboard_state.get_state())
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            try:
                payload = self._read_json_body()
                self._send_json(self.dashboard_state.start_run(payload))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except RuntimeError as exc:
                self._send_json({"error": str(exc), **self.dashboard_state.get_state()}, status=HTTPStatus.CONFLICT)
            return

        if parsed.path == "/api/stop":
            try:
                self._send_json(self.dashboard_state.request_stop())
            except RuntimeError as exc:
                self._send_json({"error": str(exc), **self.dashboard_state.get_state()}, status=HTTPStatus.CONFLICT)
            return

        self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)


class DashboardHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying dashboard state."""

    def __init__(self, server_address: tuple[str, int], handler_class, directory: str, dashboard_state: DashboardState):
        self.dashboard_state = dashboard_state
        self.directory = directory
        super().__init__(server_address, handler_class)


def serve_dashboard(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = False) -> None:
    """Start the dashboard server and optionally open the browser."""
    state = DashboardState(ROOT)
    handler = partial(DashboardRequestHandler, directory=str(ROOT))
    url = f"http://{host}:{port}/viz/"

    try:
        with DashboardHTTPServer((host, port), handler, str(ROOT), state) as server:
            print(f"Serving exp0513 dashboard at {url}")
            print("Press Ctrl+C to stop.")
            if open_browser:
                webbrowser.open(url)
            server.serve_forever()
    except PermissionError as exc:
        raise SystemExit(
            f"Could not bind to {host}:{port}. Try a different port, for example `python3 server.py --port 8765`."
        ) from exc
    except OSError as exc:
        raise SystemExit(f"Could not start dashboard server on {host}:{port}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the exp0513 local dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind. Defaults to 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on. Defaults to 8000.")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in a browser after startup.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    serve_dashboard(host=args.host, port=args.port, open_browser=args.open)


if __name__ == "__main__":
    main()
