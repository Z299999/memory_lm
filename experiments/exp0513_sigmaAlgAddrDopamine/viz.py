"""Serve the exp0513 visualization page from a tiny local HTTP server."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the exp0513 visualization page.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind. Defaults to 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on. Defaults to 8000.")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the visualization in your default browser after the server starts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    address = (args.host, args.port)
    url = f"http://{args.host}:{args.port}/viz/"

    try:
        with ThreadingHTTPServer(address, handler) as server:
            print(f"Serving exp0513 viz at {url}")
            print("Press Ctrl+C to stop.")
            if args.open:
                webbrowser.open(url)
            server.serve_forever()
    except PermissionError as exc:
        raise SystemExit(
            f"Could not bind to {args.host}:{args.port}. "
            "Try a different port, for example `python3 viz.py --port 8765`."
        ) from exc
    except OSError as exc:
        raise SystemExit(
            f"Could not start server on {args.host}:{args.port}: {exc}. "
            "Try a different port."
        ) from exc


if __name__ == "__main__":
    main()
