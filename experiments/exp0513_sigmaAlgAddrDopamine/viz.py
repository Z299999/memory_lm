"""Convenience launcher for the exp0513 dashboard."""

from __future__ import annotations

import argparse

from server import serve_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the exp0513 dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind. Defaults to 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on. Defaults to 8000.")
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Start the dashboard server without opening the browser automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    serve_dashboard(host=args.host, port=args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
