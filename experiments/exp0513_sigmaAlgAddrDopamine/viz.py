"""Convenience launcher for the exp0513 dashboard."""

from __future__ import annotations

import argparse
import socket

from server import serve_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the exp0513 dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind. Defaults to 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8000, help="Preferred port. Defaults to 8000.")
    parser.add_argument(
        "--port-span",
        type=int,
        default=25,
        help="How many consecutive ports to try when the preferred port is busy. Defaults to 25.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Start the dashboard server without opening the browser automatically.",
    )
    return parser.parse_args()


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def choose_port(host: str, preferred_port: int, port_span: int) -> int:
    for offset in range(max(port_span, 1)):
        port = preferred_port + offset
        if port_is_free(host, port):
            return port
    raise SystemExit(
        f"Could not find a free port in range {preferred_port}-{preferred_port + max(port_span, 1) - 1}. "
        "Try `python3 viz.py --port 9000`."
    )


def main() -> None:
    args = parse_args()
    chosen_port = choose_port(args.host, args.port, args.port_span)
    if chosen_port != args.port:
        print(f"Preferred port {args.port} is busy; using {chosen_port} instead.")
    serve_dashboard(host=args.host, port=chosen_port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
