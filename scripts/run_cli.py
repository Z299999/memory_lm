from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_RUNNER = ROOT / "scripts" / "run_experiment.py"
RUNS_DIR = ROOT / "runs"
RUNS_LOG_PATH = RUNS_DIR / "log.md"


def parse_scalar(value: str):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict:
    data: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, _, value = stripped.partition(":")
        data[key.strip()] = parse_scalar(value.strip())
    return data


def merged_metrics_config(run_dir: Path, metrics: dict) -> dict:
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        metrics = dict(metrics)
        metrics["config"] = load_simple_yaml(config_path)
        completed = int(metrics.get("totals", {}).get("completed_rounds", 0))
        planned = int(metrics.get("config", {}).get("rounds", 0) or 0)
        if planned and completed < planned and metrics.get("status") == "completed":
            metrics["status"] = "paused"
            metrics["ended_at"] = None
    return metrics


def try_set_ssl_cert_file() -> None:
    if os.getenv("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


def run_runner(extra_args: list[str]) -> int:
    try_set_ssl_cert_file()
    cmd = [sys.executable, str(SCRIPTS_RUNNER), *extra_args]
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def run_defaults(
    *,
    action: str,
    run_id: str | None,
    world_path: str,
    rounds: int,
    memory_budget: int,
    tested_model: str,
    host_model: str,
    host_language: str,
    auto_accept_host: bool,
    stub_llm: bool,
) -> int:
    if action == "new":
        runner_args = [
            "--world-path",
            world_path,
            "--rounds",
            str(rounds),
            "--memory-budget",
            str(memory_budget),
            "--tested-model",
            tested_model,
            "--host-model",
            host_model,
            "--host-language",
            host_language,
        ]
        if run_id:
            runner_args += ["--run-id", run_id]
        if auto_accept_host:
            runner_args.append("--auto-accept-host")
        if stub_llm:
            runner_args.append("--stub-llm")
        return run_runner(runner_args)

    if action == "resume":
        runner_args = ["--run-id", run_id or "run_001"]
        if auto_accept_host:
            runner_args.append("--auto-accept-host")
        if stub_llm:
            runner_args.append("--stub-llm")
        return run_runner(runner_args)

    if action == "smoke":
        runner_args = ["--stub-llm", "--auto-accept-host", "--world-path", world_path, "--rounds", str(rounds)]
        if run_id:
            runner_args += ["--run-id", run_id]
        return run_runner(runner_args)

    if action == "status":
        return print_run_status(run_id) if run_id else print_all_status()

    latest_run_id = find_latest_run_id()
    if not latest_run_id:
        print("No existing run found to resume.", file=sys.stderr)
        return 1
    runner_args = ["--run-id", latest_run_id]
    if auto_accept_host:
        runner_args.append("--auto-accept-host")
    if stub_llm:
        runner_args.append("--stub-llm")
    return run_runner(runner_args)


def find_latest_run_id() -> str | None:
    latest_incomplete: tuple[str, str] | None = None
    latest_any: tuple[str, str] | None = None
    if not RUNS_DIR.exists():
        return None

    for child in RUNS_DIR.iterdir():
        if not child.is_dir():
            continue
        metrics_path = child / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metrics = merged_metrics_config(child, metrics)
        started_at = metrics.get("started_at") or ""
        candidate = (started_at, child.name)
        if latest_any is None or candidate > latest_any:
            latest_any = candidate
        status = metrics.get("status")
        totals = metrics.get("totals", {})
        completed = int(totals.get("completed_rounds", 0))
        planned = int(metrics.get("config", {}).get("rounds", 0) or 0)
        is_incomplete = planned == 0 or completed < planned or status not in {"completed"}
        if is_incomplete and (latest_incomplete is None or candidate > latest_incomplete):
            latest_incomplete = candidate
    chosen = latest_incomplete or latest_any
    return chosen[1] if chosen else None


def format_status_line(metrics: dict) -> str:
    totals = metrics.get("totals", {})
    completed = int(totals.get("completed_rounds", 0))
    correct = int(totals.get("correct_rounds", 0))
    status = metrics.get("status", "unknown")
    planned = metrics.get("config", {}).get("rounds", "unknown")
    return f"{metrics.get('run_id', 'unknown_run')}: status={status}, rounds={completed}/{planned}, accuracy={correct}/{completed if completed else 0}"


def print_run_status(run_id: str) -> int:
    metrics_path = RUNS_DIR / run_id / "metrics.json"
    if not metrics_path.exists():
        print(f"Run not found: {run_id}", file=sys.stderr)
        return 1

    metrics = merged_metrics_config(RUNS_DIR / run_id, json.loads(metrics_path.read_text(encoding="utf-8")))
    config = metrics.get("config", {})
    totals = metrics.get("totals", {})
    completed = int(totals.get("completed_rounds", 0))
    correct = int(totals.get("correct_rounds", 0))
    accuracy = (correct / completed) if completed else 0.0

    print(f"run_id: {metrics.get('run_id')}")
    print(f"status: {metrics.get('status')}")
    print(f"world: {config.get('world_path')}")
    print(f"tested_model: {config.get('tested_model')}")
    print(f"host_model: {config.get('host_model')}")
    print(f"host_language: {config.get('host_language', 'zh')}")
    print(f"rounds: {completed}/{config.get('rounds')}")
    print(f"accuracy: {correct}/{completed} ({accuracy:.0%})" if completed else "accuracy: 0/0 (n/a)")
    print(f"transcript: runs/{run_id}/transcript.md")
    print(f"metrics: runs/{run_id}/metrics.json")
    return 0


def print_all_status() -> int:
    if RUNS_LOG_PATH.exists():
        print(RUNS_LOG_PATH.read_text(encoding="utf-8").rstrip())
        return 0

    any_runs = False
    for child in sorted(RUNS_DIR.iterdir()):
        metrics_path = child / "metrics.json"
        if child.is_dir() and metrics_path.exists():
            any_runs = True
            metrics = merged_metrics_config(child, json.loads(metrics_path.read_text(encoding="utf-8")))
            print(format_status_line(metrics))
    if not any_runs:
        print("No runs found yet.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convenience wrapper for the memory LM experiment runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser("new", help="Create and start a new run.")
    new_parser.add_argument("--run-id")
    new_parser.add_argument("--world-path", default="world/asterion_lab.md")
    new_parser.add_argument("--rounds", type=int, default=30)
    new_parser.add_argument("--memory-budget", type=int, default=1000)
    new_parser.add_argument("--tested-model", default="qwen3-coder-plus")
    new_parser.add_argument("--host-model", default="qwen3-coder-plus")
    new_parser.add_argument("--host-language", choices=["zh", "en"], default="zh")
    new_parser.add_argument("--host-temperature", type=float, default=0.3)
    new_parser.add_argument("--tested-temperature", type=float, default=0.2)
    new_parser.add_argument("--host-max-tokens", type=int, default=1000)
    new_parser.add_argument("--tested-max-tokens", type=int, default=500)
    new_parser.add_argument("--auto-accept-host", action="store_true")
    new_parser.add_argument("--stub-llm", action="store_true")

    resume_parser = subparsers.add_parser("resume", help="Resume an existing run by run id.")
    resume_parser.add_argument("run_id")
    resume_parser.add_argument("--auto-accept-host", action="store_true")
    resume_parser.add_argument("--stub-llm", action="store_true")

    latest_parser = subparsers.add_parser("resume-latest", help="Resume the most recent incomplete run.")
    latest_parser.add_argument("--auto-accept-host", action="store_true")
    latest_parser.add_argument("--stub-llm", action="store_true")

    smoke_parser = subparsers.add_parser("smoke", help="Run a quick offline smoke test.")
    smoke_parser.add_argument("--run-id")
    smoke_parser.add_argument("--world-path", default="world/asterion_lab.md")
    smoke_parser.add_argument("--rounds", type=int, default=3)

    status_parser = subparsers.add_parser("status", help="Show run status or print runs/log.md.")
    status_parser.add_argument("run_id", nargs="?")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "new":
        runner_args = [
            "--world-path",
            args.world_path,
            "--rounds",
            str(args.rounds),
            "--memory-budget",
            str(args.memory_budget),
            "--tested-model",
            args.tested_model,
            "--host-model",
            args.host_model,
            "--host-language",
            args.host_language,
            "--host-temperature",
            str(args.host_temperature),
            "--tested-temperature",
            str(args.tested_temperature),
            "--host-max-tokens",
            str(args.host_max_tokens),
            "--tested-max-tokens",
            str(args.tested_max_tokens),
        ]
        if args.run_id:
            runner_args += ["--run-id", args.run_id]
        if args.auto_accept_host:
            runner_args.append("--auto-accept-host")
        if args.stub_llm:
            runner_args.append("--stub-llm")
        return run_runner(runner_args)

    if args.command == "resume":
        runner_args = ["--run-id", args.run_id]
        if args.auto_accept_host:
            runner_args.append("--auto-accept-host")
        if args.stub_llm:
            runner_args.append("--stub-llm")
        return run_runner(runner_args)

    if args.command == "resume-latest":
        latest_run_id = find_latest_run_id()
        if not latest_run_id:
            print("No existing run found to resume.", file=sys.stderr)
            return 1
        runner_args = ["--run-id", latest_run_id]
        if args.auto_accept_host:
            runner_args.append("--auto-accept-host")
        if args.stub_llm:
            runner_args.append("--stub-llm")
        return run_runner(runner_args)

    if args.command == "smoke":
        runner_args = [
            "--stub-llm",
            "--auto-accept-host",
            "--world-path",
            args.world_path,
            "--rounds",
            str(args.rounds),
        ]
        if args.run_id:
            runner_args += ["--run-id", args.run_id]
        return run_runner(runner_args)

    if args.command == "status":
        if args.run_id:
            return print_run_status(args.run_id)
        return print_all_status()

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
