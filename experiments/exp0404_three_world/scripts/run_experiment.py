from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_client import LLMClientError, OpenAICompatClient
from memory_io import (
    clip_memory,
    merge_memory_block,
    read_memory,
    resolve_round_memory_path,
    round_memory_path,
    write_memory,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORLD_PATH = ROOT / "world" / "asterion_lab.md"
HOST_PROMPT_PATH = ROOT / "prompts" / "host.md"
TESTED_PROMPT_PATH = ROOT / "prompts" / "tested_agent.md"
RUNS_DIR = ROOT / "runs"
RUNS_LOG_PATH = RUNS_DIR / "log.md"

HOST_SECTION_PATTERN = re.compile(
    r"## AGENT_INPUT\s*(.*?)\s*## CANONICAL_ANSWER\s*(.*?)\s*## SCORING_RATIONALE\s*(.*?)\s*## NEXT_ROUND_INTENT\s*(.*)",
    re.DOTALL,
)
JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
LABEL_PATTERN = re.compile(r"\b(SAFE|DANGEROUS)\b", re.IGNORECASE)
ASCII_WORD_PATTERN = re.compile(r"\b[A-Za-z]{4,}\b")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the memory evolution experiment.")
    parser.add_argument("--run-id", help="Existing or new run id, e.g. run_001.")
    parser.add_argument("--world-path", default=str(DEFAULT_WORLD_PATH.relative_to(ROOT)))
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--memory-budget", type=int, default=1000)
    parser.add_argument("--memory-mode", choices=["rewrite", "prepend"], default="prepend")
    parser.add_argument("--tested-model", default="qwen-plus")
    parser.add_argument("--host-model", default="qwen-plus")
    parser.add_argument("--host-language", choices=["zh", "en"], default="zh")
    parser.add_argument("--host-temperature", type=float, default=0.3)
    parser.add_argument("--tested-temperature", type=float, default=0.2)
    parser.add_argument("--host-max-tokens", type=int, default=1000)
    parser.add_argument("--tested-max-tokens", type=int, default=500)
    parser.add_argument("--stub-llm", action="store_true")
    parser.add_argument("--auto-accept-host", action="store_true")
    return parser.parse_args()


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def resolve_world_path(world_path: str) -> Path:
    candidate = ROOT / world_path
    if candidate.exists():
        return candidate
    legacy_map = {
        "world/world.md": DEFAULT_WORLD_PATH,
    }
    fallback = legacy_map.get(world_path)
    if fallback and fallback.exists():
        return fallback
    return candidate


def render_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def effective_memory_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("memory_mode", "prepend")).strip().lower()
    return mode if mode in {"rewrite", "prepend"} else "prepend"


def normalize_answer(answer_text: str) -> str | None:
    match = LABEL_PATTERN.search(answer_text)
    if not match:
        return None
    return match.group(1).upper()


def language_rule_text(host_language: str) -> str:
    if host_language == "en":
        return "English only for host prose."
    return "中文为主，host 文本只使用中文表达；英文仅可用于固定代号、标签或必要术语。"


def host_language_looks_valid(text: str, host_language: str) -> bool:
    ascii_words = len(ASCII_WORD_PATTERN.findall(text))
    cjk_chars = len(CJK_PATTERN.findall(text))
    if host_language == "en":
        return ascii_words >= max(12, cjk_chars // 4)
    return cjk_chars >= max(20, ascii_words * 2)


def parse_host_candidate(candidate_text: str, host_language: str = "zh") -> dict[str, str]:
    match = HOST_SECTION_PATTERN.search(candidate_text.strip())
    if not match:
        raise ValueError("Host candidate is missing one or more required sections.")
    agent_input, canonical_answer, rationale, next_intent = [part.strip() for part in match.groups()]
    canonical_label = normalize_answer(canonical_answer)
    if canonical_label not in {"SAFE", "DANGEROUS"}:
        raise ValueError("Host candidate must include a canonical answer of SAFE or DANGEROUS.")
    combined_text = "\n".join([agent_input, rationale, next_intent])
    if not host_language_looks_valid(combined_text, host_language):
        expected = "English" if host_language == "en" else "Chinese"
        raise ValueError(f"Host candidate does not appear to follow the required {expected}-only language rule.")
    return {
        "agent_input": agent_input,
        "canonical_answer": canonical_label,
        "scoring_rationale": rationale,
        "next_round_intent": next_intent,
    }


def parse_agent_json(raw_text: str) -> dict[str, Any]:
    stripped = raw_text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = JSON_BLOCK_PATTERN.search(stripped)
        if not match:
            raise
        return json.loads(match.group(0))


def stabilize_memory_rewrite(previous_memory: str, candidate_memory: str) -> str:
    previous = previous_memory.strip()
    candidate = candidate_memory.strip()
    if not previous:
        return candidate
    if not candidate:
        return previous

    candidate_lines = [line.strip() for line in candidate.splitlines() if line.strip()]
    looks_like_single_note = len(candidate_lines) == 1
    large_regression = len(previous) >= 80 and len(candidate) < max(80, int(len(previous) * 0.6))

    if not (looks_like_single_note and large_regression):
        return candidate
    if candidate in previous:
        return previous

    appended = candidate
    if not appended.startswith(("-", "*", "## ", "# ")):
        appended = f"- {appended}"

    if "## Latest Update" in previous:
        return f"{previous}\n{appended}"
    return f"{previous}\n\n## Latest Update\n{appended}"


def memory_mode_prompt_fields(memory_mode: str) -> dict[str, str]:
    if memory_mode == "rewrite":
        return {
            "MEMORY_MODE_NAME": "rewrite",
            "MEMORY_MODE_SPEC": (
                "You rewrite the next full notebook as `updated_memory`. Preserve still-useful prior rules unless they "
                "are contradicted or clearly replaced by a denser summary. Keep `updated_memory` within "
                "`{{MEMORY_BUDGET}}` characters."
            ),
            "MEMORY_FORMAT_HINT": "`updated_memory` should usually be multi-line Markdown notes rather than a single sentence.",
            "MEMORY_QUALITY_HINT": (
                "Treat `updated_memory` like a cleaned-up notebook, not a stack of alerts. "
                "Unify repeated ideas instead of repeating near-duplicate sections."
            ),
            "MEMORY_OUTPUT_EXAMPLE": (
                '{\n'
                '  "response": "SAFE or DANGEROUS, optionally followed by a short explanation",\n'
                '  "recommended_action": "One concrete next action to take right now",\n'
                '  "updated_memory": "A revised Markdown notebook for future rounds"\n'
                '}'
            ),
        }
    return {
        "MEMORY_MODE_NAME": "prepend",
        "MEMORY_MODE_SPEC": (
            "You are not rewriting the full notebook. You only write this round's `new_memory_block`. "
            "The system will prepend your `new_memory_block` before the old external memory, then truncate from "
            "the tail if the total length exceeds `{{MEMORY_BUDGET}}` characters. If an older rule is still important, "
            "restate it briefly in this round's `new_memory_block`, or it may eventually disappear from the tail. "
            "Keep `new_memory_block` concise and high-value. It should be a compact refresh of what most needs to stay alive."
        ),
        "MEMORY_FORMAT_HINT": "`new_memory_block` should usually be multi-line Markdown notes rather than a single sentence.",
        "MEMORY_QUALITY_HINT": (
            "In prepend mode, write a true delta note. Prefer 1-2 changed facts, 1 revised rule, and at most 1-2 "
            "still-essential carry-forward rules. Do not restate the full standing protocol every round. "
            "Avoid repeating the same alert headline with slightly different wording."
        ),
        "MEMORY_OUTPUT_EXAMPLE": (
            '{\n'
            '  "response": "SAFE or DANGEROUS, optionally followed by a short explanation",\n'
            '  "recommended_action": "One concrete next action to take right now",\n'
            '  "new_memory_block": "A compact Markdown block to prepend before old memory"\n'
            '}'
        ),
    }


def load_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, _, value = stripped.partition(":")
        data[key.strip()] = parse_scalar(value.strip())
    return data


def dump_simple_yaml(path: Path, data: dict[str, Any]) -> None:
    lines = [f"{key}: {format_scalar(value)}" for key, value in data.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def next_run_id() -> str:
    existing = []
    for child in RUNS_DIR.iterdir():
        if child.is_dir() and re.fullmatch(r"run_\d{3}", child.name):
            existing.append(int(child.name.split("_")[1]))
    next_index = max(existing, default=0) + 1
    return f"run_{next_index:03d}"


def ensure_run_layout(run_dir: Path) -> dict[str, Path]:
    memory_dir = run_dir / "memory"
    host_dir = run_dir / "host"
    memory_dir.mkdir(parents=True, exist_ok=True)
    host_dir.mkdir(parents=True, exist_ok=True)
    return {
        "memory_dir": memory_dir,
        "host_dir": host_dir,
        "config_path": run_dir / "config.yaml",
        "transcript_path": run_dir / "transcript.md",
        "metrics_path": run_dir / "metrics.json",
    }


def round_host_path(host_dir: Path, round_id: int) -> Path:
    bucket_index = (round_id - 1) // 50
    start = bucket_index * 50 + 1
    end = start + 49
    return host_dir / f"{start:04d}_{end:04d}" / f"u_{round_id:04d}.md"


def sync_metrics_with_config(run_dir: Path, metrics: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return metrics, False

    changed = False
    file_config = load_simple_yaml(config_path)
    if metrics.get("config") != file_config:
        metrics["config"] = file_config
        changed = True

    totals = metrics.get("totals", {})
    completed = int(totals.get("completed_rounds", 0))
    planned = int(file_config.get("rounds", 0) or 0)
    if planned and completed < planned and metrics.get("status") == "completed":
        metrics["status"] = "paused"
        metrics["ended_at"] = None
        changed = True

    return metrics, changed


def initialize_new_run(run_dir: Path, config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    dump_simple_yaml(paths["config_path"], config)
    write_memory(round_memory_path(paths["memory_dir"], 0), "")
    paths["transcript_path"].write_text(
        (
            f"# Transcript for {run_dir.name}\n\n"
            f"- created_at: {utc_now()}\n"
            f"- tested_model: {config['tested_model']}\n"
            f"- host_model: {config['host_model']}\n"
            f"- host_language: {config.get('host_language', 'zh')}\n"
            f"- rounds: {config['rounds']}\n"
            f"- memory_budget: {config['memory_budget']}\n"
            f"- memory_mode: {config.get('memory_mode', 'prepend')}\n\n"
        ),
        encoding="utf-8",
    )
    metrics = {
        "run_id": run_dir.name,
        "status": "in_progress",
        "config": config,
        "rounds": [],
        "started_at": utc_now(),
        "ended_at": None,
        "totals": {
            "completed_rounds": 0,
            "correct_rounds": 0,
            "failed_rounds": 0,
            "invalid_agent_json_rounds": 0,
        },
    }
    write_json(paths["metrics_path"], metrics)
    refresh_runs_log()
    return metrics


def load_or_create_run(args: argparse.Namespace) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    run_id = args.run_id or next_run_id()
    run_dir = RUNS_DIR / run_id
    paths = ensure_run_layout(run_dir)

    if paths["config_path"].exists():
        metrics = json.loads(paths["metrics_path"].read_text(encoding="utf-8"))
        metrics, changed = sync_metrics_with_config(run_dir, metrics)
        if changed:
            write_json(paths["metrics_path"], metrics)
            refresh_runs_log()
        return run_dir, paths, metrics

    config = {
        "world_path": args.world_path,
        "rounds": args.rounds,
        "memory_budget": args.memory_budget,
        "memory_mode": args.memory_mode,
        "tested_model": args.tested_model,
        "host_model": args.host_model,
        "host_language": args.host_language,
        "host_temperature": args.host_temperature,
        "tested_temperature": args.tested_temperature,
        "host_max_tokens": args.host_max_tokens,
        "tested_max_tokens": args.tested_max_tokens,
        "stub_llm": args.stub_llm,
    }
    metrics = initialize_new_run(run_dir, config, paths)
    return run_dir, paths, metrics


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compute_accuracy(metrics: dict[str, Any]) -> tuple[int, int, float]:
    totals = metrics.get("totals", {})
    completed = int(totals.get("completed_rounds", 0))
    correct = int(totals.get("correct_rounds", 0))
    accuracy = (correct / completed) if completed else 0.0
    return correct, completed, accuracy


def summarize_memory_progress(metrics: dict[str, Any]) -> str:
    rounds = metrics.get("rounds", [])
    if not rounds:
        return "还没有写出任何 memory。"
    first = rounds[0]["clipped_memory_length"]
    last = rounds[-1]["clipped_memory_length"]
    return f"已完成 {len(rounds)} 轮，memory 长度从 {first} 变到 {last} 字符"


def detect_breakthroughs(metrics: dict[str, Any]) -> list[str]:
    rounds = metrics.get("rounds", [])
    totals = metrics.get("totals", {})
    completed = int(totals.get("completed_rounds", 0))
    invalid = int(totals.get("invalid_agent_json_rounds", 0))
    correct, _, accuracy = compute_accuracy(metrics)
    breakthroughs: list[str] = []

    if completed >= 1:
        breakthroughs.append("至少完整跑通了 1 轮，host、transcript、metrics 和 memory 产物都已落盘。")
    if completed >= 3:
        breakthroughs.append("已经不是一次性连通性测试，而是形成了稳定的多轮连续运行。")
    if completed >= 10:
        breakthroughs.append("已经进入较长程运行区间，开始适合观察 memory 的演化趋势。")
    if completed and invalid == 0:
        breakthroughs.append("所有已完成轮次里的 tested agent 输出都保持为可解析 JSON。")
    if completed and accuracy == 1.0:
        breakthroughs.append(f"在已完成的 {correct}/{completed} 轮中保持了 100% 正确率。")
    elif completed >= 5 and accuracy >= 0.8:
        breakthroughs.append(f"在已完成的 {correct}/{completed} 轮里达到了较强的阶段性正确率。")
    if rounds and rounds[0]["clipped_memory_length"] != rounds[-1]["clipped_memory_length"]:
        breakthroughs.append("memory 长度在轮次之间发生了变化，说明外部记忆正在主动演化。")
    if metrics.get("status") == "completed":
        breakthroughs.append("已经跑完了这次计划中的全部轮数。")
    if metrics.get("status") == "stopped_on_error":
        breakthroughs.append("已经暴露出一个明确故障模式，值得在下一次 run 前优先排查。")

    return breakthroughs or ["目前还没有明显突破，这次 run 还处在很早期或尚未完成。"]


def build_runs_log() -> str:
    run_metrics: list[dict[str, Any]] = []
    for child in sorted(RUNS_DIR.iterdir()):
        if not child.is_dir():
            continue
        metrics_path = child / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        run_metrics.append(metrics)

    run_metrics.sort(key=lambda item: (item.get("started_at") or "", item.get("run_id") or ""), reverse=True)

    lines = [
        "# 运行日志",
        "",
        f"- 更新时间: {utc_now()}",
        f"- 已记录运行数: {len(run_metrics)}",
        "",
        "这个文件记录每次 run 跑了什么、跑了多少轮，以及有哪些值得记下来的进展。",
        "排列顺序按时间从新到旧。",
        "",
    ]

    if not run_metrics:
        lines.append("暂无 run 记录。")
        return "\n".join(lines) + "\n"

    for metrics in run_metrics:
        config = metrics.get("config", {})
        run_id = metrics.get("run_id", "unknown_run")
        status = metrics.get("status", "unknown")
        correct, completed, accuracy = compute_accuracy(metrics)
        world_path = config.get("world_path", "unknown_world")
        breakthroughs = detect_breakthroughs(metrics)
        started_at = metrics.get("started_at") or "未知"
        ended_at = metrics.get("ended_at") or "尚未结束"

        lines.extend(
            [
                f"## {run_id}",
                "",
                f"- 开始时间: `{started_at}`",
                f"- 结束时间: `{ended_at}`",
                f"- 状态: `{status}`",
                f"- 世界: `{world_path}`",
                f"- tested model: `{config.get('tested_model', 'unknown')}`",
                f"- host model: `{config.get('host_model', 'unknown')}`",
                f"- host 语言: `{config.get('host_language', 'zh')}`",
                f"- 计划轮数: `{config.get('rounds', 'unknown')}`",
                f"- memory 模式: `{config.get('memory_mode', 'prepend')}`",
                f"- 已完成轮数: `{completed}`",
                f"- 正确率: `{correct}/{completed}` ({accuracy:.0%})" if completed else "- 正确率: `0/0`（暂无）",
                f"- memory 进展: `{summarize_memory_progress(metrics)}`",
                f"- transcript: [`{run_id}/transcript.md`](./{run_id}/transcript.md)",
                f"- metrics: [`{run_id}/metrics.json`](./{run_id}/metrics.json)",
                "",
                "### 关键进展",
                "",
            ]
        )
        for item in breakthroughs:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def refresh_runs_log() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_LOG_PATH.write_text(build_runs_log(), encoding="utf-8")


def summarize_history(round_records: list[dict[str, Any]], limit: int = 5) -> str:
    if not round_records:
        return "No prior rounds."
    lines = []
    for record in round_records[-limit:]:
        lines.append(
            (
                f"Round {record['round']}: expected={record['canonical_answer']}, "
                f"agent={record.get('agent_label') or 'INVALID'}, "
                f"correct={record['is_correct']}, "
                f"note={record['next_round_intent']}"
            )
        )
    return "\n".join(lines)


def review_host_candidate(
    candidate_text: str,
    *,
    round_id: int,
    host_path: Path,
    auto_accept: bool,
) -> str:
    current = candidate_text
    while True:
        print(f"\n===== Host Candidate: Round {round_id} =====\n")
        print(current)
        print("\n===== End Candidate =====\n")

        if auto_accept:
            return current

        choice = input("Host action [accept/edit/regenerate/quit]: ").strip().lower()
        if choice in {"accept", "a", ""}:
            return current
        if choice in {"edit", "e"}:
            current = edit_markdown(current)
            continue
        if choice in {"regenerate", "r"}:
            raise RegenerateHostCandidate()
        if choice in {"quit", "q"}:
            raise UserQuit()
        print("Please choose accept, edit, regenerate, or quit.")


def edit_markdown(initial_text: str) -> str:
    editor = os.getenv("EDITOR")
    if editor:
        with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False, encoding="utf-8") as tmp:
            tmp.write(initial_text)
            tmp.flush()
            temp_path = Path(tmp.name)
        try:
            subprocess.run([editor, str(temp_path)], check=True)
            return temp_path.read_text(encoding="utf-8")
        finally:
            temp_path.unlink(missing_ok=True)

    print("No $EDITOR found. Finish editing with a single line containing only END.")
    print("Current content will be replaced.")
    lines: list[str] = []
    while True:
        line = input()
        if line == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


class UserQuit(Exception):
    pass


class RegenerateHostCandidate(Exception):
    pass


def append_transcript(transcript_path: Path, round_record: dict[str, Any]) -> None:
    action_block = ""
    if round_record.get("recommended_action"):
        action_block = f"### Recommended Action\n\n{round_record['recommended_action']}\n\n"
    memory_block = ""
    if round_record.get("new_memory_block"):
        memory_block = f"### New Memory Block\n\n{round_record['new_memory_block']}\n\n"
    if round_record.get("updated_memory"):
        memory_block = f"### Updated Memory\n\n{round_record['updated_memory']}\n\n"
    section = (
        f"## Round {round_record['round']}\n\n"
        f"### Host Input\n\n{round_record['agent_input']}\n\n"
        f"### Canonical Answer\n\n`{round_record['canonical_answer']}`\n\n"
        f"### Agent Response\n\n{round_record['agent_response_raw']}\n\n"
        f"{action_block}"
        f"{memory_block}"
        f"### Scoring\n\n"
        f"- normalized_answer: `{round_record.get('agent_label') or 'INVALID'}`\n"
        f"- is_correct: `{round_record['is_correct']}`\n"
        f"- retry_count: `{round_record['retry_count']}`\n"
        f"- raw_memory_length: `{round_record['raw_memory_length']}`\n"
        f"- clipped_memory_length: `{round_record['clipped_memory_length']}`\n\n"
        f"### Host Notes\n\n{round_record['scoring_rationale']}\n\n"
        f"### Next Round Intent\n\n{round_record['next_round_intent']}\n\n"
    )
    with transcript_path.open("a", encoding="utf-8") as handle:
        handle.write(section)


def build_tested_messages(
    system_prompt: str,
    current_input: str,
    external_memory: str,
    memory_budget: int,
    memory_mode: str,
) -> list[dict[str, str]]:
    mode_fields = memory_mode_prompt_fields(memory_mode)
    rendered_system = render_template(
        system_prompt,
        {
            "CURRENT_INPUT": current_input.strip(),
            "EXTERNAL_MEMORY": external_memory.strip() or "(empty)",
            "MEMORY_BUDGET": str(memory_budget),
            **mode_fields,
        },
    )
    messages = [
        {"role": "system", "content": rendered_system},
        {
            "role": "user",
            "content": (
                "Solve the current round using only the external memory shown in the system prompt. "
                + (
                    "Rewrite the notebook for the next round. "
                    if memory_mode == "rewrite"
                    else "Write one compact new memory block that will be prepended before the old memory. "
                )
                + 
                "Return JSON only."
            ),
        },
    ]
    roles = [message["role"] for message in messages]
    assert roles == ["system", "user"], f"Tested agent messages must be stateless. Got roles={roles}"
    return messages


def call_tested_agent(
    client: OpenAICompatClient,
    *,
    model: str,
    system_prompt: str,
    current_input: str,
    external_memory: str,
    memory_budget: int,
    memory_mode: str,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], str, int, list[str]]:
    messages = build_tested_messages(system_prompt, current_input, external_memory, memory_budget, memory_mode)
    message_roles = [message["role"] for message in messages]
    retry_count = 0
    last_raw = ""
    repair_max_tokens = max(max_tokens, 800)
    memory_key = "updated_memory" if memory_mode == "rewrite" else "new_memory_block"

    while retry_count <= 1:
        if retry_count == 0:
            response = client.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            repair_messages = [
                {"role": "system", "content": messages[0]["content"]},
                {
                    "role": "user",
                    "content": (
                        f"The previous reply was not valid JSON and may have been truncated.\n\nPrevious reply:\n{last_raw}\n\n"
                        "Retry with a much shorter, more compressed memory output.\n"
                        f'Keep "{memory_key}" comfortably under the {memory_budget}-character total budget.\n'
                        + (
                            "Rewrite the notebook compactly and preserve only the highest-value rules.\n"
                            if memory_mode == "rewrite"
                            else (
                                "Write a delta note, not a replacement bulletin. "
                                "Only include changed facts, revised rules, and at most 1-2 still-essential carry-forward rules.\n"
                            )
                        )
                        +
                        'Prefer 2-3 short Markdown sections with short bullets, not a long case list.\n'
                        f'Return minified JSON only with keys "response", "recommended_action", and "{memory_key}".'
                    ),
                },
            ]
            response = client.chat_completion(
                model=model,
                messages=repair_messages,
                temperature=temperature,
                max_tokens=repair_max_tokens,
            )
            message_roles = [message["role"] for message in repair_messages]

        last_raw = response.content
        try:
            parsed = parse_agent_json(last_raw)
        except json.JSONDecodeError:
            retry_count += 1
            if retry_count > 1:
                break
            continue

        if (
            not isinstance(parsed, dict)
            or "response" not in parsed
            or "recommended_action" not in parsed
            or memory_key not in parsed
        ):
            retry_count += 1
            if retry_count > 1:
                break
            last_raw = json.dumps(parsed, ensure_ascii=False)
            continue
        return parsed, last_raw, retry_count, message_roles

    raise ValueError(last_raw)


def generate_host_candidate(
    client: OpenAICompatClient,
    *,
    model: str,
    host_prompt: str,
    world_text: str,
    round_id: int,
    history_summary: str,
    previous_answer: str,
    current_memory: str,
    temperature: float,
    max_tokens: int,
    host_language: str,
) -> str:
    rendered = render_template(
        host_prompt,
        {
            "WORLD": world_text.strip(),
            "HOST_LANGUAGE_RULE": language_rule_text(host_language),
            "ROUND_ID": str(round_id),
            "HISTORY_SUMMARY": history_summary.strip(),
            "PREVIOUS_ANSWER": previous_answer.strip() or "None",
            "CURRENT_MEMORY": current_memory.strip() or "(empty)",
        },
    )
    result = client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": rendered}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return result.content


def run() -> int:
    args = parse_args()
    run_dir, paths, metrics = load_or_create_run(args)
    config = metrics["config"]
    memory_mode = effective_memory_mode(config)

    world_text = load_text(resolve_world_path(str(config["world_path"])))
    host_prompt = load_text(HOST_PROMPT_PATH)
    tested_prompt = load_text(TESTED_PROMPT_PATH)

    client = OpenAICompatClient(stub=args.stub_llm or bool(config.get("stub_llm")))
    completed_rounds = len(metrics["rounds"])

    print(f"Running {run_dir.name} from round {completed_rounds + 1} / {config['rounds']}")

    try:
        for round_id in range(completed_rounds + 1, int(config["rounds"]) + 1):
            previous_record = metrics["rounds"][-1] if metrics["rounds"] else None
            previous_answer = previous_record["agent_response_raw"] if previous_record else "None"
            current_memory = read_memory(resolve_round_memory_path(paths["memory_dir"], round_id - 1))
            history_summary = summarize_history(metrics["rounds"])

            while True:
                try:
                    candidate_text = generate_host_candidate(
                        client,
                        model=str(config["host_model"]),
                        host_prompt=host_prompt,
                        world_text=world_text,
                        round_id=round_id,
                        history_summary=history_summary,
                        previous_answer=previous_answer,
                        current_memory=current_memory,
                        temperature=float(config["host_temperature"]),
                        max_tokens=int(config["host_max_tokens"]),
                        host_language=str(config.get("host_language", "zh")),
                    )
                    approved_text = review_host_candidate(
                        candidate_text,
                        round_id=round_id,
                        host_path=round_host_path(paths["host_dir"], round_id),
                        auto_accept=args.auto_accept_host,
                    )
                    host_data = parse_host_candidate(approved_text, str(config.get("host_language", "zh")))
                    host_file = round_host_path(paths["host_dir"], round_id)
                    host_file.parent.mkdir(parents=True, exist_ok=True)
                    host_file.write_text(approved_text.rstrip() + "\n", encoding="utf-8")
                    break
                except RegenerateHostCandidate:
                    continue
                except ValueError as exc:
                    print(f"Host candidate validation failed: {exc}")
                    if args.auto_accept_host:
                        return 1

            try:
                agent_output, raw_agent_text, retry_count, message_roles = call_tested_agent(
                    client,
                    model=str(config["tested_model"]),
                    system_prompt=tested_prompt,
                    current_input=host_data["agent_input"],
                    external_memory=current_memory,
                    memory_budget=int(config["memory_budget"]),
                    memory_mode=memory_mode,
                    temperature=float(config["tested_temperature"]),
                    max_tokens=int(config["tested_max_tokens"]),
                )
                invalid_json = False
                failure_reason = ""
            except ValueError as exc:
                raw_agent_text = str(exc)
                agent_output = {
                    "response": raw_agent_text,
                    "recommended_action": "",
                    ("updated_memory" if memory_mode == "rewrite" else "new_memory_block"): "",
                }
                retry_count = 1
                message_roles = ["system", "user"]
                invalid_json = True
                failure_reason = "Tested agent failed to return valid JSON twice."

            normalized = normalize_answer(str(agent_output["response"]))
            is_correct = normalized == host_data["canonical_answer"]
            new_memory_block = ""
            updated_memory = ""
            if memory_mode == "rewrite":
                updated_memory = str(agent_output.get("updated_memory", "")).strip()
                next_memory = stabilize_memory_rewrite(current_memory, updated_memory)
            else:
                new_memory_block = str(agent_output.get("new_memory_block", "")).strip()
                next_memory = merge_memory_block(current_memory, new_memory_block)
            clipped_memory, raw_length, clipped_length = clip_memory(next_memory, int(config["memory_budget"]))
            memory_path = round_memory_path(paths["memory_dir"], round_id)
            write_memory(memory_path, clipped_memory)

            round_record = {
                "round": round_id,
                "timestamp": utc_now(),
                "host_file": str(host_file.relative_to(ROOT)),
                "memory_file": str(memory_path.relative_to(ROOT)),
                "agent_input": host_data["agent_input"],
                "canonical_answer": host_data["canonical_answer"],
                "scoring_rationale": host_data["scoring_rationale"],
                "next_round_intent": host_data["next_round_intent"],
                "agent_response_raw": str(agent_output["response"]),
                "recommended_action": str(agent_output.get("recommended_action", "")).strip(),
                "new_memory_block": new_memory_block,
                "updated_memory": updated_memory,
                "memory_mode": memory_mode,
                "agent_label": normalized,
                "is_correct": bool(is_correct),
                "retry_count": retry_count,
                "invalid_json": invalid_json,
                "failure_reason": failure_reason,
                "raw_memory_length": raw_length,
                "clipped_memory_length": clipped_length,
                "tested_agent_message_count": 2,
                "tested_agent_message_roles": message_roles,
            }
            metrics["rounds"].append(round_record)
            metrics["totals"]["completed_rounds"] = len(metrics["rounds"])
            metrics["totals"]["correct_rounds"] = sum(1 for record in metrics["rounds"] if record["is_correct"])
            metrics["totals"]["failed_rounds"] = sum(
                1 for record in metrics["rounds"] if record["failure_reason"]
            )
            metrics["totals"]["invalid_agent_json_rounds"] = sum(
                1 for record in metrics["rounds"] if record["invalid_json"]
            )
            write_json(paths["metrics_path"], metrics)
            refresh_runs_log()
            append_transcript(paths["transcript_path"], round_record)

            print(
                f"Round {round_id}: agent={normalized or 'INVALID'} "
                f"expected={host_data['canonical_answer']} correct={is_correct} "
                f"memory={raw_length}->{clipped_length}"
            )

            if failure_reason:
                metrics["status"] = "stopped_on_error"
                metrics["ended_at"] = utc_now()
                write_json(paths["metrics_path"], metrics)
                refresh_runs_log()
                print(f"Stopping run because of error: {failure_reason}")
                return 1

    except UserQuit:
        metrics["status"] = "paused"
        write_json(paths["metrics_path"], metrics)
        refresh_runs_log()
        print("Run paused by user.")
        return 0
    except LLMClientError as exc:
        metrics["status"] = "stopped_on_error"
        metrics["ended_at"] = utc_now()
        write_json(paths["metrics_path"], metrics)
        refresh_runs_log()
        print(f"LLM error: {exc}", file=sys.stderr)
        return 1

    metrics["status"] = "completed"
    metrics["ended_at"] = utc_now()
    write_json(paths["metrics_path"], metrics)
    refresh_runs_log()
    print(f"Completed {run_dir.name}. Transcript: {paths['transcript_path'].relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
