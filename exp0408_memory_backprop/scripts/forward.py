from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from host import HostResult, call_host
from llm_client import OpenAICompatClient
from student_model import load_student

STUDENT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "student.md"


# ── helpers ──────────────────────────────────────────────────────────────────

def _render_student_prompt(input_text: str, memory: str) -> str:
    template = STUDENT_PROMPT_PATH.read_text(encoding="utf-8")
    if memory:
        template = template.replace("{{#if memory}}", "").replace("{{/if}}", "")
    else:
        template = re.sub(r"\{\{#if memory\}\}.*?\{\{/if\}\}\n?", "", template, flags=re.DOTALL)
    template = template.replace("{{memory}}", memory)
    template = template.replace("{{input}}", input_text)
    return template


def _call_student(*, student, input_text: str, memory: str, temperature: float, max_new_tokens: int) -> str:
    prompt = _render_student_prompt(input_text, memory)
    return student.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)


def _extract_canonical_answer(assistant_content: str) -> str:
    numbers = re.findall(r"\*?\*?(\d[\d,\.]*)\*?\*?", assistant_content)
    return numbers[-1].replace(",", "") if numbers else assistant_content.strip()


# ── run directory helpers ─────────────────────────────────────────────────────

BATCH_SIZE = 50  # files per subfolder in memory/ and host/


def _init_run_dir(runs_dir: Path, run_id: str) -> Path:
    run_dir = runs_dir / run_id
    (run_dir / "training").mkdir(parents=True, exist_ok=True)
    return run_dir


def _batch_dir(parent: Path, step: int) -> Path:
    """Return (and create) the batch subfolder for a given step number."""
    lo = ((step - 1) // BATCH_SIZE) * BATCH_SIZE + 1
    hi = lo + BATCH_SIZE - 1
    folder = parent / f"{lo:04d}-{hi:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _append_transcript(run_dir: Path, text: str) -> None:
    with (run_dir / "transcript.md").open("a", encoding="utf-8") as f:
        f.write(text)


def _append_training_sample(run_dir: Path, m_prev: str, m_next: str) -> None:
    with (run_dir / "training" / "samples.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"m_prev": m_prev, "m_next": m_next}, ensure_ascii=False) + "\n")


# ── main forward loop ─────────────────────────────────────────────────────────

def run_forward(
    *,
    data_path: Path,
    runs_dir: Path,
    run_id: str,
    host_model: str,
    student_model: str,
    host_base_url: str | None = None,
    host_api_key: str | None = None,
    host_temperature: float,
    student_temperature: float,
    host_max_tokens: int,
    student_max_new_tokens: int,
    max_feedback_rounds: int,
    forward_per_back: float,
    num_problems: int | None,
    stub: bool = False,
) -> None:
    host_client = OpenAICompatClient(base_url=host_base_url, api_key=host_api_key, stub=stub)
    student = load_student(student_model, stub=stub)
    run_dir = _init_run_dir(runs_dir, run_id)

    problems = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            msgs = item["messages"]
            problems.append({
                "question": msgs[0]["content"],
                "answer": _extract_canonical_answer(msgs[1]["content"]),
            })
    if num_problems is not None:
        problems = problems[:num_problems]

    print(f"[forward] run={run_id}  problems={len(problems)}  forward_per_back={forward_per_back}")

    memory: str = ""
    global_step = 0
    problems_since_back = 0

    (run_dir / "transcript.md").write_text(
        f"# Run {run_id}\nStarted: {datetime.now().isoformat()}\n\n", encoding="utf-8"
    )

    for prob_idx, prob in enumerate(problems):
        question = prob["question"]
        canonical = prob["answer"]

        print(f"\n[problem {prob_idx + 1}/{len(problems)}] {question[:60]}...")

        # ── Round 0: student answers the problem ──────────────────────────────
        student_response = _call_student(
            student=student,
            input_text=question,
            memory=memory,
            temperature=student_temperature,
            max_new_tokens=student_max_new_tokens,
        )

        global_step += 1
        step_id = f"{global_step:04d}"
        prob_step_id = step_id

        _append_training_sample(run_dir, memory, student_response)
        _write_text(_batch_dir(run_dir / "memory", global_step) / f"m_{step_id}.md", student_response)
        _append_transcript(run_dir, (
            f"## Step {global_step} — Problem {prob_idx + 1}\n"
            f"**Input:** {question}\n\n"
            f"**Memory in:** {memory[:100] + '...' if len(memory) > 100 else memory or '(empty)'}\n\n"
            f"**Student response:**\n{student_response}\n\n"
        ))

        memory = student_response

        # ── Feedback rounds ───────────────────────────────────────────────────
        for fb_round in range(1, max_feedback_rounds + 1):
            host_result: HostResult = call_host(
                client=host_client,
                model=host_model,
                problem=question,
                canonical_answer=canonical,
                student_response=memory,
                feedback_round=fb_round,
                max_feedback_rounds=max_feedback_rounds,
                temperature=host_temperature,
                max_tokens=host_max_tokens,
            )

            _write_text(_batch_dir(run_dir / "host", int(prob_step_id)) / f"h_{prob_step_id}_fb{fb_round}.md", (
                f"## EXTRACTED_ANSWER\n{host_result.extracted_answer}\n\n"
                f"## IS_CORRECT\n{host_result.is_correct}\n\n"
                f"## FEEDBACK_RATIONALE\n{host_result.feedback_rationale}\n\n"
                f"## NEXT_ACTION\n{host_result.next_action}\n\n"
                f"## AGENT_INPUT\n{host_result.agent_input}\n"
            ))

            correct_marker = "✓" if host_result.is_correct else "✗"
            print(f"  [fb {fb_round}] {correct_marker} extracted={host_result.extracted_answer}  next={host_result.next_action}")

            _append_transcript(run_dir, (
                f"### Feedback {fb_round}\n"
                f"**Extracted:** {host_result.extracted_answer} | **Correct:** {host_result.is_correct}\n\n"
                f"**Host → Student:** {host_result.agent_input}\n\n"
            ))

            if host_result.next_action == "next_problem":
                break

            # Student responds to feedback
            student_response = _call_student(
                student=student,
                input_text=host_result.agent_input,
                memory=memory,
                temperature=student_temperature,
                max_new_tokens=student_max_new_tokens,
            )

            global_step += 1
            step_id = f"{global_step:04d}"

            _append_training_sample(run_dir, memory, student_response)
            _write_text(_batch_dir(run_dir / "memory", global_step) / f"m_{step_id}.md", student_response)
            _append_transcript(run_dir, f"**Student response (fb {fb_round}):**\n{student_response}\n\n")

            memory = student_response

        # ── Backward pass trigger ─────────────────────────────────────────────
        problems_since_back += 1
        if not math.isinf(forward_per_back) and problems_since_back >= forward_per_back:
            print(f"[backward] triggering after {problems_since_back} problems...")
            _trigger_backward(run_dir, student)
            problems_since_back = 0

    _append_transcript(run_dir, f"\n---\nFinished: {datetime.now().isoformat()}\n")
    print(f"\n[forward] done. transcript → {run_dir / 'transcript.md'}")


def _trigger_backward(run_dir: Path, student) -> None:
    samples_path = run_dir / "training" / "samples.jsonl"
    if not samples_path.exists():
        print("[backward] no samples yet, skipping.")
        return
    from backward import run_backward
    run_backward(samples_path=samples_path, run_dir=run_dir, student=student)
