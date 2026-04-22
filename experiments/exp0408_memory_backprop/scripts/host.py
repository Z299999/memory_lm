from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import OpenAICompatClient


HOST_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "host.md"

SECTION = re.compile(
    r"## EXTRACTED_ANSWER\s*(.*?)\s*"
    r"## IS_CORRECT\s*(.*?)\s*"
    r"## FEEDBACK_RATIONALE\s*(.*?)\s*"
    r"## NEXT_ACTION\s*(.*?)\s*"
    r"## AGENT_INPUT\s*(.*)",
    re.DOTALL,
)


@dataclass
class HostResult:
    extracted_answer: str
    is_correct: bool
    feedback_rationale: str
    next_action: str          # "feedback" | "next_problem"
    agent_input: str          # message to send to student


def _render(template: str, values: dict[str, str]) -> str:
    for k, v in values.items():
        template = template.replace(f"{{{{{k}}}}}", v)
    return template


def call_host(
    *,
    client: OpenAICompatClient,
    model: str,
    problem: str,
    canonical_answer: str,
    student_response: str,
    feedback_round: int,
    max_feedback_rounds: int,
    temperature: float = 0.3,
    max_tokens: int = 1000,
) -> HostResult:
    template = HOST_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = _render(template, {
        "problem": problem,
        "canonical_answer": canonical_answer,
        "student_response": student_response,
        "feedback_round": str(feedback_round),
        "max_feedback_rounds": str(max_feedback_rounds),
    })

    result = client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return parse_host_response(result.content)


def parse_host_response(text: str) -> HostResult:
    m = SECTION.search(text)
    if not m:
        # Fallback: treat as feedback, pass raw text as agent_input
        return HostResult(
            extracted_answer="unclear",
            is_correct=False,
            feedback_rationale="(parse error)",
            next_action="feedback",
            agent_input=text.strip(),
        )

    extracted_answer = m.group(1).strip()
    is_correct = m.group(2).strip().lower() == "true"
    feedback_rationale = m.group(3).strip()
    next_action = m.group(4).strip().lower()
    agent_input = m.group(5).strip()

    if next_action not in {"feedback", "next_problem"}:
        next_action = "next_problem" if is_correct else "feedback"

    return HostResult(
        extracted_answer=extracted_answer,
        is_correct=is_correct,
        feedback_rationale=feedback_rationale,
        next_action=next_action,
        agent_input=agent_input,
    )
