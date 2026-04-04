from __future__ import annotations

from pathlib import Path


def round_memory_path(memory_dir: Path, round_id: int) -> Path:
    return memory_dir / f"round_{round_id:03d}.md"


def read_memory(memory_path: Path) -> str:
    if not memory_path.exists():
        return ""
    return memory_path.read_text(encoding="utf-8").strip()


def clip_memory(memory_text: str, budget_chars: int) -> tuple[str, int, int]:
    raw_length = len(memory_text)
    if raw_length <= budget_chars:
        return memory_text, raw_length, raw_length
    clipped = memory_text[:budget_chars].rstrip()
    return clipped, raw_length, len(clipped)


def write_memory(memory_path: Path, memory_text: str) -> None:
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    final_text = memory_text.rstrip() + "\n" if memory_text.strip() else ""
    memory_path.write_text(final_text, encoding="utf-8")

