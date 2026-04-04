from __future__ import annotations

from pathlib import Path

COMPRESSION_NOTICE_HEADER = "## Compression Notice"


def round_memory_path(memory_dir: Path, round_id: int) -> Path:
    return memory_dir / f"m_{round_id:04d}.md"


def legacy_round_memory_path(memory_dir: Path, round_id: int) -> Path:
    return memory_dir / f"round_{round_id:03d}.md"


def resolve_round_memory_path(memory_dir: Path, round_id: int) -> Path:
    current = round_memory_path(memory_dir, round_id)
    if current.exists():
        return current
    legacy = legacy_round_memory_path(memory_dir, round_id)
    if legacy.exists():
        return legacy
    return current


def read_memory(memory_path: Path) -> str:
    if not memory_path.exists():
        return ""
    return memory_path.read_text(encoding="utf-8").strip()


def overflow_notice(budget_chars: int) -> str:
    return (
        f"\n\n{COMPRESSION_NOTICE_HEADER}\n"
        f"- 上一轮记忆超出 {budget_chars} 字符上限并已被截断。"
        "下一轮请更主动压缩、合并相近规则，只保留最关键条件、反转点与例外。"
    )


def strip_prior_overflow_notice(memory_text: str) -> str:
    marker_index = memory_text.find(COMPRESSION_NOTICE_HEADER)
    if marker_index == -1:
        return memory_text
    return memory_text[:marker_index].rstrip()


def clip_memory(memory_text: str, budget_chars: int) -> tuple[str, int, int]:
    raw_length = len(memory_text)
    if raw_length <= budget_chars:
        return memory_text, raw_length, raw_length

    base_text = strip_prior_overflow_notice(memory_text).rstrip()
    notice = overflow_notice(budget_chars)

    if budget_chars <= len(notice):
        clipped = notice[:budget_chars].rstrip()
        return clipped, raw_length, len(clipped)

    prefix_budget = budget_chars - len(notice)
    clipped_prefix = base_text[:prefix_budget].rstrip()
    clipped = f"{clipped_prefix}{notice}".rstrip() if clipped_prefix else notice[:budget_chars].rstrip()
    return clipped, raw_length, len(clipped)


def write_memory(memory_path: Path, memory_text: str) -> None:
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    final_text = memory_text.rstrip() + "\n" if memory_text.strip() else ""
    memory_path.write_text(final_text, encoding="utf-8")
