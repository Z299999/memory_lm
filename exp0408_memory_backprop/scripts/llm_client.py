from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class LLMClientError(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str
    usage: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None


def load_repo_dotenv(dotenv_path: Path | None = None) -> None:
    path = dotenv_path or Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


class OpenAICompatClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        stub: bool = False,
    ) -> None:
        load_repo_dotenv()
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("DASHSCOPE_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.stub = stub

        if not self.stub and not self.api_key:
            raise LLMClientError("Missing API key. Set DASHSCOPE_API_KEY in .env.")

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 600,
    ) -> ChatResult:
        if self.stub:
            return self._stub_response(messages)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(f"HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise LLMClientError(f"Network error: {exc}") from exc

        try:
            message = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(f"Unexpected response: {raw}") from exc

        if isinstance(message, list):
            parts = [item.get("text", "") for item in message if isinstance(item, dict) and item.get("type") == "text"]
            message = "\n".join(parts)

        return ChatResult(content=str(message), usage=raw.get("usage"), raw_response=raw)

    def _stub_response(self, messages: list[dict[str, str]]) -> ChatResult:
        joined = " ".join(m["content"] for m in messages)
        if "EXTRACTED_ANSWER" in joined or "canonical" in joined.lower():
            return ChatResult(content="""## EXTRACTED_ANSWER
42

## IS_CORRECT
false

## FEEDBACK_RATIONALE
Student made arithmetic error in step 2.

## NEXT_ACTION
feedback

## AGENT_INPUT
你在第二步计算时出现了错误。题目说Kyle找到了Mimi的两倍，即48个，而不是42个。请重新从第二步开始计算。
""")
        return ChatResult(content="The answer is 42. Step 1: ... Step 2: ... Final answer: **42**.")
