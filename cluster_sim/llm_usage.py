"""Write normalized planner LLM token usage beside experiment traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from trace import append_event
from typing import Any, cast


@dataclass(frozen=True)
class LlmUsage:
    """Provider-reported token usage for one planner LLM call."""

    tick_number: int
    input_tokens: int
    output_tokens: int
    model: str


def usage_path(trace_file: Path) -> Path:
    """Return the LLM usage JSONL path adjacent to one run trace."""
    return trace_file.with_name(f"{trace_file.stem}.llm-usage.jsonl")


def append_usage(path: Path, usage: LlmUsage) -> None:
    """Append one normalized LLM usage record."""
    append_event(path, asdict(usage))


def from_openai_response(
    tick_number: int,
    response: object,
    requested_model: str,
) -> LlmUsage:
    """Normalize Responses or Chat Completions token usage."""
    usage = getattr(response, "usage", None)
    if usage is None:
        raise ValueError("LLM response has no token usage")
    return LlmUsage(
        tick_number=tick_number,
        input_tokens=_token_count(
            _first_not_none(
                getattr(usage, "input_tokens", None),
                getattr(usage, "prompt_tokens", None),
            ),
            "input_tokens",
        ),
        output_tokens=_token_count(
            _first_not_none(
                getattr(usage, "output_tokens", None),
                getattr(usage, "completion_tokens", None),
            ),
            "output_tokens",
        ),
        model=str(getattr(response, "model", None) or requested_model),
    )


def _first_not_none(first: object, second: object) -> object:
    """Prefer the Responses field while preserving valid zero counts."""
    return second if first is None else first


def _token_count(value: object, name: str) -> int:
    """Require one provider-reported nonnegative token count."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class OpenAIUsageClient:
    """Wrap OpenAI create calls without changing their responses."""

    def __init__(
        self,
        client: object,
        tick_number: int,
        requested_model: str,
        usage: list[LlmUsage],
    ) -> None:
        """Keep the SDK client and shared usage sink."""
        self._client = client
        sdk = cast(Any, client)
        self.chat = _ChatProxy(
            sdk.chat,
            tick_number,
            requested_model,
            usage,
        )
        if hasattr(client, "responses"):
            self.responses = _CreateProxy(
                sdk.responses,
                tick_number,
                requested_model,
                usage,
            )

    def __getattr__(self, name: str) -> object:
        """Delegate SDK surfaces not involved in generation."""
        return getattr(self._client, name)


class _ChatProxy:
    def __init__(
        self,
        chat: object,
        tick_number: int,
        requested_model: str,
        usage: list[LlmUsage],
    ) -> None:
        self._chat = chat
        sdk_chat = cast(Any, chat)
        self.completions = _CreateProxy(
            sdk_chat.completions,
            tick_number,
            requested_model,
            usage,
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self._chat, name)


class _CreateProxy:
    def __init__(
        self,
        resource: object,
        tick_number: int,
        requested_model: str,
        usage: list[LlmUsage],
    ) -> None:
        self._resource = resource
        self._tick_number = tick_number
        self._requested_model = requested_model
        self._usage = usage

    def create(self, *args: object, **kwargs: object) -> Any:
        response = self._resource.create(*args, **kwargs)
        self._usage.append(
            from_openai_response(
                self._tick_number,
                response,
                self._requested_model,
            )
        )
        return response

    def __getattr__(self, name: str) -> object:
        return getattr(self._resource, name)
