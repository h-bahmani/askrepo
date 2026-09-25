"""A minimal, provider-agnostic chat-completion interface.

`OpenAICompatibleClient` talks to anything that speaks the OpenAI chat-completions
wire format (OpenAI itself, Groq, OpenRouter, a local Ollama/vLLM server, ...) by
pointing `base_url` at it. Swapping providers is a config change, not a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatResponse:
    content: str | None
    tool_calls: list[ToolCall]


class LLMClient(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ChatResponse: ...


class OpenAICompatibleClient:
    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ChatResponse:
        import json

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools or None,
        )
        choice = response.choices[0].message
        tool_calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=json.loads(call.function.arguments or "{}"),
            )
            for call in (choice.tool_calls or [])
        ]
        return ChatResponse(content=choice.content, tool_calls=tool_calls)
