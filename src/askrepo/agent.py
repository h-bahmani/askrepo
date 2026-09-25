"""A small ReAct-style loop: let the model call tools until it has enough context to answer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from askrepo.llm import LLMClient
from askrepo.tools import ToolBox, ToolError

SYSTEM_PROMPT = """\
You are a codebase assistant. You cannot see the repository directly -- only
through the search_code, read_file, and list_files tools. Before answering any
question about the code, call at least one tool to check your assumptions
against the actual source. When you answer, cite the files and line ranges
you relied on, e.g. "(src/app.py:12-40)". If the tools don't turn up an
answer, say so plainly instead of guessing.
"""

MAX_ITERATIONS = 6


@dataclass
class AgentResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    iterations: int = 0


def _extract_sources(tool_name: str, raw_result: str) -> list[str]:
    try:
        payload = json.loads(raw_result)
    except json.JSONDecodeError:
        return []

    if tool_name == "search_code" and isinstance(payload, list):
        return [f"{item['path']}:{item['lines']}" for item in payload]
    if tool_name == "read_file" and isinstance(payload, dict) and "path" in payload:
        return [f"{payload['path']}:{payload['start_line']}-{payload['end_line']}"]
    return []


def run_agent(
    question: str,
    llm: LLMClient,
    toolbox: ToolBox,
    max_iterations: int = MAX_ITERATIONS,
) -> AgentResult:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    sources: list[str] = []

    for iteration in range(1, max_iterations + 1):
        response = llm.complete(messages, tools=toolbox.schemas())

        if not response.tool_calls:
            answer = response.content or "I couldn't produce an answer."
            return AgentResult(answer=answer, sources=_dedupe(sources), iterations=iteration)

        messages.append(
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in response.tool_calls
                ],
            }
        )

        for call in response.tool_calls:
            try:
                result = toolbox.call(call.name, call.arguments)
            except ToolError as exc:
                result = json.dumps({"error": str(exc)})
            else:
                sources.extend(_extract_sources(call.name, result))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return AgentResult(
        answer=(
            f"I wasn't able to reach a confident answer within {max_iterations} tool calls. "
            "Try a narrower question, or run `askrepo ask` again with more specific wording."
        ),
        sources=_dedupe(sources),
        iterations=max_iterations,
    )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
