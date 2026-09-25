from pathlib import Path

from askrepo.agent import run_agent
from askrepo.chunking import chunk_repository
from askrepo.llm import ChatResponse, ToolCall
from askrepo.retrieval import CodeIndex
from askrepo.tools import ToolBox


class ScriptedLLMClient:
    """Replays a fixed sequence of responses -- no network calls, fully deterministic."""

    def __init__(self, responses: list[ChatResponse]):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, messages, tools) -> ChatResponse:
        self.calls += 1
        return self._responses.pop(0)


def _toolbox(tmp_path: Path) -> ToolBox:
    (tmp_path / "app.py").write_text("def add(a, b):\n    return a + b\n")
    index = CodeIndex(chunk_repository(tmp_path))
    return ToolBox(root=tmp_path, index=index)


def test_agent_calls_a_tool_then_answers_with_sources(tmp_path: Path) -> None:
    llm = ScriptedLLMClient(
        [
            ChatResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="search_code", arguments={"query": "add"})],
            ),
            ChatResponse(content="`add` sums two numbers (app.py:1-2).", tool_calls=[]),
        ]
    )

    result = run_agent("What does add() do?", llm=llm, toolbox=_toolbox(tmp_path))

    assert result.iterations == 2
    assert "sums two numbers" in result.answer
    assert result.sources == ["app.py:1-2"]


def test_agent_answers_immediately_if_no_tool_needed(tmp_path: Path) -> None:
    llm = ScriptedLLMClient([ChatResponse(content="Hello!", tool_calls=[])])

    result = run_agent("hi", llm=llm, toolbox=_toolbox(tmp_path))

    assert result.answer == "Hello!"
    assert result.sources == []
    assert result.iterations == 1


def test_agent_stops_after_max_iterations(tmp_path: Path) -> None:
    looping_call = ChatResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="search_code", arguments={"query": "add"})],
    )
    llm = ScriptedLLMClient([looping_call] * 3)

    result = run_agent("What does add() do?", llm=llm, toolbox=_toolbox(tmp_path), max_iterations=3)

    assert result.iterations == 3
    assert "wasn't able to reach a confident answer" in result.answer
    assert llm.calls == 3


def test_agent_reports_tool_errors_instead_of_crashing(tmp_path: Path) -> None:
    llm = ScriptedLLMClient(
        [
            ChatResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_1", name="read_file", arguments={"path": "missing.py"})
                ],
            ),
            ChatResponse(content="That file doesn't exist in this repo.", tool_calls=[]),
        ]
    )

    result = run_agent("What's in missing.py?", llm=llm, toolbox=_toolbox(tmp_path))

    assert "doesn't exist" in result.answer
    assert result.sources == []


class ExplodingToolBox:
    """A tool that raises a plain (non-ToolError) exception -- e.g. a bug the
    tool's author didn't anticipate. The agent loop must survive this."""

    def schemas(self):
        return []

    def call(self, name, arguments):
        raise RuntimeError("boom")


def test_agent_survives_an_unexpected_tool_exception() -> None:
    llm = ScriptedLLMClient(
        [
            ChatResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="search_code", arguments={"query": "x"})],
            ),
            ChatResponse(content="Something went wrong, but I recovered.", tool_calls=[]),
        ]
    )

    result = run_agent("anything", llm=llm, toolbox=ExplodingToolBox())

    assert result.answer == "Something went wrong, but I recovered."
