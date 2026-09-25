import json
from pathlib import Path

import pytest

from askrepo.chunking import chunk_repository
from askrepo.retrieval import CodeIndex
from askrepo.tools import ToolBox, ToolError


@pytest.fixture
def toolbox(tmp_path: Path) -> ToolBox:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def calculate_shipping_cost(weight_kg, distance_km):\n"
        "    return weight_kg * distance_km * 0.42\n"
    )
    (tmp_path / "src" / "auth.py").write_text(
        "def verify_password(raw, hashed):\n    return check_hash(raw, hashed)\n"
    )
    (tmp_path / "README.md").write_text("# Demo\nA small example project.\n")
    (tmp_path / "NOTES.md").write_text("Remember to update the changelog.\n")
    index = CodeIndex(chunk_repository(tmp_path))
    return ToolBox(root=tmp_path, index=index)


def test_search_code_returns_matching_snippet(toolbox: ToolBox) -> None:
    payload = json.loads(toolbox.call("search_code", {"query": "calculate_shipping_cost"}))
    assert payload
    assert payload[0]["path"] == "src/main.py"


def test_read_file_respects_line_range(toolbox: ToolBox) -> None:
    payload = json.loads(
        toolbox.call("read_file", {"path": "src/main.py", "start_line": 2, "end_line": 2})
    )
    assert payload["text"].splitlines() == ["    return weight_kg * distance_km * 0.42"]


def test_read_file_rejects_path_traversal(toolbox: ToolBox) -> None:
    with pytest.raises(ToolError):
        toolbox.call("read_file", {"path": "../outside.py"})


def test_read_file_rejects_missing_file(toolbox: ToolBox) -> None:
    with pytest.raises(ToolError):
        toolbox.call("read_file", {"path": "does/not/exist.py"})


def test_list_files_matches_glob(toolbox: ToolBox) -> None:
    payload = json.loads(toolbox.call("list_files", {"pattern": "*.md"}))
    assert sorted(payload) == ["NOTES.md", "README.md"]


def test_unknown_tool_raises(toolbox: ToolBox) -> None:
    with pytest.raises(ToolError):
        toolbox.call("delete_repo", {})
