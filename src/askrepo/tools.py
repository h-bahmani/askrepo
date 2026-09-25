"""Tool implementations the agent can call, plus their function-calling schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from askrepo.retrieval import CodeIndex

MAX_READ_LINES = 400


class ToolError(Exception):
    """Raised when a tool cannot fulfil a request; the message is shown to the agent."""


class ToolBox:
    """Read-only tools scoped to a single repository root and its search index."""

    def __init__(self, root: Path, index: CodeIndex):
        self.root = root.resolve()
        self.index = index

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_code",
                    "description": (
                        "Keyword-search the repository and return the best-matching "
                        "code/doc snippets with their file path and line range."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search terms."},
                            "top_k": {
                                "type": "integer",
                                "description": "Max results to return (default 5).",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        f"Read up to {MAX_READ_LINES} lines of one file, optionally a "
                        "specific line range, to see more context than a search snippet gives."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Repo-relative file path.",
                            },
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List repository-relative file paths matching a glob pattern.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Glob pattern, e.g. 'src/**/*.py'.",
                            },
                        },
                        "required": [],
                    },
                },
            },
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "search_code":
            return self._search_code(**arguments)
        if name == "read_file":
            return self._read_file(**arguments)
        if name == "list_files":
            return self._list_files(**arguments)
        raise ToolError(f"unknown tool: {name}")

    def _search_code(self, query: str, top_k: int = 5) -> str:
        results = self.index.search(query, top_k=top_k)
        payload = [
            {
                "path": r.chunk.path,
                "lines": f"{r.chunk.start_line}-{r.chunk.end_line}",
                "score": round(r.score, 3),
                "snippet": r.chunk.text,
            }
            for r in results
        ]
        return json.dumps(payload)

    def _resolve_in_repo(self, rel_path: str) -> Path:
        candidate = (self.root / rel_path).resolve()
        if self.root not in candidate.parents and candidate != self.root:
            raise ToolError(f"path escapes the repository: {rel_path}")
        if not candidate.is_file():
            raise ToolError(f"no such file: {rel_path}")
        return candidate

    def _read_file(
        self, path: str, start_line: int | None = None, end_line: int | None = None
    ) -> str:
        full_path = self._resolve_in_repo(path)
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()

        start = max(1, start_line or 1)
        end = min(len(lines), end_line or len(lines))
        if end - start + 1 > MAX_READ_LINES:
            end = start + MAX_READ_LINES - 1
        body = "\n".join(lines[start - 1 : end])
        return json.dumps({"path": path, "start_line": start, "end_line": end, "text": body})

    def _list_files(self, pattern: str = "**/*") -> str:
        matches = [
            p.relative_to(self.root).as_posix()
            for p in sorted(self.root.glob(pattern))
            if p.is_file()
        ]
        return json.dumps(matches[:200])
