"""Split a repository's text files into overlapping, line-numbered chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "dist",
    "build",
    ".askrepo",
}

MAX_FILE_SIZE_BYTES = 1_000_000  # skip anything bigger; almost never source code


@dataclass(frozen=True)
class Chunk:
    """A contiguous, line-numbered slice of a single file."""

    path: str
    start_line: int
    end_line: int
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


def _is_probably_text(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def iter_source_files(root: Path, excluded_dirs: set[str] = DEFAULT_EXCLUDED_DIRS):
    """Yield every text file under `root`, skipping excluded directories and large/binary files."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in excluded_dirs for part in path.parts):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0 or size > MAX_FILE_SIZE_BYTES:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if not _is_probably_text(data):
            continue
        yield path


def chunk_file(path: Path, root: Path, max_lines: int = 60, overlap: int = 10) -> list[Chunk]:
    """Split one file into overlapping chunks of at most `max_lines` lines."""
    if max_lines <= overlap:
        raise ValueError("max_lines must be greater than overlap")

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        return []

    rel_path = path.relative_to(root).as_posix()
    chunks: list[Chunk] = []
    start = 0
    step = max_lines - overlap
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        body = "\n".join(lines[start:end])
        chunks.append(Chunk(path=rel_path, start_line=start + 1, end_line=end, text=body))
        if end == len(lines):
            break
        start += step
    return chunks


def chunk_repository(root: Path, max_lines: int = 60, overlap: int = 10) -> list[Chunk]:
    """Chunk every source file under `root` into a flat list of `Chunk`s."""
    root = root.resolve()
    chunks: list[Chunk] = []
    for file_path in iter_source_files(root):
        chunks.extend(chunk_file(file_path, root, max_lines=max_lines, overlap=overlap))
    return chunks
