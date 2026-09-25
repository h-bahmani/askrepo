from pathlib import Path

from askrepo.chunking import chunk_file, chunk_repository


def test_chunk_file_splits_with_overlap(tmp_path: Path) -> None:
    file_path = tmp_path / "example.py"
    file_path.write_text("\n".join(f"line {i}" for i in range(1, 21)))

    chunks = chunk_file(file_path, tmp_path, max_lines=10, overlap=3)

    assert [c.start_line for c in chunks] == [1, 8, 15]
    assert chunks[0].end_line == 10
    assert chunks[1].start_line == 8  # overlaps the previous chunk by 3 lines
    assert chunks[-1].end_line == 20
    assert chunks[0].path == "example.py"


def test_chunk_file_empty_file_yields_nothing(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.py"
    file_path.write_text("")

    assert chunk_file(file_path, tmp_path) == []


def test_chunk_repository_skips_excluded_dirs_and_binaries(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n")

    excluded = tmp_path / ".venv" / "lib"
    excluded.mkdir(parents=True)
    (excluded / "module.py").write_text("this should not be indexed\n")

    (tmp_path / "logo.png").write_bytes(b"\x89PNG\x00\x01\x02not-real-but-has-a-null-byte\x00")

    chunks = chunk_repository(tmp_path)

    paths = {c.path for c in chunks}
    assert paths == {"src/main.py"}
