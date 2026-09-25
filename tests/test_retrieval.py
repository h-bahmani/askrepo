from pathlib import Path

from askrepo.chunking import Chunk
from askrepo.retrieval import CodeIndex, tokenize


def test_tokenize_splits_identifiers_and_lowercases() -> None:
    assert tokenize("def calculate_total(Price, qty=2):") == [
        "def",
        "calculate_total",
        "price",
        "qty",
        "2",
    ]


def test_search_ranks_the_matching_chunk_first() -> None:
    chunks = [
        Chunk(path="a.py", start_line=1, end_line=5, text="def parse_config(path): ..."),
        Chunk(path="b.py", start_line=1, end_line=5, text="def render_template(name): ..."),
        Chunk(path="c.py", start_line=1, end_line=5, text="class HttpClient: ..."),
    ]
    index = CodeIndex(chunks)

    results = index.search("parse_config", top_k=2)

    assert results
    assert results[0].chunk.path == "a.py"


def test_search_on_empty_index_returns_no_results() -> None:
    assert CodeIndex([]).search("anything") == []


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    chunks = [Chunk(path="a.py", start_line=1, end_line=1, text="x = 1")]
    index_path = tmp_path / "index.pkl"

    CodeIndex(chunks).save(index_path)
    loaded = CodeIndex.load(index_path)

    assert loaded.chunks == chunks
    assert loaded.search("x")[0].chunk.path == "a.py"
