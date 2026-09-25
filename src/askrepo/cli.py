"""Command-line entry point: `askrepo index`, `askrepo ask`, `askrepo serve`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from askrepo.agent import run_agent
from askrepo.chunking import chunk_repository
from askrepo.config import get_settings
from askrepo.llm import OpenAICompatibleClient
from askrepo.retrieval import CodeIndex
from askrepo.tools import ToolBox

app = typer.Typer(help="Ask natural-language questions about a codebase.")
console = Console()


@app.command()
def index(
    path: Path = typer.Argument(Path("."), help="Repository root to index."),
) -> None:
    """Build (or rebuild) the local search index for a repository."""
    settings = get_settings()
    root = path.resolve()
    chunks = chunk_repository(
        root, max_lines=settings.max_chunk_lines, overlap=settings.chunk_overlap
    )
    code_index = CodeIndex(chunks)
    index_path = root / settings.index_path
    code_index.save(index_path)
    console.print(f"Indexed [bold]{len(chunks)}[/bold] chunks from {root} -> {index_path}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question about the codebase."),
    path: Path = typer.Option(Path("."), "--repo", help="Repository root."),
) -> None:
    """Ask a question about a previously indexed repository."""
    settings = get_settings()
    root = path.resolve()
    index_path = root / settings.index_path
    if not index_path.exists():
        console.print(f"[red]No index found at {index_path}.[/red] Run `askrepo index` first.")
        raise typer.Exit(code=1)
    if not settings.llm_api_key:
        console.print("[red]ASKREPO_LLM_API_KEY is not set.[/red] See .env.example.")
        raise typer.Exit(code=1)

    code_index = CodeIndex.load(index_path)
    toolbox = ToolBox(root=root, index=code_index)
    llm = OpenAICompatibleClient(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )

    result = run_agent(question, llm=llm, toolbox=toolbox)
    console.print(Markdown(result.answer))
    if result.sources:
        console.print("\n[dim]Sources:[/dim] " + ", ".join(result.sources))


@app.command()
def serve(
    path: Path = typer.Option(Path("."), "--repo", help="Repository root."),
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run the HTTP API (see askrepo.api:build_app)."""
    import uvicorn

    from askrepo.api import build_app

    uvicorn.run(build_app(path.resolve()), host=host, port=port)


if __name__ == "__main__":
    app()
