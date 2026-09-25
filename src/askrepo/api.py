"""HTTP API: POST /ask {"question": "..."} -> {"answer": "...", "sources": [...]}."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from askrepo.agent import run_agent
from askrepo.config import get_settings
from askrepo.llm import OpenAICompatibleClient
from askrepo.retrieval import CodeIndex
from askrepo.tools import ToolBox


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    iterations: int


def build_app(repo_root: Path) -> FastAPI:
    settings = get_settings()
    index_path = repo_root / settings.index_path

    app = FastAPI(title="askrepo", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        if not index_path.exists():
            raise HTTPException(status_code=409, detail="repository is not indexed yet")
        if not settings.llm_api_key:
            raise HTTPException(status_code=500, detail="ASKREPO_LLM_API_KEY is not configured")

        code_index = CodeIndex.load(index_path)
        toolbox = ToolBox(root=repo_root, index=code_index)
        llm = OpenAICompatibleClient(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        result = run_agent(request.question, llm=llm, toolbox=toolbox)
        return AskResponse(answer=result.answer, sources=result.sources, iterations=result.iterations)

    return app
