# askrepo

[![CI](https://github.com/h-bahmani/askrepo/actions/workflows/ci.yml/badge.svg)](https://github.com/h-bahmani/askrepo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Ask natural-language questions about a codebase from the command line or an HTTP API. The agent doesn't get the whole repo dumped into its context — it has to call tools (`search_code`, `read_file`, `list_files`) to go find the answer, the same way a person would, and it cites the file/line ranges it actually looked at.

Real output, asking askrepo about its own repository:

```
$ askrepo ask "what is about"
The repository implements askrepo, a tool that lets you ask natural-language questions about a
codebase from the command line or via an HTTP API.

Instead of loading the whole repository into an LLM's context, askrepo builds a searchable index
of the source files (chunked with overlapping line numbers and indexed with BM25). When a
question is asked, an agent loop calls the tools search_code, read_file, and list_files to
retrieve relevant chunks, then composes an answer together with citations to the exact file-line
ranges (e.g., src/auth.py:14-29) [README.md:1-34].

Key components (as described in the README) include:

  Module        Role
  chunking.py   Walks the repo, skips noise (e.g., VCS files, virtual-env binaries), and splits
                each file into overlapping line-range chunks.
  retrieval.py  Indexes those chunks with the BM25 algorithm (rank-bm25).
  agent.py      Runs a bounded tool-calling loop where the LLM decides which tool to invoke,
                receives the results, and eventually produces an answer with sources.
  llm.py        Handles communication with any OpenAI-compatible provider (OpenAI, Groq,
                OpenRouter, local server).

The project provides a CLI (askrepo ask ...) and a server mode (askrepo serve ...) so you can
query a repository locally or via HTTP [README.md:35-57]. Configuration is done through
environment variables prefixed with ASKREPO_ (e.g., API key, model name, index path)
[README.md:69-84].
```

## Why

Dropping an entire repository into an LLM's context window is expensive, doesn't scale past a few thousand lines, and gives no way to tell *where* an answer came from. This is the smaller, cheaper alternative: index the repo once, then let the model search it on demand and show its sources.

## How it works

```
repo files -> chunk (overlapping, line-numbered) -> BM25 index
                                                         |
question -> agent loop <-- search_code / read_file / list_files
                |
             answer + sources
```

- **Chunking** (`chunking.py`) walks the repo, skips VCS/venv/build noise and binaries, and splits each file into overlapping line ranges so an answer never gets cut off mid-context.
- **Retrieval** (`retrieval.py`) indexes chunks with BM25 (`rank-bm25`) — no embedding model or GPU required, and it's a strong baseline for identifier-heavy text like code.
- **Agent** (`agent.py`) runs a bounded tool-calling loop: the model decides which tool to call, gets the result appended to the conversation, and repeats until it either answers or hits the iteration cap.
- **LLM client** (`llm.py`) talks to anything OpenAI-compatible — OpenAI, Groq, OpenRouter, a local server — by pointing `base_url` at it.

## Quickstart

```bash
pip install -e .
cp .env.example .env   # set ASKREPO_LLM_API_KEY (and ASKREPO_LLM_BASE_URL if not using OpenAI)

askrepo index /path/to/some/repo
askrepo ask "What does the retry logic in the HTTP client do?" --repo /path/to/some/repo
```

Or run it as a service:

```bash
askrepo serve --repo /path/to/some/repo
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "Where is the database connection configured?"}'
```

## Configuration

All settings are environment variables (prefix `ASKREPO_`), read from `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `ASKREPO_LLM_API_KEY` | — | API key for your chosen provider |
| `ASKREPO_LLM_MODEL` | `gpt-4o-mini` | Model name as your provider expects it |
| `ASKREPO_LLM_BASE_URL` | — | Set to switch providers (Groq, OpenRouter, a local server); leave unset for OpenAI |
| `ASKREPO_INDEX_PATH` | `.askrepo/index.pkl` | Where the local index is stored, relative to the repo root |

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

The test suite doesn't call a real LLM — the agent loop is tested against a scripted fake client, so it's deterministic and runs offline in CI.

## Limitations

- BM25 is lexical, not semantic: it won't find a relevant chunk that uses entirely different wording than the query.
- One repo per index; there's no cross-repo search.
- The index is a local file, not persisted anywhere shared — fine for one developer on one machine, not for a team.
- Conversation size isn't budgeted against the provider's context/rate limit. A free-tier key with a small tokens-per-minute cap (e.g. Groq's on-demand tier) can reject a request once a few tool results accumulate; askrepo surfaces this as a clear error (`LLMError`) instead of a stack trace, but doesn't yet retry with a trimmed history.

## License

MIT — see [LICENSE](LICENSE).
