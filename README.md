# Dynamic Topology RAG

A proof of concept for **topology-based agentic RAG**. Instead of retrieving a flat top-k list of chunks, the pipeline builds a lightweight **Chunk Map** from clean Markdown — chunk→chunk, chunk→section, chunk→page, and chunk→paragraph links — and lets an LLM agent navigate that map with deterministic tools.

This implements the design notes in [`docs/plan.md`](docs/plan.md) and the specification in [`docs/topology_based_agentic_rag_specification.md`](docs/topology_based_agentic_rag_specification.md).

## What is implemented

- Markdown → sections, ToC, paragraphs, and word-bounded chunks.
- Topology links: previous/next chunk, parent section, page range, paragraph IDs.
- Optional page boundaries via `<!-- page: N -->` markers or form feeds.
- BM25 lexical seeding and ranked search.
- `grep` with literal or regex matching.
- Optional semantic search through LiteLLM embeddings; falls back to BM25 when no embedding model is configured.
- Agent tool loop with evidence scratchpad, hop budget, and a forced final synthesis.
- Local (Ollama) and cloud (OpenAI/Anthropic/etc.) models through LiteLLM.
- CLI inspection commands that do not require an API key or an LLM.
- Full-chunk inspection, BM25-vs-vector retrieval comparison, and complete audit-trace export.
- Unit tests for the indexer, store, tools, agent loop, and CLI inspection paths.

## Project layout

```
.
├── data/sample.md          # small demo corpus
├── docs/                   # handwritten notes, spec, implementation plan
├── src/
│   ├── indexer.py          # Markdown → Chunk Map
│   ├── models.py           # Pydantic data models
│   ├── store.py            # in-memory store, BM25, grep, vector search
│   ├── tools.py            # agent-facing tools + JSON schemas
│   └── agent.py            # tool-calling traversal loop
├── tests/
├── main.py                 # CLI entry point
├── pyproject.toml
└── requirements.txt
```

## Quick start

> Full walkthrough: [`docs/usage.md`](docs/usage.md) covers setup, model choices, every inspection command, the Chunk Map JSON, and troubleshooting.

```bash
uv sync
cp .env.example .env        # then fill in a key, or use a local model
```

Run against the sample document:

```bash
uv run python main.py -q "How does the agent recover context when a chunk ends mid-thought?"
```

Use a local Ollama model:

```bash
ollama pull qwen2.5:7b
uv run python main.py -m ollama/qwen2.5:7b -q "What is the Chunk Map?"
```

Use OpenAI or Anthropic:

```bash
export OPENAI_API_KEY="..."
uv run python main.py -m gpt-4o-mini -q "What problem does topology traversal solve?"

export ANTHROPIC_API_KEY="..."
uv run python main.py -m anthropic/claude-3-5-sonnet-20241022 -q "Summarize the open problems."
```

Any model string supported by [LiteLLM](https://docs.litellm.ai/) works. Tool calling support is required.

## Inspect without an LLM

```bash
# Table of contents
uv run python main.py --toc -f data/sample.md

# Headings and sections
uv run python main.py --headings -f data/sample.md
uv run python main.py --sections -f data/sample.md

# BM25 / grep
uv run python main.py --search "page boundary" -f data/sample.md
uv run python main.py --grep "get_chunk_para" -f docs/topology_based_agentic_rag_specification.md
uv run python main.py --grep "chunk_\\d+" --regex -f docs/plan.md

# Read a full chunk by ID
uv run python main.py --chunk sample:chunk_0004 -f data/sample.md

# Compare BM25 with vector retrieval, including full chunk text
uv run python main.py --retrieval "page boundary" --top-k 3 -f data/sample.md

# Show the exact tool schemas given to the model
uv run python main.py --list-tools
```

`--retrieval` labels the vector section with its actual mode. Without
`TOPO_RAG_EMBEDDING_MODEL`, it reports `bm25_fallback` and repeats the BM25
results; that is expected.

Index multiple documents in one store:

```bash
uv run python main.py -f docs/plan.md -f docs/topology_based_agentic_rag_specification.md \
  -q "How do the plan and the spec differ on paragraph tools?"
```

## Agent tools

| Tool | Purpose |
| --- | --- |
| `list_documents` | List loaded docs and their sizes |
| `get_toc` | Hierarchical table of contents |
| `get_all_headings` | Flat heading list |
| `get_all_sect` | Section inventory with page/chunk counts |
| `get_chunk` | Read a chunk by ID |
| `get_nearest_chunks` | Walk previous/next chunks |
| `get_chunk_section` | Peek at or read the parent section |
| `get_chunk_page` | Recover all chunks on the same page |
| `get_chunk_para` | Return the original paragraph blocks |
| `grep` | Literal or regex keyword search |
| `bm25_search` | Ranked lexical search |
| `vector_search` | Semantic search (BM25 fallback) |
| `record_evidence` | Save a distilled finding with a chunk citation |

## Page markers

Markdown has no native pages. To exercise the `chunk→page` edge, insert a marker before the content of a new page:

```markdown
<!-- page: 2 -->
```

A form-feed character (`\f`) also starts a new page. Plain Markdown is treated as page 1.

## How a query runs

1. The CLI loads Markdown and builds a `DocumentMap`.
2. BM25 retrieves the top-k seed chunks for the question.
3. The agent reads the seeds, expands through topology tools when needed, and calls `record_evidence` for important findings.
4. When it stops calling tools — or when the hop budget is exhausted — it writes a final answer with chunk-ID citations.

To audit a run:

```bash
# Print the full text of every evidence chunk used in the answer
uv run python main.py -q "What is the Chunk Map?" --show-evidence

# Save BM25 seeds, evidence chunks, and every full tool output
uv run python main.py -q "What is the Chunk Map?" --save-trace traces/chunk-map.json
```

`-v` shows terse trace previews. `--save-trace` writes the complete audit record,
including raw tool responses, which is the reliable way to verify what the model
actually saw.

## Tests

```bash
uv run pytest
```

The tests do not call an LLM or an embedding API.

## Current limitations

- **Markdown only.** PDF/HTML ingestion and OCR are intentionally out of scope; a converter should produce clean Markdown first.
- **ATX headings only** (`#` … `######`); setext headings (`===`, `---`) are not parsed.
- **Fixed word-count chunking.** Chunks never cross section or page boundaries, but chunk sizing is not semantic.
- **In-memory store.** No persistence, incremental indexing, or concurrent writes.
- **Page mapping is opt-in.** Without page markers, every chunk is on page 1.
- **Semantic search is optional.** Set `TOPO_RAG_EMBEDDING_MODEL` to enable it; otherwise `vector_search` is BM25 in disguise and says so in its response.
- **No reranker yet.** The spec's BM25 + reranker stage is reduced to BM25 for the PoC.

## Suggested next steps

1. Add a reranker stage after BM25 seeding.
2. Add an evaluation harness with the four conditions from the notes: 5-chunk, query-only, single-pass plan, and topology traversal.
3. Add sub-agents for branch-level reading and context offloading.
4. Add a Markdown/PageIndex-style ingestion adapter for PDFs.
5. Expose the tools through MCP or a persistent graph service.
