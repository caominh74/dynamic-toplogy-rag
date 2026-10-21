# Running and inspecting the data

This guide covers the local workflow: setting up the environment, running the
agent, and inspecting the Chunk Map with the CLI or from Python. The inspection
commands do **not** call an LLM, so they work without API keys.

## 1. Setup

From the repository root:

```bash
uv sync
cp .env.example .env
```

Then edit `.env` or export credentials in your shell. For a local model, no API
key is needed.

```bash
# OpenAI
export OPENAI_API_KEY="..."

# Anthropic
export ANTHROPIC_API_KEY="..."

# Ollama (must be running locally)
ollama serve
ollama pull qwen2.5:7b
```

`TOPO_RAG_MODEL` in `.env` sets the default model. A command-line `-m` overrides
it.

### OpenAI-compatible endpoints

For a server that exposes the OpenAI API (vLLM, LM Studio, SGLang, a proxy, etc.),
use the `openai/` model prefix and point `TOPO_RAG_API_BASE` at the API root:

```dotenv
TOPO_RAG_MODEL=openai/your-model-id
TOPO_RAG_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=dummy
```

`openai/` is **LiteLLM routing syntax, not part of the model ID**. LiteLLM strips
it before sending the request; for example, `openai/deepseek-flash` sends
`"model": "deepseek-flash"` to the configured base URL. You need the prefix
because LiteLLM supports many providers and must know which API client/protocol
to use. DeepSeek's API is OpenAI-compatible, so it uses the `openai/` client.

If you prefer LiteLLM's native DeepSeek provider instead, use
`deepseek/deepseek-flash` and set `DEEPSEEK_API_KEY`; in that case you do not need
the `openai/` prefix.

- The base URL usually needs `/v1`, because LiteLLM's OpenAI client appends the
  resource path.
- `OPENAI_API_KEY` can be any non-empty string for local servers that do not
  check it.
- Omit `TOPO_RAG_API_BASE` for real OpenAI; LiteLLM uses its default endpoint.
- Override per run with `--api-base`:

```bash
uv run python main.py \
  -m openai/your-model-id \
  --api-base http://localhost:8000/v1 \
  -q "What is the Chunk Map?"
```

For native Ollama, use an `ollama/` model and the Ollama endpoint without `/v1`:

```dotenv
TOPO_RAG_MODEL=ollama/qwen2.5:7b
TOPO_RAG_API_BASE=http://localhost:11434
```

## 2. Run a query

```bash
uv run python main.py \
  -f data/sample.md \
  -q "How does the agent recover context when a chunk ends mid-thought?"
```

Useful options:

| Option | Meaning |
| --- | --- |
| `-f PATH` | Markdown file to index; repeat for multiple documents |
| `-m MODEL` | LiteLLM model string |
| `--top-k N` | Number of BM25 seed chunks (default 5) |
| `--max-hops N` | Maximum tool-calling rounds (default 8) |
| `--chunk-size N` | Chunk size in words (default 400) |
| `--temperature X` | Sampling temperature (default 0) |
| `--show-evidence` | Print the full text of every evidence chunk |
| `--save-trace PATH` | Write a full JSON audit trace after the run |
| `-v` | Print the tool trace and evidence to stderr |
| `--json` | Print the full result as JSON |

Examples:

```bash
# Local model
uv run python main.py -m ollama/qwen2.5:7b -q "What is the Chunk Map?"

# Cloud model
uv run python main.py -m gpt-4o-mini -q "Summarize the open problems."

# Inspect the agent's path through the document
uv run python main.py -v -q "How are pages linked to chunks?"

# Machine-readable result
uv run python main.py --json -q "What are the four engineering problems?" \
  | jq '{answer, hops, evidence, trace: [.trace[] | {hop, tool}]}'
```

A query runs in four stages:

1. The Markdown is indexed into sections, paragraphs, and chunks.
2. BM25 retrieves the top-k seed chunks.
3. The agent reads those seeds and expands through the topology tools.
4. The agent records evidence and writes a cited answer.

## 3. Inspect the document structure

These commands are the fastest way to understand what the agent sees.

### Table of contents

```bash
uv run python main.py --toc -f data/sample.md
```

```text
sample — Topology RAG Field Notes
- [sample:sec_0000] Topology RAG Field Notes (L1, 1 chunks)
  - [sample:sec_0001] Why flat chunking fails (L2, 1 chunks)
  - [sample:sec_0002] The Chunk Map (L2, 1 chunks)
    - [sample:sec_0003] Metadata and identifiers (L3, 1 chunks)
  ...
```

Each entry shows the section ID, heading level, and number of chunks in that
section. Heading-only sections can legitimately have zero chunks.

Use `--json` to get the tree with page ranges and child nodes:

```bash
uv run python main.py --toc -f data/sample.md --json \
  | jq '.[0].toc[0]'
```

### Headings and sections

```bash
uv run python main.py --headings -f data/sample.md
uv run python main.py --sections -f data/sample.md
```

`--headings` is a flat list. `--sections` includes page ranges, parent IDs,
paragraph counts, and chunk counts. Both print one JSON object per document; add
`--json` to get a single JSON array when multiple files are loaded.

```bash
uv run python main.py --sections -f data/sample.md --json \
  | jq '.[0].sections[] | {title, page_start, page_end, chunk_count}'
```

### Lexical search (BM25)

```bash
uv run python main.py --search "page boundary" -f data/sample.md
```

```json
{
  "ok": true,
  "query": "page boundary",
  "results": [
    {
      "chunk_id": "sample:chunk_0004",
      "section_title": "Agent traversal",
      "page_start": 2,
      "score": 2.0289,
      "preview": "The agent starts with BM25 seed chunks. ..."
    }
  ]
}
```

Scores are relative; use them to rank candidates, not as calibrated relevance
probabilities. If a query returns nothing useful, try different wording or use
`--grep`.

### Read a specific chunk

```bash
uv run python main.py --chunk sample:chunk_0004 -f data/sample.md
```

```text
[sample:chunk_0004] Agent traversal (page 2, 118 words)
The agent starts with BM25 seed chunks. ...
```

`--chunk` prints the complete chunk text, section title, page range, and word
count. Repeat it to compare several chunks:

```bash
uv run python main.py \
  --chunk sample:chunk_0002 \
  --chunk sample:chunk_0004 \
  -f data/sample.md
```

Add `--json` to get the full chunk records, including paragraph IDs and topology
links. An unknown ID prints an error and exits with status 1.

### Compare BM25 and vector retrieval

```bash
uv run python main.py --retrieval "page boundary" --top-k 3 -f data/sample.md
```

This runs the query through both retrieval paths and prints the full text of each
result, not just a preview. The vector section is labelled with its actual mode:

```text
=== Vector (bm25_fallback) ===
warning: No embedding model configured; fell back to BM25. ...
```

That label matters. Without `TOPO_RAG_EMBEDDING_MODEL`, vector search is BM25 in
disguise, so identical rankings do not mean the two methods agree. To compare
real semantic retrieval, configure an embedding model first (see section 8).

Use `--json` for a machine-readable report:

```bash
uv run python main.py --retrieval "page boundary" -f data/sample.md --json \
  | jq '{query, bm25: [.bm25[] | {rank, chunk_id, score}], vector: {mode: .vector.mode, results: [.vector.results[] | {rank, chunk_id, score}]}}'
```

### Exact and regex search

```bash
# Literal, case-insensitive by default
uv run python main.py --grep "get_chunk_para" \
  -f docs/topology_based_agentic_rag_specification.md

# Regular expression
uv run python main.py --grep "chunk_[0-9]+" --regex -f docs/plan.md

# Case-sensitive
uv run python main.py --grep "BM25" --case-sensitive -f data/sample.md
```

`grep` returns chunk IDs and snippets. Use a chunk ID with `--chunk` above or the
Python inspector below, or ask the agent about it.

### Tool schemas

```bash
uv run python main.py --list-tools
```

This prints the exact function schemas passed to the model. It is useful when
debugging why an agent does or does not call a tool.

## 4. Inspect chunks from Python

`--chunk` covers routine chunk inspection. For custom analysis across the whole
document, the `DocumentMap` is a normal Pydantic object. Run this from the
repository root:

```bash
uv run python - <<'PY'
from pathlib import Path
from src.indexer import build_document_map

doc = build_document_map(
    Path("data/sample.md").read_text(encoding="utf-8"),
    doc_id="sample",
    source="data/sample.md",
    chunk_size=400,
)

print("title:", doc.title)
print("sections:", len(doc.sections), "paragraphs:", len(doc.paragraphs), "chunks:", len(doc.chunks))

for chunk_id in doc.chunk_order:
    chunk = doc.chunks[chunk_id]
    print(
        f"{chunk.chunk_id} | {chunk.section_title} | "
        f"pages {chunk.page_start}-{chunk.page_end} | "
        f"words {chunk.word_count} | prev={chunk.prev_chunk_id} | next={chunk.next_chunk_id}"
    )
PY
```

Print one chunk:

```bash
uv run python - <<'PY'
from pathlib import Path
from src.indexer import build_document_map

doc = build_document_map(Path("data/sample.md").read_text(), doc_id="sample")
chunk = doc.chunks["sample:chunk_0000"]
print(chunk.model_dump_json(indent=2))
PY
```

Inspect a paragraph and the links around a chunk:

```bash
uv run python - <<'PY'
from pathlib import Path
from src.indexer import build_document_map

doc = build_document_map(Path("data/sample.md").read_text(), doc_id="sample")
chunk = doc.chunks["sample:chunk_0000"]

print("paragraphs:")
for paragraph_id in chunk.paragraph_ids:
    paragraph = doc.paragraphs[paragraph_id]
    print(paragraph.paragraph_id, "page", paragraph.page)
    print(paragraph.text[:500])

print("\nnearest chunks:", chunk.prev_chunk_id, "<-", chunk.chunk_id, "->", chunk.next_chunk_id)
PY
```

## 5. Understand the IDs

IDs are deterministic and scoped to a document:

| Kind | Format | Example |
| --- | --- | --- |
| Document | slug of filename | `sample` |
| Section | `doc:sec_NNNN` | `sample:sec_0002` |
| Chunk | `doc:chunk_NNNN` | `sample:chunk_0004` |
| Paragraph | `doc:p_NNNN` | `sample:p_0011` |

The filename is lowercased and non-alphanumeric characters become hyphens. If
two files have the same stem, the CLI appends `-2`, `-3`, etc. Chunk IDs are
unique across the loaded store.

## 6. Prepare your own Markdown

The indexer is intentionally simple. It works best with clean Markdown produced
by a converter:

- Use one level-1 heading (`#`) for the document title.
- Use nested `##`, `###`, … headings for structure.
- Put a blank line between paragraphs.
- Fenced code blocks (``` or ~~~) are fine; `#` lines inside them are not treated
  as headings.
- Tables are preserved as text in the surrounding paragraph.
- Setext headings (`Title\n=====`) are not supported.
- YAML front matter is not parsed.

To exercise the page edge, insert a standalone marker before the content of each
new page:

```markdown
<!-- page: 2 -->

This text is on page 2.
```

A form-feed character (`\f`) also starts a new page. Without markers, the whole
document is page 1.

Chunks never cross a section boundary or a page boundary. A single paragraph
larger than `--chunk-size` is split, but all pieces keep the same paragraph ID so
`get_chunk_para` can recover the original block.

## 7. Multiple documents

Pass `-f` more than once:

```bash
uv run python main.py \
  -f docs/plan.md \
  -f docs/topology_based_agentic_rag_specification.md \
  -q "How do the plan and the spec differ on paragraph tools?"
```

Document-level tools such as `get_toc` need a `doc_id` when more than one
document is loaded. The agent's system prompt lists the available IDs. Search
tools (`bm25_search`, `grep`, `vector_search`) search across all documents unless
the agent passes a `doc_id`.

## 8. Semantic search (optional)

Set an embedding model to turn on vector search:

```bash
# OpenAI
export TOPO_RAG_EMBEDDING_MODEL=text-embedding-3-small

# Ollama
export TOPO_RAG_EMBEDDING_MODEL=ollama/nomic-embed-text
```

Without this variable, `vector_search` returns `"mode": "bm25_fallback"` and uses
BM25 instead. This is deliberate: the PoC runs without an embedding provider.
Embedding vectors are cached in memory after the first search.

## 9. Inspect the agent result

With `--json`, the result has this shape:

```json
{
  "answer": "... [sample:chunk_0004]",
  "model": "gpt-4o-mini",
  "seed_chunk_ids": ["sample:chunk_0004", "sample:chunk_0002"],
  "evidence": [
    {"chunk_id": "sample:chunk_0004", "note": "The agent walks to adjacent chunks."}
  ],
  "trace": [
    {
      "hop": 1,
      "tool": "get_chunk",
      "arguments": {"chunk_id": "sample:chunk_0004"},
      "result_chars": 1200,
      "result_preview": "{\"ok\": true, ..."
    }
  ],
  "hops": 2
}
```

- `seed_chunk_ids` are the BM25 starting points.
- `evidence` is the agent's distilled working memory.
- `trace` records every tool call, including failures.
- `hops` is the number of LLM rounds, not the number of tool calls.

For a readable trace, use `-v`. Trace and evidence go to stderr, so the answer on
stdout stays clean for piping.

### See the chunks behind the evidence

`-v` lists evidence notes but not their source text. Add `--show-evidence` to
print the complete chunk for every recorded evidence item after the answer:

```bash
uv run python main.py \
  -q "How does the agent recover context when a chunk ends mid-thought?" \
  --show-evidence
```

With `--json`, `--show-evidence` adds an `evidence_chunks` array containing the
full chunk records.

### Save a complete audit trace

```bash
uv run python main.py \
  -q "How does the agent recover context when a chunk ends mid-thought?" \
  --save-trace traces/run.json
```

The trace is a superset of the normal `--json` result. It contains:

| Key | Contents |
| --- | --- |
| `tool_outputs` | Every tool call with its complete raw result, not a preview |
| `seed_chunks` | Full records for the BM25 seed chunks |
| `evidence_chunks` | Full records for the evidence chunks |
| `trace` | The compact per-call trace shown by `-v` |
| `answer`, `evidence`, `hops`, ... | The normal result fields |

`--json` deliberately omits `tool_outputs` so normal output stays readable;
`--save-trace` includes it. Use the saved file when verifying grounding, comparing
retrieval behavior, or debugging why the agent chose a citation.

## 10. Troubleshooting

**`LLM call failed ... Check the model name, credentials, and that the model supports tool calling.`**

Check `TOPO_RAG_MODEL`, the provider API key, and whether the model supports
function/tool calling. Small base models often do not.

**`doc_id is required when multiple documents are loaded`**

Pass `doc_id` to document-level tools, or load one file at a time. The CLI's
inspection commands already iterate over all documents.

**BM25 returns weak or empty results**

BM25 is lexical. Try `--grep` for exact identifiers, or rephrase the query with
terms that appear in the document. Semantic search requires
`TOPO_RAG_EMBEDDING_MODEL`.

**`get_chunk_page` returns only one page's content**

The source has no page markers, so every chunk is on page 1. Add
`<!-- page: N -->` markers before conversion.

**A heading-only section has no chunks**

That is expected. Sections exist in the ToC even when they have no body text.
Use the child sections or `get_chunk_section` on a neighboring chunk.

**Regex grep fails**

The pattern is invalid. Drop `--regex` for a literal search, or escape special
characters such as `(`, `[`, and `.`.

## 11. Run the tests

```bash
uv run pytest
uv run ruff check .
```

The tests mock the LLM and do not require credentials or network access.
