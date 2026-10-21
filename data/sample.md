# Topology RAG Field Notes

This document is a small, self-contained corpus for the topology-based RAG proof of concept. It describes how a Chunk Map preserves document structure and how an agent can traverse that structure instead of relying only on flat similarity search.

## Why flat chunking fails

Naive RAG splits a document into fixed-size chunks and retrieves the top five by embedding similarity. This works for isolated facts, but it breaks when an answer depends on context that sits just outside the chunk boundary. A table may be split across two chunks, a pronoun may refer to a paragraph that was never retrieved, and a section's conclusion may depend on an example several pages earlier.

The usual workaround is to add overlap or to retrieve more chunks. Both increase context usage without guaranteeing that the missing structure is restored. A section heading is often a better guide than a neighboring sentence, because it tells the reader where the chunk sits in the document.

## The Chunk Map

The Chunk Map is a lightweight topology built from clean Markdown. Each chunk keeps links to its neighbors, its parent section, its page, and its original paragraph blocks. The document's table of contents is extracted separately, so an agent can start from the global structure and walk down to the evidence.

| Edge | Purpose |
| --- | --- |
| chunk to chunk | Preserve sequential reading order across boundaries |
| chunk to section | Return the full thematic context of a heading |
| chunk to page | Recover material that was split by pagination |
| chunk to paragraph | Work at a finer granularity than the chunk |

### Metadata and identifiers

Every chunk has a stable identifier, a document identifier, a section identifier, and a word count. The indexer also records the first sentence of each chunk, which gives the agent a cheap preview before it decides to read the full text. Paragraph identifiers make it possible to recover the original block even when a long paragraph was split into multiple chunks.

<!-- page: 2 -->

## Agent traversal

The agent starts with BM25 seed chunks. It reads a seed, decides whether the chunk is self-contained, and expands only when necessary. For a chunk that ends mid-thought, `get_nearest_chunks` returns the previous and next chunks. For a chunk that belongs to a larger argument, `get_chunk_section` returns the entire parent section. For a table or figure split by a page boundary, `get_chunk_page` recovers the rest of the page.

The agent records distilled findings in an evidence scratchpad before synthesizing an answer. Each note cites a chunk identifier. This keeps raw text out of the final reasoning prompt and makes the answer auditable. If the evidence is insufficient, the agent is expected to say so rather than invent a fact.

### Human versus machine navigation

A human reader often starts at the table of contents, decomposes the question into parts, and then reads the relevant sections. That strategy is intuitive but does not scale to a large repository where the table of contents is itself too big to inspect. The machine strategy starts from a small number of lexical seeds, traverses the local topology, and only consults the global table of contents when the local search stalls.

<!-- page: 3 -->

## Open engineering problems

Context management is the first problem. Reading whole sections can quickly exhaust the model's context window, so the agent needs a budget and a way to offload raw reading to sub-agents. The second problem is task offloading: a sub-agent can read one branch of the map and return a compressed finding to the orchestrator.

The third problem is post-retrieval manipulation. Results returned by `get_*` tools often need filtering, deduplication, or slicing before they enter the main prompt. The fourth problem is tool delivery. The same topology can be exposed through a skill definition, a Model Context Protocol server, a command-line interface, or a graph API. Each option has different tradeoffs for portability, latency, and observability.

## Evaluation sketch

A useful evaluation compares four settings: five-chunk naive retrieval, query-only answering, a single-pass plan, and agentic topology traversal. The first three are baselines. The fourth is the contribution. Metrics should cover answer correctness, citation accuracy, context tokens consumed, and the number of tool calls required. The central research question is whether structural traversal can compensate for the semantic context that flat chunking loses.
