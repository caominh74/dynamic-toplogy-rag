# Architecture & System Spec: Topology-Based Agentic RAG

## 1. System Overview & Core Philosophy
Traditional "Naive RAG" relies on flat chunking and semantic similarity, often losing document structure and surrounding context. Conversely, full Knowledge Graph RAG is computationally expensive to build and maintain. 

**Core Contribution:** **Topology-Based Agentic Traversal**. 
By using Markdown as the primary document representation (bypassing complex parsing/OCR overhead), we extract the native structural hierarchy of the document (Table of Contents, `#` headings, pages, paragraphs, and sequential chunk order) to build a lightweight **Chunk Map (Document Topology)**. An autonomous agent then uses deterministic structural tools to traverse this map, gather evidence, synthesize answers, and verify completeness.

---

## 2. Research Questions & Objectives (RQs)
* **Primary RQ:** Can we build a high-performing RAG pipeline relying solely on **Document Topology + Agentic Traversal**?
* **RQ 1.1 (Semantics vs. Structure):** Can deep semantic understanding be compensated for by intelligent topology-based traversal?
* **RQ 1.2 (Overcoming Naive RAG):** Can the standard shortcomings of naive chunk-based RAG (broken context, missed cross-references, arbitrary chunk boundaries) be resolved through topological retrieval?

---

## 3. Data Model: The "Chunk Map" (Topology)
Documents are ingested as Markdown and broken into chunks. Instead of isolated text blobs, every chunk is indexed with structural metadata and bidirectional links:

### 3.1. Structural Metadata & Edges
* **`chunk <-> chunk`**: Sequential links (previous/next chunk IDs) for linear reading.
* **`chunk <-> section`**: Hierarchical link mapping the chunk to its parent Markdown header (`#`, `##`, `###`).
* **`chunk <-> page`**: Physical/logical page boundary mapping.
* **`chunk <-> paragraph`**: Fine-grained block mapping (optional/secondary).
* **Temporal Metadata**: `indexed_at` and `last_updated_time` timestamps.
* **Global Structure**: Extracted **Table of Contents (ToC)** representing the global tree of the document.

---

## 4. Agent Toolset (MCP / CLI / `Skills.md` Interface)
The agent interacts with the Chunk Map using three categories of tools:

### 4.1. Entry-Point & Discovery Tools (Seeding)
* `get_toc(doc_id)`: Returns the hierarchical Table of Contents for a document.
* `get_all_headings(doc_id)`: Returns a flat or nested list of all Markdown `#` headings.
* `get_all_sect(doc_id)`: Lists all high-level document sections.
* `grep(pattern, doc_id?)`: Exact keyword/regex search across chunks (combined with BM25 reranking).
* `vector_search(query, top_k=5)`: Semantic search to find initial "seed" chunks in the topology.

### 4.2. Topological Traversal Tools (Navigation)
* `get_nearest_chunks(chunk_id, window_size=2)`: Fetches the immediate preceding and succeeding chunks (`chunk <-> chunk`) to expand local context.
* `get_chunk_section(chunk_id)`: Fetches the entire parent section (all sibling chunks under the same `#` heading) for complete thematic context.
* `get_chunk_page(chunk_id)`: Fetches all chunks belonging to the same page.

---

## 5. Agent Execution Pipeline & Workflow

### Why Not Human-Style Top-Down Reading?
* *Human Workflow:* `Query (Decompose) -> Read ToC -> Pick Parts`. 
* *Limitation:* Fails or becomes bottlenecked on massive documents or multi-document repositories where the ToC alone is too large or ambiguous.

### Machine Workflow (Hybrid Seed-and-Traverse)
The agent executes a 6-stage iterative loop with an initial context budget (e.g., **Top-5 seed chunks**):

1. **Query Analysis & Strategic Planning (`Strat Plan`)**:
   * Decompose the user query.
   * Decide whether to start via global structure (`get_toc` / `get_all_headings`) or bottom-up seeding (`grep` / `vector_search` + BM25 Reranker).
2. **Initial Seeding (`Top-k Chunk`)**:
   * Retrieve the top $k$ (e.g., $k=5$) most relevant chunks as entry points into the Chunk Map.
3. **Topological Traversal (`Traverse Chunk Map`)**:
   * Inspect seed chunks. If a chunk cuts off mid-thought or references surrounding context, call `get_nearest_chunks` or `get_chunk_section` to walk the graph outward.
4. **Evidence Gathering (`Gather Evidence`)**:
   * Extract and store relevant facts, quotes, and section references into working memory.
5. **Synthesis (`Synthesize`)**:
   * Combine gathered evidence from traversed nodes into a coherent draft answer.
6. **Self-Verification (`Recheck`)**:
   * Evaluate if the synthesized answer fully resolves the original query. If gaps remain, trigger another traversal step; otherwise, output the final **Answer**.

---

## 6. Advanced System Design & Open Engineering Challenges
When implementing or scaling this pipeline, the following architectural patterns and open problems must be addressed:

1. **Context Management & Offloading (RLM-inspired)**:
   * *Concept:* Inspired by **Recursive Language Models (RLM)** and document-aware retrieval systems (like **PageIndex**).
   * *Problem:* Traversing whole sections or multiple pages will quickly bloat the primary agent's context window.
   * *Solution:* Implement strict context budgeting (e.g., keeping only 5 active chunks in primary memory) and offload raw reading/summarization tasks.
2. **Task Offloading to Sub-Agents**:
   * Spawn lightweight sub-agents to traverse specific branches of the Chunk Map (e.g., *"Sub-agent A: read Section 3.2 and return only facts about X"*) and return compressed findings to the orchestrator agent.
3. **Post-Retrieval Data Manipulation**:
   * Provide the agent with post-processing tools to filter, sort, deduplicate, or slice the text payloads returned after calling `get_*()` functions before they enter the main reasoning prompt.
4. **Tool Context & Agent Interface Delivery**:
   * Determine the cleanest protocol to expose the Chunk Map and traversal tools to the agent:
     * **`Skills.md`**: Prompt-based skill definitions for coding/CLI agents.
     * **MCP (Model Context Protocol)**: Standardized tool-calling server exposing the document topology.
     * **CLI / Graph API**: Direct command-line utilities or graph query interfaces.
