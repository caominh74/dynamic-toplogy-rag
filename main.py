"""CLI entry point for the topology-based agentic RAG PoC."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.agent import AGENT_TOOL_SCHEMAS, TopoRAGAgent
from src.indexer import build_document_map, slugify
from src.models import Chunk
from src.store import ChunkStore, StoreError
from src.tools import DocumentTopologyTools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="topo-rag",
        description=(
            "Answer questions by traversing a Markdown document topology "
            "(BM25 seed chunks + agentic graph traversal)."
        ),
    )
    parser.add_argument("query", nargs="?", help="Question to answer.")
    parser.add_argument(
        "-q",
        "--query",
        dest="query_option",
        metavar="QUESTION",
        help="Question to answer (alternative to the positional query).",
    )
    parser.add_argument(
        "-f",
        "--file",
        action="append",
        dest="files",
        metavar="PATH",
        help="Markdown file to index. Repeat for multiple documents. Default: data/sample.md",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=os.getenv("TOPO_RAG_MODEL", "gpt-4o-mini"),
        help="LiteLLM model string (default: %(default)s).",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("TOPO_RAG_API_BASE"),
        help="Optional LiteLLM/Ollama API base URL.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Seed chunks (default: 5).")
    parser.add_argument(
        "--max-hops", type=int, default=8, help="Maximum tool-calling rounds (default: 8)."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=400, help="Chunk size in words (default: 400)."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature (default: 0)."
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Print the full text of every chunk cited as evidence.",
    )
    parser.add_argument(
        "--save-trace",
        metavar="PATH",
        help=(
            "Write a full JSON audit trace (tool outputs, seed chunks, and "
            "evidence chunks) after an agent run."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print tool trace and evidence to stderr."
    )

    inspection = parser.add_mutually_exclusive_group()
    inspection.add_argument("--toc", action="store_true", help="Print the table of contents.")
    inspection.add_argument(
        "--headings", action="store_true", help="Print all Markdown headings."
    )
    inspection.add_argument(
        "--sections", action="store_true", help="Print all sections with chunk counts."
    )
    inspection.add_argument(
        "--search", metavar="QUERY", help="Run BM25 search and exit (no LLM call)."
    )
    inspection.add_argument(
        "--retrieval",
        metavar="QUERY",
        help="Compare BM25 and vector retrieval with full chunk text and exit.",
    )
    inspection.add_argument(
        "--chunk",
        action="append",
        metavar="CHUNK_ID",
        help="Print a full chunk by ID. Repeat for multiple chunks.",
    )
    inspection.add_argument(
        "--grep", metavar="PATTERN", help="Run grep and exit (no LLM call)."
    )
    inspection.add_argument(
        "--list-tools", action="store_true", help="Print agent tool schemas and exit."
    )
    parser.add_argument(
        "--regex", action="store_true", help="Treat --grep PATTERN as a regular expression."
    )
    parser.add_argument(
        "--case-sensitive", action="store_true", help="Make --grep matching case-sensitive."
    )
    return parser


def _load_documents(paths: list[str], chunk_size: int):
    if chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    documents = []
    used_ids: set[str] = set()
    for path_string in paths:
        path = Path(path_string).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        base_id = slugify(path.stem)
        doc_id = base_id
        suffix = 2
        while doc_id in used_ids:
            doc_id = f"{base_id}-{suffix}"
            suffix += 1
        used_ids.add(doc_id)

        documents.append(
            build_document_map(
                path.read_text(encoding="utf-8"),
                doc_id=doc_id,
                source=str(path),
                chunk_size=chunk_size,
            )
        )
    return documents


def _default_files() -> list[str]:
    return [str(Path(__file__).resolve().parent / "data" / "sample.md")]


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _print_toc(payload: dict[str, Any]) -> None:
    print(f"{payload['doc_id']} — {payload['title']}")

    def walk(nodes: list[dict[str, Any]], depth: int = 0) -> None:
        for node in nodes:
            print(
                f"{'  ' * depth}- [{node['section_id']}] {node['title']} "
                f"(L{node['level']}, {node['chunk_count']} chunks)"
            )
            walk(node["children"], depth + 1)

    walk(payload["toc"])


def _page_label(page_start: int, page_end: int) -> str:
    if page_start == page_end:
        return f"page {page_start}"
    return f"pages {page_start}-{page_end}"


def _print_chunk(chunk: Chunk) -> None:
    print(
        f"[{chunk.chunk_id}] {chunk.section_title} "
        f"({_page_label(chunk.page_start, chunk.page_end)}, "
        f"{chunk.word_count} words)"
    )
    print(chunk.text)


def _print_hits(
    title: str,
    hits: list[dict[str, Any]],
    warning: str | None = None,
) -> None:
    print(f"\n=== {title} ===")
    if warning:
        print(f"warning: {warning}")
    if not hits:
        print("(no results)")
        return
    for hit in hits:
        print(
            f"\n#{hit['rank']} {hit['chunk_id']} | score {hit['score']:.4f} | "
            f"{hit['section_title']} | "
            f"{_page_label(hit['page_start'], hit['page_end'])}"
        )
        print(hit["text"])


def _print_retrieval(payload: dict[str, Any]) -> None:
    print(f"Query: {payload['query']}")
    print(f"Top-k: {payload['top_k']}")
    _print_hits("BM25", payload["bm25"])
    vector = payload["vector"]
    _print_hits(
        f"Vector ({vector['mode']})",
        vector["results"],
        warning=vector.get("warning"),
    )


def _chunk_dicts(store: ChunkStore, chunk_ids: list[str]) -> list[dict[str, Any]]:
    chunks = []
    for chunk_id in dict.fromkeys(chunk_ids):
        try:
            chunks.append(store.get_chunk(chunk_id).model_dump())
        except StoreError:
            continue
    return chunks


def _print_evidence(
    store: ChunkStore, evidence: list[dict[str, str]]
) -> None:
    print("\n--- Evidence chunks ---")
    if not evidence:
        print("(no evidence recorded)")
        return
    for item in evidence:
        try:
            chunk = store.get_chunk(item["chunk_id"])
        except StoreError as exc:
            print(f"\n[{item['chunk_id']}] error: {exc}")
            continue
        print(
            f"\n[{chunk.chunk_id}] {chunk.section_title} "
            f"({_page_label(chunk.page_start, chunk.page_end)}, "
            f"{chunk.word_count} words)"
        )
        print(chunk.text)
        print(f"Evidence note: {item['note']}")


def _run(args: argparse.Namespace) -> int:
    files = args.files or _default_files()
    documents = _load_documents(files, args.chunk_size)
    store = ChunkStore(documents)
    tools = DocumentTopologyTools(store)

    if args.list_tools:
        _print_json(AGENT_TOOL_SCHEMAS)
        return 0

    if args.toc:
        payloads = [json.loads(tools.get_toc(doc_id)) for doc_id in store.documents]
        if args.json:
            _print_json(payloads)
        else:
            for index, payload in enumerate(payloads):
                if index:
                    print()
                _print_toc(payload)
        return 0

    if args.headings:
        payloads = [
            json.loads(tools.get_all_headings(doc_id)) for doc_id in store.documents
        ]
        if args.json:
            _print_json(payloads)
        else:
            for payload in payloads:
                _print_json(payload)
        return 0

    if args.sections:
        payloads = [
            json.loads(tools.get_all_sect(doc_id)) for doc_id in store.documents
        ]
        if args.json:
            _print_json(payloads)
        else:
            for payload in payloads:
                _print_json(payload)
        return 0

    if args.chunk:
        chunks = []
        errors = []
        for chunk_id in args.chunk:
            try:
                chunks.append(store.get_chunk(chunk_id))
            except StoreError as exc:
                errors.append({"chunk_id": chunk_id, "error": str(exc)})
        if args.json:
            _print_json(
                {
                    "ok": not errors,
                    "chunks": [chunk.model_dump() for chunk in chunks],
                    "errors": errors,
                }
            )
        else:
            for index, chunk in enumerate(chunks):
                if index:
                    print()
                _print_chunk(chunk)
            for error in errors:
                print(f"error: {error['error']}", file=sys.stderr)
        return 1 if errors else 0

    if args.retrieval:
        query = args.retrieval
        top_k = max(1, int(args.top_k))
        bm25_hits = store.bm25_search(query, top_k=top_k)
        vector_payload = store.vector_search(query, top_k=top_k)

        def enrich(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
            rows = []
            for rank, hit in enumerate(hits, start=1):
                row = {key: value for key, value in hit.items() if key != "preview"}
                row["rank"] = rank
                row["text"] = store.get_chunk(hit["chunk_id"]).text
                rows.append(row)
            return rows

        payload = {
            "ok": True,
            "query": query,
            "top_k": top_k,
            "bm25": enrich(bm25_hits),
            "vector": {
                **vector_payload,
                "results": enrich(vector_payload["results"]),
            },
        }
        if args.json:
            _print_json(payload)
        else:
            _print_retrieval(payload)
        return 0

    if args.search:
        payload = json.loads(tools.bm25_search(args.search, top_k=args.top_k))
        _print_json(payload)
        return 0

    if args.grep:
        payload = json.loads(
            tools.grep(
                args.grep,
                regex=args.regex,
                case_sensitive=args.case_sensitive,
                max_results=50,
            )
        )
        _print_json(payload)
        return 0

    if not args.query:
        raise ValueError(
            "Provide a query, or use one of --toc, --headings, --sections, "
            "--chunk, --retrieval, --search, --grep, --list-tools."
        )

    agent = TopoRAGAgent(
        store,
        model_name=args.model,
        top_k=args.top_k,
        max_hops=args.max_hops,
        temperature=args.temperature,
        api_base=args.api_base,
    )
    result = agent.run_query(args.query)
    evidence_chunks = _chunk_dicts(
        store, [item["chunk_id"] for item in result.evidence]
    )

    if args.json:
        payload = result.to_dict()
        if args.show_evidence:
            payload["evidence_chunks"] = evidence_chunks
        _print_json(payload)
    else:
        print(result.answer)
        if args.show_evidence:
            _print_evidence(store, result.evidence)
        if args.verbose:
            print("\n--- Trace ---", file=sys.stderr)
            for entry in result.trace:
                print(
                    f"[hop {entry['hop']}] {entry['tool']} "
                    f"{json.dumps(entry['arguments'], ensure_ascii=False)} "
                    f"-> {entry['result_chars']} chars",
                    file=sys.stderr,
                )
            print("\n--- Evidence ---", file=sys.stderr)
            for item in result.evidence:
                print(f"[{item['chunk_id']}] {item['note']}", file=sys.stderr)
            print(f"\nSeeds: {', '.join(result.seed_chunk_ids) or '(none)'}", file=sys.stderr)

    if args.save_trace:
        trace_payload = result.to_dict(include_tool_outputs=True)
        trace_payload["seed_chunks"] = _chunk_dicts(store, result.seed_chunk_ids)
        trace_payload["evidence_chunks"] = evidence_chunks
        trace_path = Path(args.save_trace).expanduser()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(trace_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved full trace to {trace_path}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.query_option is not None:
        if args.query is not None:
            parser.error(
                "Provide the query either positionally or with -q/--query, not both."
            )
        args.query = args.query_option
    try:
        return _run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - present a clean CLI error
        print(f"error: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
