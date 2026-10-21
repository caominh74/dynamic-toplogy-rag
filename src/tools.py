"""Agent-facing tools over the document topology.

Every tool returns a JSON string with an ``ok`` flag so the agent can recover
from invalid IDs or arguments instead of crashing the loop.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

from .models import Chunk, DocumentMap, Section
from .store import ChunkStore, StoreError

TOOL_MAX_CHARS = 50_000


def json_ok(**payload: Any) -> str:
    return json.dumps({"ok": True, **payload}, ensure_ascii=False, indent=2)


def json_error(message: str, **extra: Any) -> str:
    return json.dumps(
        {"ok": False, "error": message, **extra}, ensure_ascii=False, indent=2
    )


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 20)].rstrip() + "\n... [truncated]"


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _chunk_payload(chunk: Chunk, max_chars: int) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "section_id": chunk.section_id,
        "section_title": chunk.section_title,
        "heading_path": chunk.heading_path,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "word_count": chunk.word_count,
        "prev_chunk_id": chunk.prev_chunk_id,
        "next_chunk_id": chunk.next_chunk_id,
        "text": _truncate(chunk.text, max_chars),
    }


def _section_payload(section: Section) -> dict[str, Any]:
    return {
        "section_id": section.section_id,
        "doc_id": section.doc_id,
        "title": section.title,
        "level": section.level,
        "parent_id": section.parent_id,
        "heading_path": section.heading_path,
        "page_start": section.page_start,
        "page_end": section.page_end,
        "chunk_count": len(section.chunk_ids),
        "paragraph_count": len(section.paragraph_ids),
    }


def _toc_tree(document: DocumentMap) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    for entry in document.toc:
        node = {
            "section_id": entry.section_id,
            "title": entry.title,
            "level": entry.level,
            "parent_id": entry.parent_id,
            "page_start": entry.page_start,
            "page_end": entry.page_end,
            "chunk_count": len(entry.chunk_ids),
            "children": [],
        }
        while stack and stack[-1]["level"] >= entry.level:
            stack.pop()
        if stack:
            stack[-1]["children"].append(node)
        else:
            roots.append(node)
        stack.append(node)
    return roots


class DocumentTopologyTools:
    """Read-only navigation tools plus lexical/semantic search."""

    def __init__(self, store: ChunkStore):
        self.store = store

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def list_documents(self) -> str:
        return json_ok(documents=self.store.describe_documents())

    def get_toc(self, doc_id: str | None = None) -> str:
        try:
            document = self.store.document(doc_id)
        except StoreError as exc:
            return json_error(str(exc))
        return json_ok(
            doc_id=document.doc_id,
            title=document.title,
            toc=_toc_tree(document),
            section_count=len(document.toc),
        )

    def get_all_headings(self, doc_id: str | None = None) -> str:
        try:
            document = self.store.document(doc_id)
        except StoreError as exc:
            return json_error(str(exc))
        headings = [
            {
                "section_id": entry.section_id,
                "title": entry.title,
                "level": entry.level,
                "parent_id": entry.parent_id,
                "heading_path": entry.heading_path,
                "page_start": entry.page_start,
                "chunk_count": len(entry.chunk_ids),
            }
            for entry in document.toc
            if entry.level > 0
        ]
        return json_ok(doc_id=document.doc_id, headings=headings)

    def get_all_sect(self, doc_id: str | None = None) -> str:
        try:
            document = self.store.document(doc_id)
        except StoreError as exc:
            return json_error(str(exc))
        sections = [
            _section_payload(section)
            for section in sorted(document.sections.values(), key=lambda item: item.order)
        ]
        return json_ok(doc_id=document.doc_id, sections=sections)

    # ------------------------------------------------------------------
    # Chunk retrieval
    # ------------------------------------------------------------------
    def get_chunk(self, chunk_id: str, max_chars: int = 6000) -> str:
        try:
            chunk = self.store.get_chunk(chunk_id)
        except StoreError as exc:
            return json_error(str(exc))
        return json_ok(
            chunk=_chunk_payload(chunk, _clamp(max_chars, 200, TOOL_MAX_CHARS))
        )

    def get_nearest_chunks(self, chunk_id: str, window: int = 2) -> str:
        try:
            chunks = self.store.nearest_chunks(chunk_id, _clamp(window, 0, 10))
        except StoreError as exc:
            return json_error(str(exc))
        return json_ok(
            center_chunk_id=chunk_id,
            window=int(window),
            chunks=[_chunk_payload(chunk, 2500) for chunk in chunks],
        )

    def get_chunk_section(
        self, chunk_id: str, mode: str = "peek", max_chars: int = 12000
    ) -> str:
        if mode not in {"peek", "read"}:
            return json_error("mode must be 'peek' or 'read'")
        try:
            self.store.get_chunk(chunk_id)
            section = self.store.section_for_chunk(chunk_id)
            chunks = self.store.section_chunks(chunk_id)
        except StoreError as exc:
            return json_error(str(exc))

        payload: dict[str, Any] = {
            "section": _section_payload(section),
            "mode": mode,
            "chunk_ids": [item.chunk_id for item in chunks],
        }
        if mode == "peek":
            payload["chunks"] = [
                {
                    "chunk_id": item.chunk_id,
                    "page_start": item.page_start,
                    "page_end": item.page_end,
                    "preview": item.first_sentence,
                }
                for item in chunks
            ]
        else:
            combined = "\n\n".join(
                f"[{item.chunk_id}] {item.text}" for item in chunks
            )
            limit = _clamp(max_chars, 500, TOOL_MAX_CHARS)
            payload["truncated"] = len(combined) > limit
            payload["text"] = _truncate(combined, limit)
        return json_ok(**payload)

    def get_chunk_page(self, chunk_id: str, max_chars: int = 12000) -> str:
        try:
            chunk = self.store.get_chunk(chunk_id)
            chunks = self.store.page_chunks(chunk_id)
        except StoreError as exc:
            return json_error(str(exc))

        combined = "\n\n".join(f"[{item.chunk_id}] {item.text}" for item in chunks)
        limit = _clamp(max_chars, 500, TOOL_MAX_CHARS)
        return json_ok(
            center_chunk_id=chunk.chunk_id,
            page_start=min(item.page_start for item in chunks) if chunks else chunk.page_start,
            page_end=max(item.page_end for item in chunks) if chunks else chunk.page_end,
            chunk_ids=[item.chunk_id for item in chunks],
            truncated=len(combined) > limit,
            text=_truncate(combined, limit),
        )

    def get_chunk_para(self, chunk_id: str, max_chars: int = 8000) -> str:
        try:
            chunk = self.store.get_chunk(chunk_id)
            paragraphs = self.store.chunk_paragraphs(chunk_id)
        except StoreError as exc:
            return json_error(str(exc))

        limit = _clamp(max_chars, 500, TOOL_MAX_CHARS)
        per_paragraph = max(300, limit // max(1, len(paragraphs)))
        combined = "\n\n".join(
            f"[{paragraph.paragraph_id}] {paragraph.text}" for paragraph in paragraphs
        )
        return json_ok(
            chunk_id=chunk.chunk_id,
            paragraphs=[
                {
                    "paragraph_id": paragraph.paragraph_id,
                    "page": paragraph.page,
                    "text": _truncate(paragraph.text, per_paragraph),
                }
                for paragraph in paragraphs
            ],
            truncated=len(combined) > limit,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def grep(
        self,
        pattern: str,
        doc_id: str | None = None,
        regex: bool = False,
        case_sensitive: bool = False,
        max_results: int = 20,
    ) -> str:
        try:
            result = self.store.grep(
                pattern,
                doc_id,
                regex=regex,
                case_sensitive=case_sensitive,
                max_results=_clamp(max_results, 1, 100),
            )
        except StoreError as exc:
            return json_error(str(exc))
        return json_ok(**result)

    def bm25_search(
        self, query: str, top_k: int = 5, doc_id: str | None = None
    ) -> str:
        try:
            results = self.store.bm25_search(
                query, top_k=_clamp(top_k, 1, 20), doc_id=doc_id
            )
        except StoreError as exc:
            return json_error(str(exc))
        return json_ok(query=query, results=results)

    def vector_search(
        self, query: str, top_k: int = 5, doc_id: str | None = None
    ) -> str:
        try:
            result = self.store.vector_search(
                query, top_k=_clamp(top_k, 1, 20), doc_id=doc_id
            )
        except StoreError as exc:
            return json_error(str(exc))
        return json_ok(query=query, **result)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def dispatch(self, name: str, arguments: dict[str, Any] | str | None = None) -> str:
        if name not in _TOOL_NAMES:
            return json_error(
                f"Unknown tool: {name}. Available tools: {', '.join(sorted(_TOOL_NAMES))}"
            )
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                return json_error(f"Tool arguments are not valid JSON: {exc}")
            arguments = parsed
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return json_error("Tool arguments must be a JSON object.")

        method = getattr(self, name)
        signature = inspect.signature(method)
        allowed = {
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.kind
            in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        }
        filtered = {key: value for key, value in arguments.items() if key in allowed}
        try:
            return method(**filtered)
        except TypeError as exc:
            return json_error(f"Invalid arguments for {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - keep the agent loop alive
            return json_error(f"{name} failed: {exc}")


_DOC_ID_SCHEMA = {
    "type": "string",
    "description": "Document ID. Omit when only one document is loaded.",
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List loaded documents with IDs, titles, and chunk counts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_toc",
            "description": (
                "Return the hierarchical table of contents of a document, including "
                "section IDs and chunk counts."
            ),
            "parameters": {
                "type": "object",
                "properties": {"doc_id": _DOC_ID_SCHEMA},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_headings",
            "description": "Return a flat list of every Markdown heading with its level and section ID.",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": _DOC_ID_SCHEMA},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_sect",
            "description": "Return an inventory of all sections with page ranges, chunk counts, and paragraph counts.",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": _DOC_ID_SCHEMA},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk",
            "description": (
                "Read the full text and metadata of a known chunk. Use this first on "
                "the seed chunks provided in the user message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string", "description": "Exact chunk ID."},
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters of chunk text to return.",
                        "default": 6000,
                    },
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_nearest_chunks",
            "description": (
                "Return the previous and next chunks around a chunk. Use this when a "
                "chunk begins or ends mid-thought, or references nearby context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string", "description": "Center chunk ID."},
                    "window": {
                        "type": "integer",
                        "description": "Number of chunks to walk in each direction.",
                        "default": 2,
                        "minimum": 0,
                        "maximum": 10,
                    },
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk_section",
            "description": (
                "Return the chunk's entire parent section. Use mode='peek' for short "
                "previews first, then mode='read' for the full section text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string", "description": "Any chunk in the section."},
                    "mode": {
                        "type": "string",
                        "enum": ["peek", "read"],
                        "default": "peek",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters when mode='read'.",
                        "default": 12000,
                    },
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk_page",
            "description": (
                "Return every chunk that overlaps the same page as the given chunk. "
                "Only useful when the document contains page markers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string", "description": "Chunk on the target page."},
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters of page text to return.",
                        "default": 12000,
                    },
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk_para",
            "description": "Return the original paragraph blocks contained in a chunk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string", "description": "Target chunk ID."},
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum total characters of paragraph text.",
                        "default": 8000,
                    },
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Exact keyword or regular-expression search over chunk text. Returns "
                "matching chunk IDs and snippets. Use it for names, identifiers, and "
                "rare terms that BM25 may rank poorly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Literal text or regex."},
                    "doc_id": _DOC_ID_SCHEMA,
                    "regex": {
                        "type": "boolean",
                        "description": "Interpret pattern as a regular expression.",
                        "default": False,
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Match case-sensitively.",
                        "default": False,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matches.",
                        "default": 20,
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bm25_search",
            "description": (
                "Rank chunks lexically with BM25. Use this to find additional seed "
                "chunks when the initial seeds are insufficient."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results.",
                        "default": 5,
                    },
                    "doc_id": _DOC_ID_SCHEMA,
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": (
                "Semantic search over chunks. Falls back to BM25 if no embedding "
                "model is configured. Use it when wording differs from the document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language query."},
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results.",
                        "default": 5,
                    },
                    "doc_id": _DOC_ID_SCHEMA,
                },
                "required": ["query"],
            },
        },
    },
]

_TOOL_NAMES = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
