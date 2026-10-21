"""In-memory document store with BM25, grep, optional vector search, and topology queries."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable
from typing import Any

import litellm
from rank_bm25 import BM25Okapi

from .models import Chunk, DocumentMap, Paragraph, Section

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
PREVIEW_CHARS = 240


class StoreError(Exception):
    """A user-facing store error (unknown document/chunk, ambiguous doc_id)."""


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimension mismatch")
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


class ChunkStore:
    """Holds one or more Chunk Maps and the indexes over their chunks."""

    def __init__(self, documents: Iterable[DocumentMap] | None = None):
        self.documents: dict[str, DocumentMap] = {}
        self.chunks: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._embedding_model = os.getenv("TOPO_RAG_EMBEDDING_MODEL") or None
        self._embeddings: dict[str, list[float]] = {}
        for document in documents or []:
            self.add_document(document)

    # ------------------------------------------------------------------
    # Documents and chunks
    # ------------------------------------------------------------------
    def add_document(self, document: DocumentMap) -> None:
        if document.doc_id in self.documents:
            raise StoreError(f"Duplicate document ID: {document.doc_id}")
        for chunk_id in document.chunk_order:
            if chunk_id in self.chunks:
                raise StoreError(f"Duplicate chunk ID: {chunk_id}")
        self.documents[document.doc_id] = document
        for chunk_id in document.chunk_order:
            self.chunks[chunk_id] = document.chunks[chunk_id]
        # Invalidate the lexical index; embeddings are cached per chunk and stay valid.
        self._bm25 = None
        self._bm25_ids = []

    def resolve_doc_id(self, doc_id: str | None) -> str:
        if doc_id:
            if doc_id not in self.documents:
                available = ", ".join(self.documents) or "<none>"
                raise StoreError(
                    f"Unknown document ID: {doc_id}. Available documents: {available}"
                )
            return doc_id
        if len(self.documents) == 1:
            return next(iter(self.documents))
        if not self.documents:
            raise StoreError("No documents are loaded.")
        available = ", ".join(self.documents)
        raise StoreError(
            f"doc_id is required when multiple documents are loaded: {available}"
        )

    def document(self, doc_id: str | None = None) -> DocumentMap:
        return self.documents[self.resolve_doc_id(doc_id)]

    def get_chunk(self, chunk_id: str) -> Chunk:
        chunk = self.chunks.get(chunk_id)
        if chunk is None:
            raise StoreError(
                f"Unknown chunk ID: {chunk_id}. Use bm25_search or grep to find valid chunks."
            )
        return chunk

    def describe_documents(self) -> list[dict[str, Any]]:
        return [
            {
                "doc_id": document.doc_id,
                "title": document.title,
                "source": document.source,
                "chunk_count": len(document.chunk_order),
                "section_count": len(document.sections),
                "paragraph_count": len(document.paragraphs),
            }
            for document in self.documents.values()
        ]

    def stats(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "sections": sum(len(document.sections) for document in self.documents.values()),
            "paragraphs": sum(
                len(document.paragraphs) for document in self.documents.values()
            ),
            "chunks": len(self.chunks),
        }

    # ------------------------------------------------------------------
    # Topology queries
    # ------------------------------------------------------------------
    def nearest_chunks(self, chunk_id: str, window: int = 1) -> list[Chunk]:
        if window < 0:
            raise StoreError("window must be >= 0")
        center = self.get_chunk(chunk_id)
        chunk_ids = [center.chunk_id]

        cursor = center
        for _ in range(window):
            if not cursor.prev_chunk_id:
                break
            cursor = self.get_chunk(cursor.prev_chunk_id)
            chunk_ids.insert(0, cursor.chunk_id)

        cursor = center
        for _ in range(window):
            if not cursor.next_chunk_id:
                break
            cursor = self.get_chunk(cursor.next_chunk_id)
            chunk_ids.append(cursor.chunk_id)

        return [self.chunks[current_id] for current_id in chunk_ids]

    def section_for_chunk(self, chunk_id: str) -> Section:
        chunk = self.get_chunk(chunk_id)
        document = self.document(chunk.doc_id)
        section = document.sections.get(chunk.section_id)
        if section is None:
            raise StoreError(f"Section not found for chunk: {chunk_id}")
        return section

    def section_chunks(self, chunk_id: str) -> list[Chunk]:
        section = self.section_for_chunk(chunk_id)
        document = self.document(section.doc_id)
        return [
            document.chunks[current_id]
            for current_id in section.chunk_ids
            if current_id in document.chunks
        ]

    def page_chunks(self, chunk_id: str) -> list[Chunk]:
        chunk = self.get_chunk(chunk_id)
        document = self.document(chunk.doc_id)
        return [
            document.chunks[current_id]
            for current_id in document.chunk_order
            if document.chunks[current_id].page_start <= chunk.page_end
            and document.chunks[current_id].page_end >= chunk.page_start
        ]

    def chunk_paragraphs(self, chunk_id: str) -> list[Paragraph]:
        chunk = self.get_chunk(chunk_id)
        document = self.document(chunk.doc_id)
        return [
            document.paragraphs[paragraph_id]
            for paragraph_id in chunk.paragraph_ids
            if paragraph_id in document.paragraphs
        ]

    # ------------------------------------------------------------------
    # Lexical search
    # ------------------------------------------------------------------
    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        chunk_ids = [
            chunk_id
            for document in self.documents.values()
            for chunk_id in document.chunk_order
        ]
        self._bm25_ids = chunk_ids
        if not chunk_ids:
            self._bm25 = None
            return
        corpus = [tokenize(self.chunks[chunk_id].text) for chunk_id in chunk_ids]
        if not any(corpus):
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(corpus)

    def bm25_search(
        self, query: str, top_k: int = 5, doc_id: str | None = None
    ) -> list[dict[str, Any]]:
        if doc_id:
            self.resolve_doc_id(doc_id)
        if top_k <= 0:
            return []
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            zip(self._bm25_ids, scores), key=lambda item: float(item[1]), reverse=True
        )

        results: list[dict[str, Any]] = []
        for chunk_id, score in ranked:
            chunk = self.chunks[chunk_id]
            if doc_id and chunk.doc_id != doc_id:
                continue
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "section_id": chunk.section_id,
                    "section_title": chunk.section_title,
                    "heading_path": chunk.heading_path,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "score": round(float(score), 4),
                    "preview": _preview(chunk.text),
                }
            )
            if len(results) >= top_k:
                break
        return results

    def grep(
        self,
        pattern: str,
        doc_id: str | None = None,
        *,
        regex: bool = False,
        case_sensitive: bool = False,
        max_results: int = 20,
    ) -> dict[str, Any]:
        if doc_id:
            self.resolve_doc_id(doc_id)
        if max_results < 1:
            raise StoreError("max_results must be >= 1")

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(pattern if regex else re.escape(pattern), flags)
        except re.error as exc:
            raise StoreError(f"Invalid regular expression: {exc}") from exc

        matches: list[dict[str, Any]] = []
        truncated = False
        for document in self.documents.values():
            if doc_id and document.doc_id != doc_id:
                continue
            for chunk_id in document.chunk_order:
                chunk = document.chunks[chunk_id]
                for match in compiled.finditer(chunk.text):
                    start = max(0, match.start() - 80)
                    end = min(len(chunk.text), match.end() + 80)
                    snippet = chunk.text[start:end].replace("\n", " ").strip()
                    if start > 0:
                        snippet = "…" + snippet
                    if end < len(chunk.text):
                        snippet += "…"
                    matches.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "section_title": chunk.section_title,
                            "page_start": chunk.page_start,
                            "match": match.group(0),
                            "snippet": snippet,
                        }
                    )
                    if len(matches) >= max_results:
                        truncated = True
                        break
                if truncated:
                    break
            if truncated:
                break

        return {
            "pattern": pattern,
            "regex": regex,
            "count": len(matches),
            "truncated": truncated,
            "matches": matches,
        }

    # ------------------------------------------------------------------
    # Optional semantic search
    # ------------------------------------------------------------------
    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self._embedding_model:
            raise RuntimeError("No embedding model configured.")
        vectors: list[list[float]] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = litellm.embedding(model=self._embedding_model, input=batch)
            data = response.data
            if len(data) != len(batch):
                raise RuntimeError(
                    f"Embedding response returned {len(data)} vectors for {len(batch)} inputs."
                )
            for item in data:
                embedding = (
                    item.get("embedding")
                    if isinstance(item, dict)
                    else getattr(item, "embedding", None)
                )
                if embedding is None:
                    raise RuntimeError("Embedding response did not contain a vector.")
                vectors.append([float(value) for value in embedding])
        return vectors

    def _ensure_embeddings(self, chunk_ids: list[str]) -> None:
        missing = [
            chunk_id for chunk_id in chunk_ids if chunk_id not in self._embeddings
        ]
        if not missing:
            return
        vectors = self._embed_texts([self.chunks[chunk_id].text for chunk_id in missing])
        for chunk_id, vector in zip(missing, vectors):
            self._embeddings[chunk_id] = vector

    def vector_search(
        self, query: str, top_k: int = 5, doc_id: str | None = None
    ) -> dict[str, Any]:
        if doc_id:
            self.resolve_doc_id(doc_id)
        if top_k <= 0:
            return {"mode": "vector", "results": []}

        chunk_ids = [
            chunk_id
            for document in self.documents.values()
            if not doc_id or document.doc_id == doc_id
            for chunk_id in document.chunk_order
        ]

        if not self._embedding_model:
            return {
                "mode": "bm25_fallback",
                "warning": (
                    "No embedding model configured; fell back to BM25. Set "
                    "TOPO_RAG_EMBEDDING_MODEL (e.g. text-embedding-3-small or "
                    "ollama/nomic-embed-text) to enable semantic search."
                ),
                "results": self.bm25_search(query, top_k=top_k, doc_id=doc_id),
            }

        try:
            query_vector = self._embed_texts([query])[0]
            self._ensure_embeddings(chunk_ids)
            scored: list[tuple[str, float]] = []
            for chunk_id in chunk_ids:
                score = _cosine(query_vector, self._embeddings[chunk_id])
                scored.append((chunk_id, score))
            scored.sort(key=lambda item: item[1], reverse=True)

            results: list[dict[str, Any]] = []
            for chunk_id, score in scored[:top_k]:
                chunk = self.chunks[chunk_id]
                results.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "section_id": chunk.section_id,
                        "section_title": chunk.section_title,
                        "heading_path": chunk.heading_path,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "score": round(score, 4),
                        "preview": _preview(chunk.text),
                    }
                )
            return {
                "mode": "vector",
                "model": self._embedding_model,
                "results": results,
            }
        except Exception as exc:  # noqa: BLE001 - fall back for any provider failure
            return {
                "mode": "bm25_fallback",
                "warning": f"Vector search failed ({exc}); fell back to BM25.",
                "results": self.bm25_search(query, top_k=top_k, doc_id=doc_id),
            }
