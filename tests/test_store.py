from __future__ import annotations

import pytest

from src.indexer import build_document_map
from src.store import ChunkStore, StoreError

DOC_A = """# Alpha

Apples are red and crunchy.

## Beta

Bananas are yellow and soft.

<!-- page: 2 -->

Cherries are small and red.
"""

DOC_B = """# Other

Dogs bark loudly.

## Cats

Cats meow softly.
"""


@pytest.fixture()
def store():
    document_a = build_document_map(DOC_A, doc_id="a", chunk_size=8)
    document_b = build_document_map(DOC_B, doc_id="b", chunk_size=8)
    return ChunkStore([document_a, document_b])


def test_bm25_search_finds_relevant_chunk(store):
    results = store.bm25_search("cherries", top_k=3, doc_id="a")
    assert results
    assert "cherries" in results[0]["preview"].lower()
    assert results[0]["doc_id"] == "a"


def test_grep_literal_and_regex(store):
    literal = store.grep("bananas", doc_id="a")
    assert literal["count"] == 1
    assert literal["matches"][0]["chunk_id"].startswith("a:")

    regex = store.grep(r"c(herries|ats)", regex=True)
    assert regex["count"] == 2

    with pytest.raises(StoreError, match="Invalid regular expression"):
        store.grep("(", regex=True)


def test_nearest_chunks_walk_the_chain(store):
    chunk_ids = store.document("a").chunk_order
    assert len(chunk_ids) >= 3
    center = chunk_ids[1]
    chunks = store.nearest_chunks(center, window=1)
    assert [chunk.chunk_id for chunk in chunks] == [chunk_ids[0], chunk_ids[1], chunk_ids[2]]

    assert store.nearest_chunks(chunk_ids[0], window=1)[0].chunk_id == chunk_ids[0]
    assert store.nearest_chunks(chunk_ids[-1], window=1)[-1].chunk_id == chunk_ids[-1]


def test_section_page_and_paragraph_queries(store):
    chunk = next(
        chunk
        for chunk in store.document("a").chunks.values()
        if "cherries" in chunk.text.lower()
    )

    section_chunks = store.section_chunks(chunk.chunk_id)
    assert all(item.section_id == chunk.section_id for item in section_chunks)

    page_chunks = store.page_chunks(chunk.chunk_id)
    assert page_chunks
    assert all(
        item.page_start <= chunk.page_end and item.page_end >= chunk.page_start
        for item in page_chunks
    )

    paragraphs = store.chunk_paragraphs(chunk.chunk_id)
    assert paragraphs
    assert paragraphs[0].paragraph_id in chunk.paragraph_ids


def test_document_resolution_requires_id_with_multiple_documents(store):
    with pytest.raises(StoreError, match="doc_id is required"):
        store.document()

    assert store.document("b").title == "Other"

    with pytest.raises(StoreError, match="Unknown document"):
        store.document("missing")


def test_unknown_chunk_raises(store):
    with pytest.raises(StoreError, match="Unknown chunk"):
        store.get_chunk("a:chunk_9999")

    with pytest.raises(StoreError, match="window"):
        store.nearest_chunks("a:chunk_0000", window=-1)


def test_vector_search_falls_back_to_bm25(monkeypatch):
    monkeypatch.delenv("TOPO_RAG_EMBEDDING_MODEL", raising=False)
    document = build_document_map(DOC_A, doc_id="a", chunk_size=8)
    local_store = ChunkStore([document])
    result = local_store.vector_search("red fruit", top_k=2)
    assert result["mode"] == "bm25_fallback"
    assert "TOPO_RAG_EMBEDDING_MODEL" in result["warning"]
    assert result["results"]


def test_stats(store):
    stats = store.stats()
    assert stats["documents"] == 2
    assert stats["chunks"] == len(store.chunks)
