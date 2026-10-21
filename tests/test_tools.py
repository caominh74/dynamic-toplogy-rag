from __future__ import annotations

import json

import pytest

from src.indexer import build_document_map
from src.store import ChunkStore
from src.tools import TOOL_SCHEMAS, DocumentTopologyTools

DOC = """# Alpha

Apples are red and crunchy.

## Beta

Bananas are yellow and soft.

<!-- page: 2 -->

Cherries are small and red.
"""


@pytest.fixture()
def tools():
    document = build_document_map(DOC, doc_id="a", chunk_size=8)
    return DocumentTopologyTools(ChunkStore([document]))


def _load(result: str):
    payload = json.loads(result)
    assert "ok" in payload
    return payload


def test_toc_headings_sections_and_documents(tools):
    toc = _load(tools.get_toc())
    assert toc["ok"] is True
    assert toc["doc_id"] == "a"
    assert toc["toc"][0]["title"] == "Alpha"
    assert toc["toc"][0]["children"][0]["title"] == "Beta"

    headings = _load(tools.get_all_headings())
    assert [item["title"] for item in headings["headings"]] == ["Alpha", "Beta"]

    sections = _load(tools.get_all_sect())
    assert [item["title"] for item in sections["sections"]] == ["Alpha", "Beta"]

    documents = _load(tools.list_documents())
    assert documents["documents"][0]["doc_id"] == "a"


def test_get_chunk_and_missing_chunk(tools):
    chunk_id = tools.store.document("a").chunk_order[0]
    payload = _load(tools.get_chunk(chunk_id, max_chars=1000))
    assert payload["ok"] is True
    assert payload["chunk"]["chunk_id"] == chunk_id
    assert payload["chunk"]["text"]

    missing = _load(tools.get_chunk("a:chunk_9999"))
    assert missing["ok"] is False
    assert "Unknown chunk" in missing["error"]


def test_navigation_tools(tools):
    chunk_id = tools.store.document("a").chunk_order[0]
    nearest = _load(tools.get_nearest_chunks(chunk_id, window=1))
    assert nearest["center_chunk_id"] == chunk_id
    assert nearest["chunks"]

    section = _load(tools.get_chunk_section(chunk_id, mode="peek"))
    assert section["mode"] == "peek"
    assert section["chunks"][0]["preview"]

    full = _load(tools.get_chunk_section(chunk_id, mode="read", max_chars=5000))
    assert full["text"]
    assert full["truncated"] is False

    page = _load(tools.get_chunk_page(chunk_id))
    assert page["chunk_ids"]

    paragraph = _load(tools.get_chunk_para(chunk_id))
    assert paragraph["paragraphs"]
    assert paragraph["paragraphs"][0]["text"]


def test_invalid_section_mode(tools):
    chunk_id = tools.store.document("a").chunk_order[0]
    payload = _load(tools.get_chunk_section(chunk_id, mode="everything"))
    assert payload["ok"] is False


def test_search_tools(tools):
    bm25 = _load(tools.bm25_search("cherries", top_k=2))
    assert bm25["ok"] is True
    assert "cherries" in bm25["results"][0]["preview"].lower()

    grep = _load(tools.grep("bananas"))
    assert grep["count"] == 1

    vector = _load(tools.vector_search("red fruit", top_k=2))
    assert vector["mode"] in {"vector", "bm25_fallback"}
    assert vector["results"]


def test_dispatch_filters_extra_arguments_and_reports_unknown_tools(tools):
    chunk_id = tools.store.document("a").chunk_order[0]
    payload = _load(
        tools.dispatch(
            "get_chunk", {"chunk_id": chunk_id, "this_argument_does_not_exist": 1}
        )
    )
    assert payload["ok"] is True

    payload = _load(tools.dispatch("get_chunk", json.dumps({"chunk_id": chunk_id})))
    assert payload["ok"] is True

    payload = _load(tools.dispatch("delete_everything", {}))
    assert payload["ok"] is False
    assert "Unknown tool" in payload["error"]

    payload = _load(tools.dispatch("get_chunk", {"wrong": "shape"}))
    assert payload["ok"] is False


def test_schemas_are_unique_and_well_formed():
    names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    assert len(names) == len(set(names))
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert schema["function"]["parameters"]["type"] == "object"
