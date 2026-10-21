from __future__ import annotations

from src.indexer import build_document_map

MARKDOWN = """# Alpha

Alpha intro about apples.

## Beta

Beta text about bananas and tables.

<!-- page: 2 -->

Beta page two text about cherries.

### Gamma

Gamma text about grapes.

# Delta

Delta text about apples and oranges.
"""


def _section(document, title):
    return next(section for section in document.sections.values() if section.title == title)


def test_heading_hierarchy_and_title():
    document = build_document_map(MARKDOWN, doc_id="notes", chunk_size=10)

    assert document.title == "Alpha"
    assert [entry.title for entry in document.toc] == [
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
    ]

    alpha = _section(document, "Alpha")
    beta = _section(document, "Beta")
    gamma = _section(document, "Gamma")
    delta = _section(document, "Delta")

    assert alpha.level == 1 and alpha.parent_id is None
    assert beta.level == 2 and beta.parent_id == alpha.section_id
    assert gamma.level == 3 and gamma.parent_id == beta.section_id
    assert delta.level == 1 and delta.parent_id is None
    assert gamma.heading_path == ["Alpha", "Beta", "Gamma"]


def test_chunk_topology_and_page_links():
    document = build_document_map(MARKDOWN, doc_id="notes", chunk_size=10)

    assert len(document.chunk_order) >= 5
    assert document.chunk_order[0].endswith(":chunk_0000")

    for index, chunk_id in enumerate(document.chunk_order):
        chunk = document.chunks[chunk_id]
        expected_prev = document.chunk_order[index - 1] if index > 0 else None
        expected_next = (
            document.chunk_order[index + 1]
            if index + 1 < len(document.chunk_order)
            else None
        )
        assert chunk.prev_chunk_id == expected_prev
        assert chunk.next_chunk_id == expected_next

    cherry_chunks = [
        chunk for chunk in document.chunks.values() if "cherries" in chunk.text
    ]
    assert len(cherry_chunks) == 1
    assert cherry_chunks[0].page_start == 2
    assert cherry_chunks[0].section_title == "Beta"


def test_paragraph_metadata_and_oversized_paragraph_split():
    text = "# A\n\n" + " ".join(f"word{i}" for i in range(20)) + "\n"
    document = build_document_map(text, doc_id="long", chunk_size=5)

    assert len(document.paragraphs) == 1
    paragraph = next(iter(document.paragraphs.values()))
    assert len(document.chunks) == 4
    for chunk in document.chunks.values():
        assert chunk.paragraph_ids == [paragraph.paragraph_id]


def test_page_marker_and_form_feed():
    marker_document = build_document_map(
        "# One\n\nBefore.\n<!-- page: 3 -->\nAfter.\n", doc_id="pages", chunk_size=10
    )
    pages = sorted(
        (chunk.page_start, chunk.page_end) for chunk in marker_document.chunks.values()
    )
    assert pages == [(1, 1), (3, 3)]

    form_feed_document = build_document_map(
        "# One\n\nBefore.\fAfter.\n", doc_id="ff", chunk_size=10
    )
    pages = sorted(
        (chunk.page_start, chunk.page_end) for chunk in form_feed_document.chunks.values()
    )
    assert pages == [(1, 1), (2, 2)]


def test_headings_inside_code_fences_are_ignored():
    text = """# Real

```python
# Not a heading
print("hello")
```

Body text.
"""
    document = build_document_map(text, doc_id="code", chunk_size=10)
    assert [entry.title for entry in document.toc] == ["Real"]
    assert len(document.chunks) == 1


def test_heading_only_section_has_no_chunks():
    text = "# A\n## B\nText under B.\n"
    document = build_document_map(text, doc_id="empty-section", chunk_size=10)

    section_a = _section(document, "A")
    section_b = _section(document, "B")
    assert section_a.chunk_ids == []
    assert len(section_b.chunk_ids) == 1
    assert section_b.parent_id == section_a.section_id


def test_preamble_is_indexed():
    text = "Loose introduction.\n\n# First\n\nBody.\n"
    document = build_document_map(text, doc_id="pre", chunk_size=10)
    assert document.toc[0].title == "Preamble"
    assert document.toc[0].level == 0
    assert document.toc[1].title == "First"
    assert document.toc[1].parent_id == document.toc[0].section_id


def test_empty_document():
    document = build_document_map("", doc_id="blank")
    assert document.toc == []
    assert document.sections == {}
    assert document.chunks == {}
    assert document.title == "blank"
