"""Build a document topology (Chunk Map) from clean Markdown.

The indexer intentionally works on Markdown produced by a converter that
bypasses traditional parsing/OCR. It extracts the document's native structure
instead of treating the text as one flat blob.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Chunk, DocumentMap, Paragraph, Section, TocEntry

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
PAGE_RE = re.compile(r"^\s*<!--\s*page\s*:?\s*(\d+)\s*-->\s*$", re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*(```|~~~)")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def slugify(value: str) -> str:
    """Turn a filename/title into a stable document ID."""

    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value or "doc"


def _is_fence(line: str) -> bool:
    return bool(FENCE_RE.match(line))


def _first_sentence(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    sentence = SENTENCE_RE.split(text, maxsplit=1)[0]
    if len(sentence) > limit:
        sentence = sentence[: limit - 1].rstrip() + "…"
    return sentence


def _split_words(text: str, size: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


@dataclass
class _RawSection:
    section_id: str
    title: str
    level: int
    page: int
    parent_id: str | None
    heading_path: list[str]
    order: int
    lines: list[tuple[int, str]] = field(default_factory=list)


def _paragraphs_for_section(
    raw: _RawSection, doc_id: str, start_order: int
) -> list[Paragraph]:
    """Split a section into blank-line-delimited paragraphs, never crossing pages."""

    paragraphs: list[Paragraph] = []
    buffer: list[str] = []
    buffer_page: int | None = None

    def flush() -> None:
        nonlocal buffer, buffer_page
        text = "\n".join(buffer).strip()
        if text:
            paragraphs.append(
                Paragraph(
                    paragraph_id=f"{doc_id}:p_{start_order + len(paragraphs):04d}",
                    doc_id=doc_id,
                    section_id=raw.section_id,
                    page=buffer_page or raw.page,
                    order=start_order + len(paragraphs),
                    text=text,
                )
            )
        buffer = []
        buffer_page = None

    for page, line in raw.lines:
        if buffer_page is not None and page != buffer_page:
            flush()
        if not line.strip():
            flush()
            continue
        if buffer_page is None:
            buffer_page = page
        buffer.append(line)
    flush()
    return paragraphs


def _chunks_for_section(
    section: Section,
    paragraphs: list[Paragraph],
    doc_id: str,
    chunk_size: int,
    start_chunk_index: int,
    start_order: int,
) -> list[Chunk]:
    """Pack paragraphs into word-bounded chunks without crossing section/page edges."""

    chunks: list[Chunk] = []
    current_texts: list[str] = []
    current_para_ids: list[str] = []
    current_pages: list[int] = []
    current_words = 0

    def make(text: str, para_ids: list[str], pages: list[int]) -> None:
        text = text.strip()
        if not text or not pages:
            return
        chunk_index = start_chunk_index + len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:chunk_{chunk_index:04d}",
                doc_id=doc_id,
                section_id=section.section_id,
                section_title=section.title,
                heading_path=list(section.heading_path),
                text=text,
                page_start=min(pages),
                page_end=max(pages),
                paragraph_ids=list(dict.fromkeys(para_ids)),
                first_sentence=_first_sentence(text),
                word_count=len(text.split()),
                order=start_order + len(chunks),
            )
        )

    def flush() -> None:
        nonlocal current_texts, current_para_ids, current_pages, current_words
        if not current_texts:
            return
        make("\n\n".join(current_texts), current_para_ids, current_pages)
        current_texts = []
        current_para_ids = []
        current_pages = []
        current_words = 0

    for paragraph in paragraphs:
        paragraph_words = len(paragraph.text.split())
        if paragraph_words == 0:
            continue

        if current_texts and (
            current_words + paragraph_words > chunk_size
            or paragraph.page not in current_pages
        ):
            flush()

        # A single oversized paragraph is split, but keeps its paragraph ID so
        # get_chunk_para can still return the original block.
        if paragraph_words > chunk_size:
            for piece in _split_words(paragraph.text, chunk_size):
                make(piece, [paragraph.paragraph_id], [paragraph.page])
            continue

        current_texts.append(paragraph.text)
        current_words += paragraph_words
        current_para_ids.append(paragraph.paragraph_id)
        if paragraph.page not in current_pages:
            current_pages.append(paragraph.page)

    flush()
    return chunks


def build_document_map(
    markdown_text: str,
    doc_id: str = "doc",
    *,
    source: str | None = None,
    title: str | None = None,
    chunk_size: int = 400,
) -> DocumentMap:
    """Parse Markdown into sections, paragraphs, chunks, and topology links.

    Page mapping is optional. Use ``<!-- page: 2 -->`` (or ``<!-- page 2 -->``)
    markers, or form-feed characters, to introduce page boundaries. Plain
    Markdown is treated as page 1.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    raw_sections: list[_RawSection] = []
    stack: list[_RawSection] = []
    current: _RawSection | None = None
    document_root: _RawSection | None = None
    preamble: list[tuple[int, str]] = []
    page = 1
    in_fence = False

    def add_section(
        section_title: str,
        level: int,
        section_page: int,
        parent: _RawSection | None,
        heading_path: list[str],
    ) -> _RawSection:
        section = _RawSection(
            section_id=f"{doc_id}:sec_{len(raw_sections):04d}",
            title=section_title,
            level=level,
            page=section_page,
            parent_id=parent.section_id if parent else None,
            heading_path=heading_path,
            order=len(raw_sections),
        )
        raw_sections.append(section)
        return section

    for raw_line in markdown_text.split("\n"):
        line = raw_line.rstrip("\r")

        if not in_fence:
            page_marker = PAGE_RE.match(line)
            if page_marker:
                page = max(1, int(page_marker.group(1)))
                continue

        # Form feeds can appear inline; each one starts a new page.
        segments = line.split("\f")
        for index, segment in enumerate(segments):
            if index > 0:
                page += 1

            if _is_fence(segment):
                in_fence = not in_fence

            heading = None if in_fence else HEADING_RE.match(segment)
            if heading:
                if current is None and any(text.strip() for _, text in preamble):
                    preamble_section = add_section(
                        "Preamble", 0, preamble[0][0], None, ["Preamble"]
                    )
                    preamble_section.lines = preamble
                    preamble = []
                    document_root = preamble_section

                level = len(heading.group(1))
                heading_title = heading.group(2).strip()
                while stack and stack[-1].level >= level:
                    stack.pop()
                parent = stack[-1] if stack else document_root
                heading_path = [section.title for section in stack] + [heading_title]
                current = add_section(
                    heading_title, level, page, parent, heading_path
                )
                stack.append(current)
                continue

            if current is None:
                preamble.append((page, segment))
            else:
                current.lines.append((page, segment))

    if current is None and any(text.strip() for _, text in preamble):
        preamble_section = add_section("Preamble", 0, preamble[0][0], None, ["Preamble"])
        preamble_section.lines = preamble
        document_root = preamble_section

    sections: dict[str, Section] = {}
    toc: list[TocEntry] = []
    paragraphs: dict[str, Paragraph] = {}
    chunks: dict[str, Chunk] = {}
    chunk_order: list[str] = []
    paragraph_counter = 0

    for raw in raw_sections:
        section_paragraphs = _paragraphs_for_section(raw, doc_id, paragraph_counter)
        paragraph_counter += len(section_paragraphs)
        for paragraph in section_paragraphs:
            paragraphs[paragraph.paragraph_id] = paragraph

        page_start = min(
            (paragraph.page for paragraph in section_paragraphs), default=raw.page
        )
        page_end = max(
            (paragraph.page for paragraph in section_paragraphs), default=raw.page
        )
        section = Section(
            section_id=raw.section_id,
            doc_id=doc_id,
            title=raw.title,
            level=raw.level,
            parent_id=raw.parent_id,
            heading_path=list(raw.heading_path),
            page_start=page_start,
            page_end=page_end,
            paragraph_ids=[paragraph.paragraph_id for paragraph in section_paragraphs],
            order=raw.order,
        )

        section_chunks = _chunks_for_section(
            section,
            section_paragraphs,
            doc_id,
            chunk_size,
            start_chunk_index=len(chunks),
            start_order=len(chunk_order),
        )
        for chunk in section_chunks:
            section.chunk_ids.append(chunk.chunk_id)
            chunks[chunk.chunk_id] = chunk
            chunk_order.append(chunk.chunk_id)

        sections[section.section_id] = section
        toc.append(
            TocEntry(
                section_id=section.section_id,
                title=section.title,
                level=section.level,
                parent_id=section.parent_id,
                heading_path=list(section.heading_path),
                page_start=section.page_start,
                page_end=section.page_end,
                chunk_ids=list(section.chunk_ids),
            )
        )

    for index, chunk_id in enumerate(chunk_order):
        chunk = chunks[chunk_id]
        chunk.prev_chunk_id = chunk_order[index - 1] if index > 0 else None
        chunk.next_chunk_id = (
            chunk_order[index + 1] if index + 1 < len(chunk_order) else None
        )

    document_title = (
        title
        or next((raw.title for raw in raw_sections if raw.level == 1), None)
        or next((raw.title for raw in raw_sections if raw.level > 0), None)
        or doc_id
    )

    return DocumentMap(
        doc_id=doc_id,
        title=document_title,
        source=source,
        toc=toc,
        sections=sections,
        paragraphs=paragraphs,
        chunks=chunks,
        chunk_order=chunk_order,
    )
