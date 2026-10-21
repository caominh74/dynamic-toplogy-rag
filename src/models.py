"""Data models for the document topology (Chunk Map)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TocEntry(BaseModel):
    """A node in the document's table of contents."""

    section_id: str
    title: str
    level: int
    parent_id: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    page_start: int = 1
    page_end: int = 1
    chunk_ids: list[str] = Field(default_factory=list)


class Section(BaseModel):
    """A Markdown heading and the content beneath it."""

    section_id: str
    doc_id: str
    title: str
    level: int
    parent_id: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    page_start: int = 1
    page_end: int = 1
    chunk_ids: list[str] = Field(default_factory=list)
    paragraph_ids: list[str] = Field(default_factory=list)
    order: int = 0


class Paragraph(BaseModel):
    """A blank-line-delimited block within a section."""

    paragraph_id: str
    doc_id: str
    section_id: str
    page: int
    order: int
    text: str


class Chunk(BaseModel):
    """A retrievable unit with links back into the document topology."""

    chunk_id: str
    doc_id: str
    section_id: str
    section_title: str
    heading_path: list[str] = Field(default_factory=list)
    text: str
    page_start: int = 1
    page_end: int = 1
    paragraph_ids: list[str] = Field(default_factory=list)
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    first_sentence: str = ""
    word_count: int = 0
    order: int = 0


class DocumentMap(BaseModel):
    """The complete Chunk Map for one Markdown document."""

    doc_id: str
    title: str
    source: str | None = None
    toc: list[TocEntry] = Field(default_factory=list)
    sections: dict[str, Section] = Field(default_factory=dict)
    paragraphs: dict[str, Paragraph] = Field(default_factory=dict)
    chunks: dict[str, Chunk] = Field(default_factory=dict)
    chunk_order: list[str] = Field(default_factory=list)
