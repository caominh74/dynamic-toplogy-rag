"""Agentic traversal loop."""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import litellm

from .store import ChunkStore, StoreError
from .tools import (
    TOOL_SCHEMAS,
    DocumentTopologyTools,
    json_error,
    json_ok,
)

EVIDENCE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "record_evidence",
        "description": (
            "Save a distilled finding from a chunk to working memory. Call this before "
            "your final answer. Keep notes short and include the supporting chunk ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_id": {
                    "type": "string",
                    "description": "Chunk that supports the finding.",
                },
                "note": {
                    "type": "string",
                    "description": "One to three sentences summarizing the finding.",
                },
            },
            "required": ["chunk_id", "note"],
        },
    },
}

AGENT_TOOL_SCHEMAS = [*TOOL_SCHEMAS, EVIDENCE_TOOL_SCHEMA]


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass
class AgentResult:
    answer: str
    model: str
    seed_chunk_ids: list[str]
    evidence: list[dict[str, str]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    hops: int = 0
    tool_outputs: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def to_dict(self, *, include_tool_outputs: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not include_tool_outputs:
            payload.pop("tool_outputs", None)
        return payload


class TopoRAGAgent:
    """A small tool-calling agent that navigates the Chunk Map."""

    def __init__(
        self,
        store: ChunkStore,
        model_name: str | None = None,
        *,
        top_k: int = 5,
        max_hops: int = 8,
        temperature: float = 0.0,
        api_base: str | None = None,
    ):
        self.store = store
        self.model_name = (
            model_name or os.getenv("TOPO_RAG_MODEL") or "gpt-4o-mini"
        )
        self.top_k = max(1, int(top_k))
        self.max_hops = max(1, int(max_hops))
        self.temperature = float(temperature)
        self.api_base = api_base or os.getenv("TOPO_RAG_API_BASE")
        self.tools = DocumentTopologyTools(store)
        self.evidence: list[dict[str, str]] = []
        self.trace: list[dict[str, Any]] = []
        self.tool_outputs: list[dict[str, Any]] = []

        # Keep provider differences from cluttering the CLI.
        litellm.suppress_debug_info = True
        litellm.drop_params = True

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------
    def record_evidence(self, chunk_id: str, note: str) -> str:
        try:
            self.store.get_chunk(chunk_id)
        except StoreError as exc:
            return json_error(str(exc))

        note = " ".join((note or "").split()).strip()
        if not note:
            return json_error("note must not be empty")
        if len(note) > 4000:
            note = note[:3990].rstrip() + "..."

        existing = any(
            item["chunk_id"] == chunk_id and item["note"] == note
            for item in self.evidence
        )
        if not existing:
            self.evidence.append({"chunk_id": chunk_id, "note": note})
        return json_ok(
            recorded=not existing,
            chunk_id=chunk_id,
            evidence_count=len(self.evidence),
        )

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------
    def _system_prompt(self) -> str:
        documents = "\n".join(
            f"- {item['doc_id']}: {item['title']} "
            f"({item['chunk_count']} chunks, {item['section_count']} sections)"
            for item in self.store.describe_documents()
        )
        return (
            "You are a document research agent that answers questions by traversing "
            "a document topology. The document has been split into chunks linked by "
            "section, page, paragraph, and sequential order.\n\n"
            f"Loaded documents:\n{documents}\n\n"
            "Workflow:\n"
            "1. Read the BM25 seed chunks in the user message with get_chunk.\n"
            "2. Expand context with get_nearest_chunks, get_chunk_section, "
            "get_chunk_page, or get_chunk_para when a chunk is incomplete or relevant.\n"
            "3. Use get_toc, get_all_headings, and get_all_sect for global navigation. "
            "Use grep, bm25_search, and vector_search to find new leads.\n"
            "4. Record each important finding with record_evidence(chunk_id, note) "
            "before answering.\n"
            "5. Answer only from retrieved evidence. Cite supporting chunk IDs inline, "
            "using the exact ID returned by the tools, for example [sample:chunk_0003]. "
            "If the evidence is insufficient, say so.\n\n"
            "Be efficient: stop when the question is answered. Do not read the entire "
            "document. If multiple documents are loaded, pass the correct doc_id to "
            "document-level tools."
        )

    def _user_prompt(self, query: str, seeds: list[dict[str, Any]]) -> str:
        if seeds:
            seed_lines = "\n".join(
                f"- {seed['chunk_id']} | {seed['section_title']} | "
                f"page {seed['page_start']} | score {seed['score']} | {seed['preview']}"
                for seed in seeds
            )
        else:
            seed_lines = "(No BM25 matches. Use get_toc or grep to start.)"
        return (
            f"Question: {query}\n\n"
            "Initial BM25 seed chunks (read these first):\n"
            f"{seed_lines}"
        )

    def _finalize(self, messages: list[Any], query: str) -> str:
        if self.evidence:
            evidence_text = "\n".join(
                f"- {item['chunk_id']}: {item['note']}" for item in self.evidence
            )
        else:
            evidence_text = "(No evidence notes were recorded.)"
        messages.append(
            {
                "role": "user",
                "content": (
                    "The tool budget is exhausted. Write the final answer now using "
                    "only the retrieved evidence above. Cite chunk IDs inline. Do not "
                    f"call tools.\n\nQuestion: {query}\n\nEvidence notes:\n{evidence_text}"
                ),
            }
        )
        response = self._completion(messages, tools=None)
        message = response.choices[0].message
        answer = (_field(message, "content") or "").strip()
        return answer or "I could not produce an answer from the retrieved evidence."

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run_query(
        self,
        query: str,
        *,
        top_k: int | None = None,
        max_hops: int | None = None,
    ) -> AgentResult:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if not self.store.documents:
            raise ValueError("No documents are loaded.")

        self.evidence = []
        self.trace = []
        self.tool_outputs = []
        seed_count = max(1, int(top_k or self.top_k))
        hop_limit = max(1, int(max_hops or self.max_hops))

        seeds = self.store.bm25_search(query, top_k=seed_count)
        seed_ids = [seed["chunk_id"] for seed in seeds]
        messages: list[Any] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._user_prompt(query, seeds)},
        ]

        answer = ""
        hops = 0
        for hop in range(1, hop_limit + 1):
            hops = hop
            response = self._completion(messages, tools=AGENT_TOOL_SCHEMAS)
            message = response.choices[0].message
            messages.append(message)

            calls = _field(message, "tool_calls") or []
            if not calls:
                answer = (_field(message, "content") or "").strip()
                break

            for call in calls:
                function = _field(call, "function") or {}
                name = _field(function, "name", "")
                arguments = self._parse_arguments(
                    _field(function, "arguments", "{}")
                )

                if name == "record_evidence":
                    kwargs = self._filter_arguments(self.record_evidence, arguments)
                    try:
                        result = self.record_evidence(**kwargs)
                    except TypeError as exc:
                        result = json_error(
                            f"Invalid arguments for record_evidence: {exc}"
                        )
                else:
                    result = self.tools.dispatch(name, arguments)

                call_id = _field(call, "id") or f"call_{hop}_{len(self.trace)}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": result,
                    }
                )
                self.trace.append(
                    {
                        "hop": hop,
                        "tool": name,
                        "arguments": arguments,
                        "result_chars": len(result),
                        "result_preview": result[:300],
                    }
                )
                self.tool_outputs.append(
                    {
                        "hop": hop,
                        "tool": name,
                        "arguments": arguments,
                        "result": result,
                    }
                )

        if not answer:
            answer = self._finalize(messages, query)

        return AgentResult(
            answer=answer,
            model=self.model_name,
            seed_chunk_ids=seed_ids,
            evidence=list(self.evidence),
            trace=list(self.trace),
            hops=hops,
            tool_outputs=list(self.tool_outputs),
        )

    # ------------------------------------------------------------------
    # LLM plumbing
    # ------------------------------------------------------------------
    def _completion(
        self, messages: list[Any], tools: list[dict[str, Any]] | None = None
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.api_base:
            kwargs["api_base"] = self.api_base
        try:
            return litellm.completion(**kwargs)
        except Exception as exc:
            detail = str(exc)
            if "LLM Provider NOT provided" in detail:
                raise RuntimeError(
                    f"LiteLLM could not infer a provider from model "
                    f"{self.model_name!r}. Use a provider-prefixed model such as "
                    "'openai/deepseek-flash' for an OpenAI-compatible endpoint, "
                    "or 'deepseek/deepseek-flash' for the native DeepSeek API."
                ) from exc
            raise RuntimeError(
                f"LLM call failed for model {self.model_name!r}: {exc}. "
                "Check the model name, credentials, and that the model supports tool calling."
            ) from exc

    @staticmethod
    def _parse_arguments(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if raw in (None, ""):
            return {}
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _filter_arguments(function: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        signature = inspect.signature(function)
        allowed = {
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.kind
            in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        }
        return {key: value for key, value in arguments.items() if key in allowed}
