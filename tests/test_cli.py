from __future__ import annotations

import json

import main as cli
from src.agent import AgentResult

DOC = """# Apples

Apples are red.

## Bananas

Bananas are yellow.
"""


def _write_doc(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(DOC, encoding="utf-8")
    return path


def test_chunk_prints_full_text_and_unknown_id_fails(tmp_path, capsys):
    path = _write_doc(tmp_path)

    assert cli.main(["--chunk", "doc:chunk_0000", "-f", str(path)]) == 0
    output = capsys.readouterr().out
    assert "[doc:chunk_0000]" in output
    assert "Apples are red." in output

    assert cli.main(["--chunk", "doc:chunk_9999", "-f", str(path)]) == 1
    captured = capsys.readouterr()
    assert "Unknown chunk ID" in captured.err


def test_retrieval_compares_bm25_and_vector_fallback(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("TOPO_RAG_EMBEDDING_MODEL", "")
    path = _write_doc(tmp_path)

    assert (
        cli.main(
            [
                "--retrieval",
                "apples",
                "--top-k",
                "1",
                "--json",
                "-f",
                str(path),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["bm25"][0]["chunk_id"] == "doc:chunk_0000"
    assert "Apples are red." in payload["bm25"][0]["text"]
    assert "preview" not in payload["bm25"][0]
    assert payload["vector"]["mode"] == "bm25_fallback"


def test_show_evidence_and_save_trace(tmp_path, capsys, monkeypatch):
    path = _write_doc(tmp_path)
    trace_path = tmp_path / "trace.json"

    def fake_run(self, query, **kwargs):
        return AgentResult(
            answer="Apples are red [doc:chunk_0000].",
            model="fake-model",
            seed_chunk_ids=["doc:chunk_0000"],
            evidence=[
                {
                    "chunk_id": "doc:chunk_0000",
                    "note": "The document says apples are red.",
                }
            ],
            trace=[
                {
                    "hop": 1,
                    "tool": "get_chunk",
                    "arguments": {"chunk_id": "doc:chunk_0000"},
                    "result_chars": 10,
                    "result_preview": "full result is in tool_outputs",
                }
            ],
            hops=1,
            tool_outputs=[
                {
                    "hop": 1,
                    "tool": "get_chunk",
                    "arguments": {"chunk_id": "doc:chunk_0000"},
                    "result": "full result",
                }
            ],
        )

    monkeypatch.setattr(cli.TopoRAGAgent, "run_query", fake_run)

    assert (
        cli.main(
            [
                "What color are apples?",
                "--show-evidence",
                "--save-trace",
                str(trace_path),
                "-f",
                str(path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Apples are red [doc:chunk_0000]." in output
    assert "--- Evidence chunks ---" in output
    assert "The document says apples are red." in output

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["tool_outputs"][0]["result"] == "full result"
    assert trace["seed_chunks"][0]["chunk_id"] == "doc:chunk_0000"
    assert trace["evidence_chunks"][0]["chunk_id"] == "doc:chunk_0000"
    assert "Apples are red." in trace["evidence_chunks"][0]["text"]
