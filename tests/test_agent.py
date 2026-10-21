from __future__ import annotations

import json
from types import SimpleNamespace

import src.agent as agent_module
from src.agent import TopoRAGAgent
from src.indexer import build_document_map
from src.store import ChunkStore

DOC = """# Alpha

Apples are red.

## Beta

Bananas are yellow.
"""


def _response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(name, arguments, call_id):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _agent():
    document = build_document_map(DOC, doc_id="a", chunk_size=10)
    store = ChunkStore([document])
    seed_id = store.bm25_search("apples", top_k=1)[0]["chunk_id"]
    return TopoRAGAgent(store, model_name="fake-model", max_hops=4), seed_id


def test_agent_traverses_and_records_evidence(monkeypatch):
    agent, seed_id = _agent()
    responses = [
        _response(tool_calls=[_tool_call("get_chunk", {"chunk_id": seed_id}, "call_1")]),
        _response(
            tool_calls=[
                _tool_call(
                    "record_evidence",
                    {"chunk_id": seed_id, "note": "Apples are red."},
                    "call_2",
                )
            ]
        ),
        _response(content="Apples are red [a:chunk_0000]."),
    ]
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    monkeypatch.setattr(agent_module.litellm, "completion", fake_completion)

    result = agent.run_query("What color are apples?")

    assert result.answer == "Apples are red [a:chunk_0000]."
    assert seed_id in result.seed_chunk_ids
    assert result.evidence == [{"chunk_id": seed_id, "note": "Apples are red."}]
    assert [entry["tool"] for entry in result.trace] == [
        "get_chunk",
        "record_evidence",
    ]
    assert [entry["tool"] for entry in result.tool_outputs] == [
        "get_chunk",
        "record_evidence",
    ]
    assert '"ok": true' in result.tool_outputs[0]["result"]
    assert "tool_outputs" not in result.to_dict()
    full_trace = result.to_dict(include_tool_outputs=True)
    assert len(full_trace["tool_outputs"]) == 2
    assert result.hops == 3
    assert len(calls) == 3
    assert calls[0]["tools"]


def test_agent_finalizes_when_hop_budget_is_exhausted(monkeypatch):
    agent, seed_id = _agent()
    agent.max_hops = 1
    responses = [
        _response(tool_calls=[_tool_call("get_chunk", {"chunk_id": seed_id}, "call_1")]),
        _response(content="Final answer from evidence."),
    ]
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    monkeypatch.setattr(agent_module.litellm, "completion", fake_completion)

    result = agent.run_query("What color are apples?")

    assert result.answer == "Final answer from evidence."
    assert result.hops == 1
    assert len(calls) == 2
    assert "tools" not in calls[1]


def test_record_evidence_rejects_unknown_chunk():
    agent, _ = _agent()
    payload = json.loads(agent.record_evidence("a:chunk_9999", "nope"))
    assert payload["ok"] is False
    assert "Unknown chunk" in payload["error"]


def test_invalid_tool_call_does_not_crash_loop(monkeypatch):
    agent, _ = _agent()
    responses = [
        _response(tool_calls=[_tool_call("not_a_tool", {}, "call_1")]),
        _response(content="Recovered."),
    ]
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    monkeypatch.setattr(agent_module.litellm, "completion", fake_completion)

    result = agent.run_query("What color are apples?")

    assert result.answer == "Recovered."
    assert result.trace[0]["tool"] == "not_a_tool"
    assert '"ok": false' in result.trace[0]["result_preview"]
