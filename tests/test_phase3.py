"""Automated tests for Phase 3: LangGraph Multi-Agent Orchestration.

Tests:
1. Research Agent retrieval and evidence preservation.
2. Memory Agent cross-user isolation within AgentState.
3. Verification Agent missing evidence handling.
4. Verification Agent conflict detection (versions/dates).
5. Citation validation (valid vs. fabricated citations).
6. Personal memory boundary preservation (memory != company evidence).
7. End-to-end LangGraph StateGraph execution with a mocked LLM.
8. End-to-end LangGraph StateGraph execution with deterministic fallback (no API key required).
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

from src.agents.state import AgentState
from src.agents.research_agent import research_agent_node
from src.agents.memory_agent import memory_agent_node
from src.agents.verification_agent import (
    verification_agent_node,
    detect_document_conflicts,
    validate_citations,
)
from src.agents.graph import create_research_assistant_graph, run_assistant
from src.memory_store import SQLiteMemoryStore


@pytest.fixture
def isolated_memory_store(tmp_path: Path) -> SQLiteMemoryStore:
    """Fixture for an isolated SQLite test database."""
    return SQLiteMemoryStore(db_path=tmp_path / "test_agents_memory.db")


def test_research_agent_retrieval():
    """Verify Research Agent retrieves chunks from ChromaDB and preserves metadata."""
    state: AgentState = {
        "question": "What is Asterline Labs?",
        "user_id": "test_user",
        "retrieved_evidence": [],
        "relevant_memories": [],
        "verification_findings": {},
        "final_answer": "",
        "execution_steps": [],
    }

    result = research_agent_node(state)
    assert "retrieved_evidence" in result
    evidence = result["retrieved_evidence"]
    assert len(evidence) > 0

    top = evidence[0]
    assert "chunk_id" in top
    assert "title" in top
    assert "text" in top
    assert any("ResearchAgent" in step for step in result["execution_steps"])


def test_memory_agent_isolation(isolated_memory_store: SQLiteMemoryStore):
    """Verify Memory Agent only injects memories for the active user_id."""
    isolated_memory_store.add_memory("alice", "Alice prefers concise markdown tables.")
    isolated_memory_store.add_memory("bob", "Bob works on Windows exclusively.")

    # Query as Alice
    state_alice: AgentState = {
        "question": "How to format output?",
        "user_id": "alice",
        "retrieved_evidence": [],
        "relevant_memories": [],
        "verification_findings": {},
        "final_answer": "",
        "execution_steps": [],
    }
    res_alice = memory_agent_node(state_alice, memory_store=isolated_memory_store)
    assert len(res_alice["relevant_memories"]) == 1
    assert "Alice prefers" in res_alice["relevant_memories"][0]
    assert "Bob" not in res_alice["relevant_memories"][0]

    # Query as Bob
    state_bob: AgentState = {
        "question": "OS environment context",
        "user_id": "bob",
        "retrieved_evidence": [],
        "relevant_memories": [],
        "verification_findings": {},
        "final_answer": "",
        "execution_steps": [],
    }
    res_bob = memory_agent_node(state_bob, memory_store=isolated_memory_store)
    assert len(res_bob["relevant_memories"]) == 1
    assert "Bob works on" in res_bob["relevant_memories"][0]
    assert "Alice" not in res_bob["relevant_memories"][0]


def test_missing_evidence_handling():
    """Verify Verification Agent communicates uncertainty when 0 chunks are retrieved."""
    state: AgentState = {
        "question": "What is the secret recipe for cafeteria lunch?",
        "user_id": "alice",
        "retrieved_evidence": [],  # Empty retrieval
        "relevant_memories": [],
        "verification_findings": {},
        "final_answer": "",
        "execution_steps": [],
    }

    result = verification_agent_node(state)
    assert result["verification_findings"]["missing_evidence"] is True
    assert "insufficient evidence" in result["final_answer"].lower()


def test_conflict_detection_logic():
    """Verify conflict detector identifies conflicting versions, dates, or supersessions."""
    mock_chunks = [
        {
            "chunk_id": "pricing:0",
            "doc_id": "pricing",
            "category": "commercial",
            "version": "1.0",
            "published": "20240101",
            "text": "Asterline Team edition costs $15 per month."
        },
        {
            "chunk_id": "pricing:1",
            "doc_id": "pricing",
            "category": "commercial",
            "version": "2.0",
            "published": "20260301",
            "text": "This document supersedes all prior pricing: Asterline Team edition is $25 per month."
        }
    ]

    has_conflict, details = detect_document_conflicts(mock_chunks)
    assert has_conflict is True
    assert any("supersedes" in d.lower() or "versions" in d.lower() for d in details)


def test_citation_validation():
    """Verify citations referencing valid chunks pass and fabricated ones fail."""
    mock_chunks = [
        {"chunk_id": "company:0", "doc_id": "company", "text": "Asterline overview."}
    ]

    valid_text = "Asterline Labs is based in Pune [company:0]."
    valid, invalid = validate_citations(valid_text, mock_chunks)
    assert valid == ["company:0"]
    assert invalid == []

    fabricated_text = "Asterline was founded in 1990 [fake_doc:99]."
    valid_fab, invalid_fab = validate_citations(fabricated_text, mock_chunks)
    assert valid_fab == []
    assert invalid_fab == ["fake_doc:99"]


def test_end_to_end_graph_with_mocked_llm(isolated_memory_store: SQLiteMemoryStore):
    """Verify full 3-agent LangGraph workflow execution with a mocked LLM."""
    isolated_memory_store.add_memory("charlie", "I am a DevOps engineer.")

    # Create Mock LLM that returns a cited response
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="Asterline Labs develops workflow monitoring tools with offices in Pune and Dublin [company:0]."
    )

    app = create_research_assistant_graph(llm=mock_llm, memory_store=isolated_memory_store)

    initial_state: AgentState = {
        "question": "What is Asterline Labs?",
        "user_id": "charlie",
        "retrieved_evidence": [],
        "relevant_memories": [],
        "verification_findings": {},
        "final_answer": "",
        "execution_steps": [],
    }

    final_state = app.invoke(initial_state)

    # 1. Research Agent ran
    assert len(final_state["retrieved_evidence"]) > 0
    # 2. Memory Agent ran and isolated Charlie's memory
    assert len(final_state["relevant_memories"]) == 1
    assert "DevOps engineer" in final_state["relevant_memories"][0]
    # 3. Verification Agent verified citations and produced answer
    assert "Pune and Dublin" in final_state["final_answer"]
    assert "company:0" in final_state["verification_findings"]["valid_citations"]
    # 4. Audit trail shows all 3 agents
    trace = " ".join(final_state["execution_steps"])
    assert "MemoryAgent" in trace
    assert "ResearchAgent" in trace
    assert "VerificationAgent" in trace


def test_end_to_end_graph_deterministic_fallback():
    """Verify full graph execution completes without API keys using deterministic grounding."""
    res = run_assistant(
        question="Where are Asterline Labs offices located?",
        user_id="anonymous_user",
    )

    assert res["final_answer"]
    assert len(res["retrieved_evidence"]) > 0
    assert "Sources Cited:" in res["final_answer"]
    assert res["verification_findings"]["is_grounded"] is True
