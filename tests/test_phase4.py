"""Automated tests for Phase 4: Streamlit UI imports, Improved Conflict Detection, and Evals."""

import py_compile
from pathlib import Path
from src.agents.verification_agent import detect_document_conflicts
from src.agents.research_agent import research_agent_node, MIN_RELEVANCE_THRESHOLD
from src.agents.state import AgentState
from evals.run_evals import run_evaluations


def test_streamlit_app_compiles():
    """Verify that app.py has no syntax errors and compiles cleanly."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    assert app_path.exists()
    compiled = py_compile.compile(str(app_path))
    assert compiled is not None


def test_improved_conflict_detection_no_false_positives():
    """
    Verify that two documents covering DIFFERENT topics with different versions
    are NOT flagged as contradictory (eliminated false positives).
    """
    chunks_different_docs = [
        {
            "chunk_id": "company:0",
            "doc_id": "company",
            "title": "Company overview",
            "category": "onboarding",
            "version": "1.0",
            "published": "20260301",
            "text": "Asterline Labs is based in Pune and Dublin."
        },
        {
            "chunk_id": "faq:0",
            "doc_id": "faq",
            "title": "Product FAQ",
            "category": "onboarding",
            "version": "3.0",
            "published": "20260311",
            "text": "Run Explorer displays workflow attempts and status changes."
        }
    ]

    has_conflict, details = detect_document_conflicts(chunks_different_docs)
    assert has_conflict is False
    assert len(details) == 0


def test_relevance_threshold_filters_out_of_domain():
    """Verify out-of-domain questions are filtered out by minimum relevance threshold."""
    state: AgentState = {
        "question": "What are the breakfast cafeteria hours in Pune office?",
        "user_id": "eval_user",
        "retrieved_evidence": [],
        "relevant_memories": [],
        "verification_findings": {},
        "final_answer": "",
        "execution_steps": [],
    }

    result = research_agent_node(state)
    # Out of domain query should have 0 chunks passing threshold
    assert len(result["retrieved_evidence"]) == 0
    assert any("threshold" in step for step in result["execution_steps"])


def test_eval_suite_runs_successfully():
    """Verify that the evaluation suite runs and passes 100% of benchmark test cases."""
    summary = run_evaluations()
    assert summary["total_cases"] == 5
    assert summary["passed"] == 5
    assert summary["pass_rate_percent"] == 100.0
