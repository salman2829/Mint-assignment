"""Shared TypedDict state schema for LangGraph multi-agent orchestration."""

from typing import TypedDict, List, Dict, Any, Optional


class VerificationFindings(TypedDict, total=False):
    """Detailed findings from the Verification Agent."""
    is_grounded: bool
    missing_evidence: bool
    conflicts_detected: bool
    conflict_details: List[str]
    valid_citations: List[str]
    invalid_citations: List[str]
    notes: str


class AgentState(TypedDict):
    """
    Shared orchestration state passed between agents in the LangGraph workflow.
    
    Fields:
        question: The user's research query.
        user_id: The explicit identifier of the active user.
        retrieved_evidence: Document chunks returned from local ChromaDB.
        relevant_memories: Personal preferences retrieved from SQLite for user_id.
        verification_findings: Evidence grounding and conflict analysis report.
        final_answer: The verified, cited response for the user.
        execution_steps: Chronological audit log of agent actions.
    """
    question: str
    user_id: str
    retrieved_evidence: List[Dict[str, Any]]
    relevant_memories: List[str]
    verification_findings: Dict[str, Any]
    final_answer: str
    execution_steps: List[str]
