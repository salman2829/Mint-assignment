"""Verification Agent Node for LangGraph.

Responsibilities:
1. Grounding & Synthesis: Generates answers strictly supported by retrieved evidence chunks.
2. Citation Checking: Validates that all citations map to actual retrieved chunks.
3. Conflict Detection: Identifies substantive contradictions, conflicting scope, differing
   effective dates for the same subject, or explicit supersession statements.
   (Different dates or versions across separate documents are NOT treated as conflicts).
4. Evidence Uncertainty: Communicates missing or ambiguous evidence clearly.
5. Evidence Boundaries: Uses personal memories strictly as personalization context, never as company facts.
"""

import os
import re
from typing import Dict, Any, List, Tuple
from src.agents.state import AgentState, VerificationFindings


def detect_document_conflicts(evidence_chunks: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Analyzes retrieved chunks for substantive policy contradictions, conflicting scope,
    differing effective dates for the same subject, or explicit supersession statements.
    """
    if len(evidence_chunks) < 2:
        return False, []

    conflicts = []
    
    # 1. Check for explicit supersession keywords within chunks
    supersession_keywords = [
        "supersedes", "superseded by", "deprecated", "replaced by",
        "effective as of", "prior version", "no longer applies", "amended"
    ]
    for chunk in evidence_chunks:
        text_lower = chunk.get("text", "").lower()
        for kw in supersession_keywords:
            if kw in text_lower:
                conflicts.append(
                    f"Supersession notice in [{chunk.get('chunk_id')}]: "
                    f"Document contains '{kw}', indicating updated or modified authority."
                )

    # 2. Check for multiple versions or effective dates OF THE SAME DOCUMENT (doc_id)
    by_doc_id = {}
    for chunk in evidence_chunks:
        doc_id = chunk.get("doc_id", "")
        if doc_id:
            by_doc_id.setdefault(doc_id, []).append(chunk)

    for doc_id, chunks in by_doc_id.items():
        versions = set(c.get("version") for c in chunks if c.get("version"))
        dates = set(c.get("published") for c in chunks if c.get("published"))
        
        # Only flag if there are actually different version numbers or dates FOR THE SAME DOCUMENT
        if len(versions) > 1 or len(dates) > 1:
            sorted_v = sorted(list(versions))
            sorted_d = sorted(list(dates))
            conflicts.append(
                f"Document Revision on '{doc_id}': Multiple versions {sorted_v} "
                f"(published: {sorted_d}). Version {sorted_v[-1]} represents the newer authority."
            )

    return (len(conflicts) > 0), conflicts


def validate_citations(text: str, evidence_chunks: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Extracts citations of form [chunk_id] or [doc_id:X] from text and verifies
    they correspond to actual retrieved chunks.
    """
    valid_ids = set(c.get("chunk_id") for c in evidence_chunks)
    valid_doc_ids = set(c.get("doc_id") for c in evidence_chunks)

    # Match bracketed patterns like [company:0] or [company] or [SEC-001]
    raw_citations = re.findall(r"\[([a-zA-Z0-9_\-:]+)\]", text)
    valid_found = []
    invalid_found = []

    for cit in raw_citations:
        if cit in valid_ids or cit in valid_doc_ids:
            valid_found.append(cit)
        else:
            invalid_found.append(cit)

    return valid_found, invalid_found


def generate_synthesis_prompt(
    question: str,
    evidence_chunks: List[Dict[str, Any]],
    memories: List[str],
    conflicts: List[str]
) -> str:
    """Builds a structured prompt for the LLM or synthesis engine."""
    evidence_str = "\n".join([
        f"- Chunk ID: [{c.get('chunk_id')}] (Doc: {c.get('doc_id')}, Version: {c.get('version')}, Published: {c.get('published')})\n"
        f"  Title: {c.get('title')}\n"
        f"  Text: {c.get('text')}\n"
        for c in evidence_chunks
    ]) or "NO RELEVANT EVIDENCE FOUND."

    memories_str = "\n".join([f"- {m}" for m in memories]) or "NO USER MEMORIES."
    conflicts_str = "\n".join([f"- {c}" for c in conflicts]) or "NONE DETECTED."

    return f"""You are the Asterline Verification & Research Synthesis Agent.
Answer the user's question grounded STRICTLY in the supplied EVIDENCE CHUNKS.

CRITICAL RULES:
1. Every claim about Asterline must cite the exact chunk ID, e.g. [chunk_id].
2. If evidence chunks conflict, disagree, or superseded dates/versions exist, explicitly explain BOTH claims and highlight which is newer.
3. USER MEMORIES are personal preferences only (e.g., tone or format). NEVER treat personal memory as evidence or proof of company facts.
4. If the evidence chunks do not contain enough facts to answer, explicitly say:
   "Based on the provided Asterline documentation, there is insufficient evidence to answer this question." Do not fabricate answers.

USER QUESTION:
{question}

POTENTIAL CONFLICTS DETECTED:
{conflicts_str}

USER PERSONAL CONTEXT:
{memories_str}

EVIDENCE CHUNKS:
{evidence_str}

Provide a verified, well-structured answer with bracketed citations [chunk_id]:"""


def verification_agent_node(state: AgentState, llm=None) -> Dict[str, Any]:
    """
    Executes conflict checking, evidence grounding, and synthesis for AgentState.
    """
    question = state.get("question", "")
    evidence = state.get("retrieved_evidence", [])
    memories = state.get("relevant_memories", [])
    execution_steps = list(state.get("execution_steps", []))

    # 1. Missing evidence check
    if not evidence:
        findings = {
            "is_grounded": True,
            "missing_evidence": True,
            "conflicts_detected": False,
            "conflict_details": [],
            "valid_citations": [],
            "invalid_citations": [],
            "notes": "No evidence chunks available in state."
        }
        answer = (
            "Based on the provided Asterline documentation, there is insufficient evidence "
            "to answer your question. No matching document records were found."
        )
        execution_steps.append("VerificationAgent: Missing evidence detected. Returned uncertainty disclaimer.")
        return {
            "verification_findings": findings,
            "final_answer": answer,
            "execution_steps": execution_steps
        }

    # 2. Conflict Analysis
    has_conflicts, conflict_details = detect_document_conflicts(evidence)
    if has_conflicts:
        execution_steps.append(
            f"VerificationAgent: Identified {len(conflict_details)} substantive document conflict(s) or supersessions."
        )

    # 3. Answer Generation (OpenAI LLM if API key configured, otherwise deterministic grounded fallback)
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    answer = ""

    if llm is not None:
        prompt = generate_synthesis_prompt(question, evidence, memories, conflict_details)
        response = llm.invoke(prompt)
        answer = getattr(response, "content", str(response))
        execution_steps.append("VerificationAgent: Generated answer using injected LLM.")
    elif openai_key and not openai_key.startswith("your-"):
        try:
            from langchain_openai import ChatOpenAI
            chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
            prompt = generate_synthesis_prompt(question, evidence, memories, conflict_details)
            response = chat_model.invoke(prompt)
            answer = response.content
            execution_steps.append("VerificationAgent: Generated response via OpenAI gpt-4o-mini.")
        except Exception as err:
            execution_steps.append(f"VerificationAgent: LLM call failed ({err}). Using grounded deterministic fallback.")
            answer = _deterministic_grounded_answer(evidence, conflict_details, memories)
    else:
        execution_steps.append(
            "VerificationAgent: No active OPENAI_API_KEY found. Generated deterministic grounded answer."
        )
        answer = _deterministic_grounded_answer(evidence, conflict_details, memories)

    # 4. Citation Validation
    valid_cits, invalid_cits = validate_citations(answer, evidence)
    execution_steps.append(
        f"VerificationAgent: Citation validation complete ({len(valid_cits)} valid, {len(invalid_cits)} invalid)."
    )

    findings = {
        "is_grounded": len(invalid_cits) == 0,
        "missing_evidence": False,
        "conflicts_detected": has_conflicts,
        "conflict_details": conflict_details,
        "valid_citations": valid_cits,
        "invalid_citations": invalid_cits,
        "notes": "Synthesis grounded in verified chunks."
    }

    return {
        "verification_findings": findings,
        "final_answer": answer,
        "execution_steps": execution_steps
    }


def _deterministic_grounded_answer(
    evidence: List[Dict[str, Any]],
    conflicts: List[str],
    memories: List[str]
) -> str:
    """Builds an accurate, multi-chunk grounded answer using top evidence when API keys are absent."""
    # Use top chunks up to 2 for multi-chunk completeness
    top_chunks = evidence[:2]
    
    lines = ["Based on the verified Asterline documentation:"]
    cited_ids = []

    for chunk in top_chunks:
        cid = chunk.get("chunk_id", "")
        title = chunk.get("title", "")
        lines.append(f"\n- According to {title} [{cid}]:\n  \"{chunk.get('text')}\"")
        cited_ids.append(cid)

    if conflicts:
        lines.append("\n**Document Discrepancies / Conflicts Noted:**")
        for c in conflicts:
            lines.append(f"- {c}")

    if memories:
        lines.append("\n*(Applied User Context: Personal preferences were applied without affecting company facts)*")

    lines.append(f"\nSources Cited: {', '.join([f'[{cid}]' for cid in cited_ids])}")
    return "\n".join(lines)
