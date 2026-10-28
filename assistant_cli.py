"""Command-line interface to interact with the LangGraph Multi-Agent Assistant (Phase 3).

Usage:
    python assistant_cli.py --user alice "What is Asterline Labs?"
    python assistant_cli.py --user bob "Where are Asterline offices and what regions exist?"
"""

import argparse
import sys
from src.agents.graph import run_assistant


def main():
    parser = argparse.ArgumentParser(description="Asterline Multi-Agent Assistant CLI (Phase 3)")
    parser.add_argument("question", nargs="*", help="Question to submit to the multi-agent assistant")
    parser.add_argument("--user", default="user_default", help="User ID (enforces memory isolation)")

    args = parser.parse_args()
    question = " ".join(args.question).strip() if args.question else ""

    if not question:
        question = "What is Asterline Labs and where are its offices located?"
        print(f"[CLI] No question supplied. Using default test question: '{question}'\n")

    print("=" * 80)
    print(f" SUBMITTING TO MULTI-AGENT ORCHESTRATOR")
    print(f" User:     {args.user}")
    print(f" Question: {question}")
    print("=" * 80 + "\n")

    result = run_assistant(question=question, user_id=args.user)

    # 1. Retrieved User Context
    memories = result.get("relevant_memories", [])
    print(f"--- 1. ACTIVE USER MEMORIES ({len(memories)}) ---")
    if memories:
        for m in memories:
            print(f"  {m}")
    else:
        print("  (No personal memories found for this user)")
    print()

    # 2. Retrieved Evidence Chunks
    evidence = result.get("retrieved_evidence", [])
    print(f"--- 2. RETRIEVED DOCUMENT EVIDENCE ({len(evidence)} chunks) ---")
    for rank, chunk in enumerate(evidence, start=1):
        print(f"  [{rank}] {chunk.get('chunk_id')} | {chunk.get('title')} (v{chunk.get('version')}, {chunk.get('published')})")
        snippet = chunk.get("text", "")[:120].replace("\n", " ") + "..."
        print(f"      {snippet}")
    print()

    # 3. Verification & Conflict Findings
    findings = result.get("verification_findings", {})
    print("--- 3. VERIFICATION & CONFLICT ANALYSIS ---")
    print(f"  Grounded:            {findings.get('is_grounded', False)}")
    print(f"  Missing Evidence:    {findings.get('missing_evidence', False)}")
    print(f"  Conflicts Detected:  {findings.get('conflicts_detected', False)}")
    if findings.get("conflict_details"):
        print("  Discrepancy Details:")
        for cd in findings["conflict_details"]:
            print(f"    - {cd}")
    if findings.get("valid_citations"):
        print(f"  Valid Citations:     {', '.join(findings['valid_citations'])}")
    print()

    # 4. Final Answer
    print("=" * 80)
    print(" 4. FINAL GROUNDED RESPONSE")
    print("=" * 80)
    print(result.get("final_answer", ""))
    print("=" * 80 + "\n")

    # 5. Agent Execution Steps (Audit Trail)
    print("--- 5. AGENT EXECUTION TRACE ---")
    for step in result.get("execution_steps", []):
        print(f"  -> {step}")
    print("--------------------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
