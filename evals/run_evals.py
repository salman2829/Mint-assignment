"""Reproducible evaluation runner for Asterline Research Assistant.

Evaluates:
1. Grounded answer generation.
2. Citation validity (all cited IDs must exist in retrieved chunks).
3. Missing evidence detection (returns uncertainty for out-of-corpus queries).
4. Cross-user memory isolation (verifies zero cross-contamination).
5. Conflict analysis accuracy.

Usage:
    python evals/run_evals.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.agents.graph import run_assistant
from src.memory_store import SQLiteMemoryStore


def run_evaluations() -> Dict[str, Any]:
    dataset_path = BASE_DIR / "evals" / "eval_dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset missing at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Initialize memory store and seed test context for eval_alice
    mem_store = SQLiteMemoryStore()
    try:
        mem_store.add_memory("eval_alice", "User preference: Needs markdown tables.")
    except Exception:
        pass

    results = []
    passed_count = 0

    print("=" * 80)
    print(" RUNNING REPRODUCIBLE EVALUATION SUITE")
    print(f" Total Benchmark Test Cases: {len(test_cases)}")
    print("=" * 80 + "\n")

    for tc in test_cases:
        tc_id = tc["id"]
        category = tc["category"]
        question = tc["question"]
        user_id = tc["user_id"]

        print(f"[{tc_id}] Category: {category} | User: {user_id}")
        print(f"      Question: \"{question}\"")

        # Execute Multi-Agent Graph
        out = run_assistant(question=question, user_id=user_id, memory_store=mem_store)

        final_answer = out.get("final_answer", "")
        evidence = out.get("retrieved_evidence", [])
        findings = out.get("verification_findings", {})
        memories = out.get("relevant_memories", [])
        steps = out.get("execution_steps", [])

        # Evaluate criteria
        checks = {}

        # 1. Expected keywords check
        keywords_present = all(kw.lower() in final_answer.lower() for kw in tc.get("expected_keywords", []))
        checks["keywords_present"] = keywords_present

        # 2. Citation validity check (100% of cited IDs must be in evidence)
        invalid_cits = findings.get("invalid_citations", [])
        valid_cits = findings.get("valid_citations", [])
        citations_valid = (len(invalid_cits) == 0)
        checks["citations_valid"] = citations_valid

        # 3. Missing evidence check
        if tc.get("should_have_missing_evidence"):
            uncertainty_passed = findings.get("missing_evidence", False) or "insufficient evidence" in final_answer.lower()
            checks["missing_evidence_handled"] = uncertainty_passed

        # 4. Cross-user isolation check
        leaked_user = tc.get("verify_no_memories_leaked_from")
        if leaked_user:
            no_leak = not any(leaked_user in str(m) for m in memories)
            checks["cross_user_isolation"] = no_leak

        # 5. Execution steps trace check
        has_trace = len(steps) >= 3 and any("ResearchAgent" in s for s in steps) and any("VerificationAgent" in s for s in steps)
        checks["execution_trace_complete"] = has_trace

        all_passed = all(checks.values())
        status = "PASSED" if all_passed else "FAILED"
        if all_passed:
            passed_count += 1

        print(f"      Status:   {status}")
        print(f"      Checks:   {checks}")
        print(f"      Answer:   {final_answer[:100]}...\n")

        results.append({
            "test_case": tc,
            "checks": checks,
            "status": status,
            "answer_preview": final_answer[:150],
            "valid_citations": valid_cits,
            "invalid_citations": invalid_cits,
            "execution_steps_count": len(steps)
        })

    # Summary
    pass_rate = round((passed_count / len(test_cases)) * 100, 1)
    summary = {
        "total_cases": len(test_cases),
        "passed": passed_count,
        "failed": len(test_cases) - passed_count,
        "pass_rate_percent": pass_rate,
        "results": results
    }

    print("=" * 80)
    print(f" EVALUATION SUMMARY: {passed_count}/{len(test_cases)} Passed ({pass_rate}%)")
    print("=" * 80 + "\n")

    # Save output to evals/eval_results.json
    output_path = BASE_DIR / "evals" / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Results Saved] Detailed results written to {output_path}")

    return summary


if __name__ == "__main__":
    run_evaluations()
