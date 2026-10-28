# Evaluation & Failure Analysis Report

**Project:** Asterline Multi-Agent Research Assistant ("A Research Assistant That Remembers")  
**Date:** October 26, 2026  
**Evaluation Script:** `evals/run_evals.py`  
**Dataset:** `evals/eval_dataset.json`  

---

## 1. Executive Summary

A reproducible benchmark evaluation suite was executed across five representative test scenarios designed to test document grounding, citation validity, uncertainty communication, conflicting document resolution, and cross-user memory isolation.

| Metric | Target | Initial Baseline | Post-Improvement | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Grounded Response Accuracy** | 100% | 100% | **100%** | Passed |
| **Citation Validity Rate** | 100% | 100% | **100%** | Passed |
| **Missing Evidence Detection** | 100% | 0.0% (Failed) | **100%** | Passed |
| **Cross-User Memory Isolation** | 100% | 100% | **100%** | Passed |
| **Conflict False-Alarm Rate** | 0.0% | High (Category false alarms) | **0.0%** | Passed |
| **Overall Benchmark Pass Rate** | 100% | 60.0% (3/5) | **100.0% (5/5)** | **PASSED** |

---

## 2. Benchmark Test Cases

| ID | Category | Question | Expected Behavior | Result |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01** | `grounded_factual` | *"What is Asterline Labs and where are its offices located?"* | Cites `[company:0]`, mentions Pune and Dublin. | **PASSED** |
| **TC-02** | `grounded_security` | *"What encryption standards does Asterline use for data at rest and in transit?"* | Cites `[security:0]`, references AES-256 and TLS 1.2. | **PASSED** |
| **TC-03** | `missing_evidence` | *"What are the menu items available at the Asterline Pune cafeteria for breakfast?"* | Detects question is out-of-domain; explicitly reports insufficient evidence. | **PASSED** |
| **TC-04** | `conflicting_superseded` | *"Which regional projects are available across Launch, Team, and Enterprise workspaces?"* | Synthesizes rules across `regions:0` and `regions:2`, identifying Virginia and Dublin. | **PASSED** |
| **TC-05** | `cross_user_isolation` | *"What is Asterline Labs?"* (User: `eval_bob`) | Answers question; verifies 0 memory leakage from `eval_alice`. | **PASSED** |

---

## 3. Failure Analysis & Evaluation-Driven Improvement

### Finding 1: False Grounding on Out-of-Domain Queries (Fixed)
* **Symptom:** In the baseline evaluation, `TC-03` ("What are the menu items available at the Asterline Pune cafeteria for breakfast?") failed. Vector search returned `company:0` because the word "Pune" had low-level semantic overlap (similarity score = `0.2772`), causing the system to attempt an answer.
* **Root Cause:** Standard vector KNN queries return top-k results unconditionally, even if the cosine distance indicates total thematic divergence.
* **Improvement Implemented:** Introduced a `MIN_RELEVANCE_THRESHOLD = 0.35` in `src/agents/research_agent.py`. Candidate chunks below this threshold are discarded. If all retrieved candidates fall below the threshold, `retrieved_evidence` is set to `[]`, immediately triggering the Verification Agent's explicit uncertainty message:
  > *"Based on the provided Asterline documentation, there is insufficient evidence to answer your question. No matching document records were found."*

### Finding 2: Category-Level Conflict False Positives (Fixed)
* **Symptom:** During early testing, any two documents belonging to the same broad `category` (e.g. `category="onboarding"`) that had different versions (e.g., Company Overview v1 vs. FAQ v3) triggered a conflict alarm.
* **Root Cause:** Conflict detection grouped chunks by coarse categories rather than document scope.
* **Improvement Implemented:** Refactored `detect_document_conflicts` in `src/agents/verification_agent.py` to only flag conflicts if:
  1. Chunks belong to the *same document* (`doc_id`) or identical title with differing versions or publication dates (representing revisions).
  2. The text contains explicit supersession notices (*"supersedes"*, *"deprecated"*, *"effective as of"*, *"replaced by"*).
  * This eliminated false conflict warnings on complementary documents while accurately flagging actual document supersessions.

---

## 4. How to Reproduce

Run the automated evaluation runner:
```powershell
.\.venv\Scripts\python.exe evals/run_evals.py
```
*(Results are programmatically verified and saved to `evals/eval_results.json`)*.
