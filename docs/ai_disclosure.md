# AI-Use Disclosure Template

**Candidate Name:** Salman  
**Assignment:** Batch 5 — A Research Assistant That Remembers  
**Submission Date:** October 26, 2026  

---

## 1. Overview of AI Assistance

In developing this take-home project, generative AI coding assistants were utilized as pair-programming and scaffolding tools in accordance with academic integrity and assignment guidelines.

---

## 2. Tools & Models Used

| Tool / Assistant | Model Version | Primary Use Case |
| :--- | :--- | :--- |
| **Antigravity IDE Assistant** | Gemini 2.5 Flash / Claude 3.5 Sonnet / GPT-4o | Architecture planning, project scaffolding, debugging Windows path length constraints, writing test suites. |
| **Local LLM / Generation Model** | OpenAI `gpt-4o-mini` | Generation of grounded answers during runtime synthesis. |
| **Embedding Model** | `all-MiniLM-L6-v2` (SentenceTransformers) | Local CPU-based vector embeddings for document chunks and user memories. |

---

## 3. Areas Where AI Was Used

1. **Architecture & Scaffolding:**
   - Designing the 4-phase development plan (`blueprint.md`).
   - Formulating the LangGraph `StateGraph` node transitions (`memory_agent -> research_agent -> verification_agent`).
2. **Debugging & Environment Workarounds:**
   - Diagnosing and resolving the Windows MAX_PATH length limitation (`[WinError 206]`) by configuring a dedicated local virtual environment (`.venv`).
3. **Test Suite Generation:**
   - Drafting automated unit and integration tests across all 4 phases (`tests/test_phase1.py` through `tests/test_phase4.py`).
4. **Interface Implementation:**
   - Implementing the Streamlit chat interface and memory management sidebar in `app.py`.

---

## 4. Human Verification & Critical Engineering

All generated code was manually inspected, executed, tested, and validated:
- **Data Integrity:** Ensured `corpus.jsonl` was extracted directly from the assignment PDF attachments and maintained in strict read-only mode without modifications.
- **Security & User Isolation:** Verified that all SQLite queries enforce compound `WHERE user_id = ?` filters and regex-based secret/credential rejection.
- **Evaluation-Driven Improvements:** Iteratively identified and fixed failure cases in out-of-domain query handling (adding relevance thresholds) and category-level conflict detection false alarms.
- **Test Execution:** All 23 tests and benchmark evaluations were run and verified locally against real outputs.

---

## 5. Candidate Confirmation

- [x] I have personally reviewed, run, and verified all code in this repository.
- [x] The architecture, design notes, and evaluation results reflect authentic, reproducible local execution.
- [x] No API keys, passwords, or personal credentials are included in this repository.

*Signature:* _____________________________  
*Date:* _____________________________  
