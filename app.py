r"""Asterline Research Assistant - Streamlit Chat Interface (Phase 4).

Single-command entry point:
    .\.venv\Scripts\python.exe -m streamlit run app.py

Features:
1. User Identity Switcher (Simulates separate users with strict memory isolation).
2. Chat interface with conversational history per user.
3. Verification Agent findings: Grounding status, conflict detection cards, and citations.
4. Expandable Document Evidence viewer showing chunk metadata, versions, and source URLs.
5. Expandable Agent Execution Audit Log showing chronological actions.
6. Memory Management Sidebar: List, Add, Correct, and Forget memories in real-time.
7. Helpful status alerts for LLM API keys and LangSmith observability.
"""

import os
import streamlit as st
from datetime import datetime, timezone
from dotenv import load_dotenv

# Ensure environment variables are loaded/updated from .env
load_dotenv(override=True)

from src.memory_store import SQLiteMemoryStore, validate_no_secrets
from src.agents.graph import run_assistant
from src.config import SQLITE_DB_PATH


# Configure Streamlit page
st.set_page_config(
    page_title="Asterline Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize persistent memory store
memory_store = SQLiteMemoryStore()

# -----------------------------------------------------------------------------
# Session State Initialization (User-Isolated Chat History)
# -----------------------------------------------------------------------------
if "current_user" not in st.session_state:
    st.session_state["current_user"] = "alice"

if "conversations" not in st.session_state:
    # Isolated dictionary: {user_id: [{"role": ..., "content": ..., "metadata": ...}]}
    st.session_state["conversations"] = {
        "alice": [],
        "bob": [],
        "charlie": []
    }

# -----------------------------------------------------------------------------
# Sidebar: User Switching & Persistent Memory Management
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("👤 User Identity")
    st.caption("Switch users to verify strict cross-user memory & session isolation.")

    user_options = ["alice", "bob", "charlie", "custom"]
    selected_option = st.selectbox(
        "Active User Profile:",
        options=user_options,
        index=0 if st.session_state["current_user"] == "alice" else 1
    )

    if selected_option == "custom":
        active_user = st.text_input("Enter Custom User ID:", value="user_guest").strip().lower()
    else:
        active_user = selected_option

    # Update active user in session state
    st.session_state["current_user"] = active_user
    if active_user not in st.session_state["conversations"]:
        st.session_state["conversations"][active_user] = []

    st.divider()

    # -------------------------------------------------------------------------
    # Active Memory Inspector & Editor
    # -------------------------------------------------------------------------
    st.subheader(f"🧠 Memory Bank: `{active_user}`")
    st.caption("Active context stored in SQLite. Only accessible by this user.")

    # Fetch active memories for this user
    active_mems = memory_store.list_memories(user_id=active_user, active_only=True)

    if active_mems:
        for mem in active_mems:
            with st.expander(f"📌 Memory #{mem.id}", expanded=False):
                st.write(mem.memory_text)
                st.caption(f"Updated: {mem.updated_at[:19]} UTC")
                
                # Edit / Correct Memory
                new_text = st.text_input(f"Edit text (#{mem.id}):", value=mem.memory_text, key=f"edit_mem_{mem.id}")
                col_up, col_del = st.columns([1, 1])
                with col_up:
                    if st.button("Save", key=f"btn_save_{mem.id}"):
                        try:
                            memory_store.update_memory(active_user, mem.id, new_text)
                            st.success("Memory updated!")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                with col_del:
                    if st.button("Forget", key=f"btn_del_{mem.id}"):
                        memory_store.delete_memory(active_user, mem.id, hard=False)
                        st.warning("Memory forgotten!")
                        st.rerun()
    else:
        st.info("No active memories stored for this user.")

    st.markdown("#### ➕ Add New Memory")
    new_mem_input = st.text_area(
        "Remember a preference or context:",
        placeholder="e.g. I prefer concise markdown summaries.",
        height=80,
        key="new_memory_textarea"
    )
    if st.button("Store Memory", type="primary", use_container_width=True):
        if new_mem_input.strip():
            try:
                rec = memory_store.add_memory(user_id=active_user, memory_text=new_mem_input)
                st.success(f"Stored Memory #{rec.id}!")
                st.rerun()
            except ValueError as ve:
                st.error(f"Security Alert: {ve}")
            except Exception as ex:
                st.error(f"Error: {ex}")
        else:
            st.warning("Please enter memory text.")

    st.divider()

    # -------------------------------------------------------------------------
    # Environment & Credentials Status
    # -------------------------------------------------------------------------
    st.subheader("⚙️ System Status")
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and not openai_key.startswith("your-"):
        st.success("🟢 OpenAI Model: Active (`gpt-4o-mini`)")
    else:
        st.warning("🟡 OpenAI API Key: Not set (Using Grounded Deterministic Synthesizer)")

    langsmith_key = os.getenv("LANGCHAIN_API_KEY", "").strip()
    if langsmith_key and not langsmith_key.startswith("your-"):
        st.success("🟢 LangSmith Tracing: Enabled")
    else:
        st.info("⚪ LangSmith Tracing: Disabled (Local test mode)")


# -----------------------------------------------------------------------------
# Main Chat Area
# -----------------------------------------------------------------------------
st.title("Asterline Research Assistant")
st.markdown(
    f"Multi-agent AI research assistant connected to Asterline's verified knowledge base (`corpus.jsonl`). "
    f"Currently chatting as **`{active_user}`**."
)

# Display existing conversation history for the ACTIVE user only
user_history = st.session_state["conversations"][active_user]

for msg in user_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display expandable audit and evidence cards if present
        if "metadata" in msg and msg["metadata"]:
            meta = msg["metadata"]
            
            # Conflict Alert Card
            findings = meta.get("verification_findings", {})
            if findings.get("conflicts_detected"):
                st.warning("⚠️ **Document Discrepancies / Conflicts Noted:**")
                for c in findings.get("conflict_details", []):
                    st.markdown(f"- {c}")

            col_ev, col_tr = st.columns([1, 1])
            with col_ev:
                with st.expander(f"📚 View Retrieved Evidence ({len(meta.get('evidence', []))} chunks)"):
                    for i, chunk in enumerate(meta.get("evidence", []), 1):
                        st.markdown(f"**[{i}] {chunk.get('title')}** `[{chunk.get('chunk_id')}]`")
                        st.caption(f"Category: {chunk.get('category')} | Version: {chunk.get('version')} | Published: {chunk.get('published')}")
                        st.text(chunk.get("text"))
            with col_tr:
                with st.expander("🔍 View Multi-Agent Execution Trace"):
                    for step in meta.get("execution_steps", []):
                        st.caption(f"-> {step}")


# -----------------------------------------------------------------------------
# User Chat Input Handling
# -----------------------------------------------------------------------------
user_prompt = st.chat_input(f"Ask Asterline Knowledge Base as {active_user}...")

if user_prompt:
    # 1. Append user question to active conversation
    user_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Execute Multi-Agent Graph
    with st.chat_message("assistant"):
        with st.spinner("Multi-Agent Orchestrator is analyzing question, memories, and documents..."):
            try:
                result = run_assistant(question=user_prompt, user_id=active_user)
                final_answer = result.get("final_answer", "No answer generated.")
                evidence = result.get("retrieved_evidence", [])
                findings = result.get("verification_findings", {})
                steps = result.get("execution_steps", [])

                # Render final answer
                st.markdown(final_answer)

                # Render conflict warnings if detected
                if findings.get("conflicts_detected"):
                    st.warning("⚠️ **Document Discrepancies / Conflicts Noted:**")
                    for cd in findings.get("conflict_details", []):
                        st.markdown(f"- {cd}")

                # Render evidence and trace expanders
                col_e, col_t = st.columns([1, 1])
                with col_e:
                    with st.expander(f"📚 Retrieved Evidence ({len(evidence)} chunks)"):
                        for i, ch in enumerate(evidence, 1):
                            st.markdown(f"**[{i}] {ch.get('title')}** `[{ch.get('chunk_id')}]`")
                            st.caption(f"Doc ID: {ch.get('doc_id')} | Version: {ch.get('version')} | Published: {ch.get('published')}")
                            st.text(ch.get("text"))
                with col_t:
                    with st.expander("🔍 Multi-Agent Execution Trace"):
                        for s in steps:
                            st.caption(f"-> {s}")

                # Save assistant response to conversation history
                user_history.append({
                    "role": "assistant",
                    "content": final_answer,
                    "metadata": {
                        "evidence": evidence,
                        "verification_findings": findings,
                        "execution_steps": steps
                    }
                })

            except Exception as e:
                st.error(f"Assistant Execution Error: {e}")
