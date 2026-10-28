"""Automated tests for Phase 2: User-Isolated Persistent Memory (SQLite).

Verifies:
1. CRUD operations (Add, List, Update/Correct, Delete/Forget).
2. Strict cross-user isolation across all operations.
3. Database persistence across store re-instantiations.
4. Secret sanitization preventing API keys/passwords from being stored.
5. Distinction between personal context and company-document evidence.
6. Semantic retrieval of relevant user memories.
"""

import pytest
from pathlib import Path
from src.memory_store import SQLiteMemoryStore, MemoryRecord


@pytest.fixture
def temp_store(tmp_path: Path) -> SQLiteMemoryStore:
    """Fixture providing a temporary SQLite memory store."""
    db_file = tmp_path / "test_memories.db"
    return SQLiteMemoryStore(db_path=db_file)


def test_add_and_list_memories(temp_store: SQLiteMemoryStore):
    """Verify adding memories and listing active memories for a user."""
    rec1 = temp_store.add_memory("alice", "I prefer summary bullet points.")
    rec2 = temp_store.add_memory("alice", "My workspace is in the Dublin region.")

    assert rec1.id > 0
    assert rec2.id > rec1.id
    assert rec1.user_id == "alice"
    assert rec1.is_active is True
    assert rec1.created_at
    assert rec1.updated_at

    memories = temp_store.list_memories("alice")
    assert len(memories) == 2
    assert memories[0].memory_text == "I prefer summary bullet points."
    assert memories[1].memory_text == "My workspace is in the Dublin region."


def test_cross_user_isolation(temp_store: SQLiteMemoryStore):
    """
    CRITICAL TEST: Verify that User B cannot access, read, update,
    or delete User A's memories under any circumstances.
    """
    alice_mem = temp_store.add_memory("alice", "Alice secret project name is Falcon.")

    # 1. Bob cannot see Alice's memory in his list
    bob_mems = temp_store.list_memories("bob")
    assert len(bob_mems) == 0

    # 2. Bob cannot retrieve Alice's memory by supplying her memory ID
    retrieved = temp_store.get_memory(user_id="bob", memory_id=alice_mem.id)
    assert retrieved is None

    # 3. Bob cannot update/overwrite Alice's memory by supplying her memory ID
    update_success = temp_store.update_memory(
        user_id="bob", memory_id=alice_mem.id, new_text="Tampered text"
    )
    assert update_success is False

    # Alice's memory remains intact
    alice_check = temp_store.get_memory("alice", alice_mem.id)
    assert alice_check is not None
    assert alice_check.memory_text == "Alice secret project name is Falcon."

    # 4. Bob cannot delete Alice's memory
    delete_success = temp_store.delete_memory(user_id="bob", memory_id=alice_mem.id)
    assert delete_success is False

    alice_mems = temp_store.list_memories("alice")
    assert len(alice_mems) == 1

    # 5. Bob searching memories gets zero results from Alice
    bob_search = temp_store.retrieve_relevant_memories(
        user_id="bob", query="Falcon project"
    )
    assert len(bob_search) == 0


def test_memory_update_and_persistence_across_reopen(tmp_path: Path):
    """Verify that memory updates persist after closing and re-opening the database."""
    db_file = tmp_path / "persistent_test.db"
    
    # Session 1: Create and update
    store1 = SQLiteMemoryStore(db_path=db_file)
    rec = store1.add_memory("charlie", "Initial role: Junior Analyst.")
    success = store1.update_memory("charlie", rec.id, "Promoted role: Senior Analyst.")
    assert success is True

    # Session 2: New store instance simulating a brand new application restart
    store2 = SQLiteMemoryStore(db_path=db_file)
    reloaded = store2.get_memory("charlie", rec.id)

    assert reloaded is not None
    assert reloaded.memory_text == "Promoted role: Senior Analyst."
    assert reloaded.updated_at >= reloaded.created_at


def test_forget_and_soft_delete(temp_store: SQLiteMemoryStore):
    """Verify forgetting a memory sets is_active=False and excludes it from active lists."""
    rec = temp_store.add_memory("dana", "I work exclusively on Windows.")
    
    # Soft delete (forget)
    deleted = temp_store.delete_memory("dana", rec.id, hard=False)
    assert deleted is True

    # Active list must be empty
    active_mems = temp_store.list_memories("dana", active_only=True)
    assert len(active_mems) == 0

    # Inactive list preserves audit trail
    all_mems = temp_store.list_memories("dana", active_only=False)
    assert len(all_mems) == 1
    assert all_mems[0].is_active is False


def test_hard_delete(temp_store: SQLiteMemoryStore):
    """Verify hard delete completely purges the row from SQLite."""
    rec = temp_store.add_memory("evan", "Temporary context.")
    deleted = temp_store.delete_memory("evan", rec.id, hard=True)
    assert deleted is True

    all_mems = temp_store.list_memories("evan", active_only=False)
    assert len(all_mems) == 0


def test_secret_sanitization_rejects_credentials(temp_store: SQLiteMemoryStore):
    """Verify that trying to store API keys or passwords raises ValueError."""
    with pytest.raises(ValueError, match="Security Exception"):
        temp_store.add_memory("frank", "My OpenAI key is sk-1234567890abcdef1234567890abcdef")

    with pytest.raises(ValueError, match="Security Exception"):
        temp_store.add_memory("frank", "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0")

    with pytest.raises(ValueError, match="Security Exception"):
        temp_store.add_memory("frank", "password: SuperSecretPassword123")

    # Confirm no memories were stored
    assert len(temp_store.list_memories("frank")) == 0


def test_company_evidence_flag_isolation():
    """Verify that user personal context is explicitly flagged as NOT company evidence."""
    rec = MemoryRecord(
        id=1,
        user_id="grace",
        memory_text="Asterline pricing is $5 per month.",
        is_active=True,
        created_at="2026-10-26T00:00:00Z",
        updated_at="2026-10-26T00:00:00Z",
    )
    assert rec.is_company_evidence is False
    assert "[User Personal Context | ID:1]" in rec.format_as_context()


def test_relevant_memory_retrieval(temp_store: SQLiteMemoryStore):
    """Verify semantic/keyword retrieval of relevant user memories for follow-up questions."""
    temp_store.add_memory("heidi", "I prefer explanations using Python code examples.")
    temp_store.add_memory("heidi", "I drink black tea in the morning.")
    temp_store.add_memory("heidi", "My deployment environment is AWS ECS with Terraform.")

    results = temp_store.retrieve_relevant_memories(
        user_id="heidi",
        query="Which cloud container and infrastructure tools do I use?",
        top_k=1,
    )
    assert len(results) == 1
    assert "AWS ECS" in results[0].memory_text
