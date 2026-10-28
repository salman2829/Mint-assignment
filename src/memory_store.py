"""SQLite Persistent User Memory Store with Strict User Isolation.

Manages persistent user context across sessions while enforcing:
1. Strict user-level isolation on every query and mutation.
2. Separation of user personal context from authoritative company evidence.
3. Proactive secret/credential sanitization (no passwords, tokens, or API keys).
"""

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union
from pydantic import BaseModel, Field

from src.config import SQLITE_DB_PATH


# Patterns for detecting potential secrets, tokens, and credentials
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}", re.IGNORECASE),         # OpenAI / typical API key
    re.compile(r"gh[pousr]-[a-zA-Z0-9]{20,}", re.IGNORECASE),     # GitHub token
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE), # Bearer token
    re.compile(r"ey[a-zA-Z0-9_\-]{20,}\.ey[a-zA-Z0-9_\-]{20,}", re.IGNORECASE), # JWT
    re.compile(r"(api[_-]?key|password|passwd|secret)\s*[:=]\s*\S+", re.IGNORECASE), # Key/secret assignments
]


def validate_no_secrets(text: str) -> None:
    """Raises ValueError if text contains API keys, passwords, or credentials."""
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise ValueError(
                "Security Exception: Storing API keys, tokens, or passwords in user memory is prohibited."
            )


class MemoryRecord(BaseModel):
    """Schema for a single persistent memory."""
    id: int = Field(..., description="Unique memory ID")
    user_id: str = Field(..., description="Unique user identifier owner")
    memory_text: str = Field(..., description="Stored memory or preference content")
    is_active: bool = Field(True, description="Active status (supports forgetting/soft-delete)")
    created_at: str = Field(..., description="ISO 8601 UTC creation timestamp")
    updated_at: str = Field(..., description="ISO 8601 UTC last update timestamp")

    @property
    def is_company_evidence(self) -> bool:
        """Personal memories are NEVER company-document evidence."""
        return False

    def format_as_context(self) -> str:
        """Formats memory cleanly for prompt injection with explicit distinction."""
        return f"[User Personal Context | ID:{self.id}]: {self.memory_text}"


class SQLiteMemoryStore:
    """Persistent user memory repository backed by SQLite with strict user isolation."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path or SQLITE_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Creates a SQLite connection with row factory enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes tables and indexes using parameterized DDL."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    memory_text TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_active ON memories(user_id, is_active);")
            conn.commit()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def add_memory(self, user_id: str, memory_text: str) -> MemoryRecord:
        """
        Saves a memory for a specific user.
        
        Args:
            user_id: User identifier.
            memory_text: Content of the preference or context.
            
        Raises:
            ValueError: If user_id is empty, memory_text is empty, or secrets detected.
        """
        user_id = str(user_id).strip()
        memory_text = str(memory_text).strip()

        if not user_id:
            raise ValueError("user_id must not be empty.")
        if not memory_text:
            raise ValueError("memory_text must not be empty.")

        validate_no_secrets(memory_text)

        now = self._now_iso()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memories (user_id, memory_text, is_active, created_at, updated_at)
                VALUES (?, ?, 1, ?, ?)
                """,
                (user_id, memory_text, now, now)
            )
            memory_id = cursor.lastrowid
            conn.commit()

        return MemoryRecord(
            id=memory_id,
            user_id=user_id,
            memory_text=memory_text,
            is_active=True,
            created_at=now,
            updated_at=now,
        )

    def list_memories(self, user_id: str, active_only: bool = True) -> List[MemoryRecord]:
        """
        Lists all memories for the specified user with strict isolation.
        
        Args:
            user_id: Target user identifier.
            active_only: If True, returns only active (non-forgotten) memories.
        """
        user_id = str(user_id).strip()
        if not user_id:
            return []

        query = "SELECT id, user_id, memory_text, is_active, created_at, updated_at FROM memories WHERE user_id = ?"
        params = [user_id]

        if active_only:
            query += " AND is_active = 1"
        query += " ORDER BY id ASC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [
            MemoryRecord(
                id=row["id"],
                user_id=row["user_id"],
                memory_text=row["memory_text"],
                is_active=bool(row["is_active"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def get_memory(self, user_id: str, memory_id: int) -> Optional[MemoryRecord]:
        """
        Retrieves a single memory by ID, strictly enforcing that it belongs to user_id.
        Returns None if memory does not exist or belongs to another user.
        """
        user_id = str(user_id).strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, user_id, memory_text, is_active, created_at, updated_at
                FROM memories
                WHERE id = ? AND user_id = ?
                """,
                (memory_id, user_id)
            )
            row = cursor.fetchone()

        if not row:
            return None

        return MemoryRecord(
            id=row["id"],
            user_id=row["user_id"],
            memory_text=row["memory_text"],
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update_memory(self, user_id: str, memory_id: int, new_text: str) -> bool:
        """
        Updates/corrects an existing memory, strictly scoped to user_id.
        Returns False if memory_id does not belong to user_id.
        """
        user_id = str(user_id).strip()
        new_text = str(new_text).strip()

        if not new_text:
            raise ValueError("Updated memory text must not be empty.")
        validate_no_secrets(new_text)

        now = self._now_iso()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE memories
                SET memory_text = ?, updated_at = ?
                WHERE id = ? AND user_id = ? AND is_active = 1
                """,
                (new_text, now, memory_id, user_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_memory(self, user_id: str, memory_id: int, hard: bool = False) -> bool:
        """
        Forgets/deletes a memory, strictly scoped to user_id.
        
        Args:
            user_id: User identifier.
            memory_id: ID of the memory to forget.
            hard: If True, completely deletes the record; if False, marks is_active = 0.
        """
        user_id = str(user_id).strip()
        now = self._now_iso()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if hard:
                cursor.execute(
                    "DELETE FROM memories WHERE id = ? AND user_id = ?",
                    (memory_id, user_id)
                )
            else:
                cursor.execute(
                    """
                    UPDATE memories
                    SET is_active = 0, updated_at = ?
                    WHERE id = ? AND user_id = ? AND is_active = 1
                    """,
                    (now, memory_id, user_id)
                )
            conn.commit()
            return cursor.rowcount > 0

    def retrieve_relevant_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 3
    ) -> List[MemoryRecord]:
        """
        Retrieves top relevant memories for a user based on semantic similarity.
        Strictly searches ONLY the active memories belonging to user_id.
        """
        memories = self.list_memories(user_id=user_id, active_only=True)
        if not memories or not query.strip():
            return []

        # If only a few memories, or for semantic ranking, use local SentenceTransformer
        try:
            import numpy as np
            from src.vector_store import get_embedding_model

            model = get_embedding_model()
            query_emb = model.encode([query], normalize_embeddings=True)
            mem_texts = [m.memory_text for m in memories]
            mem_embs = model.encode(mem_texts, normalize_embeddings=True)

            # Cosine similarity: dot product of normalized vectors
            scores = np.dot(mem_embs, query_emb.T).flatten()
            ranked_indices = np.argsort(scores)[::-1][:top_k]

            return [memories[idx] for idx in ranked_indices]
        except Exception:
            # Fallback to simple keyword overlap if embedding model fails
            q_tokens = set(query.lower().split())
            scored = []
            for m in memories:
                overlap = len(set(m.memory_text.lower().split()) & q_tokens)
                scored.append((overlap, m))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [m for _, m in scored[:top_k]]
