"""Memory Agent Node for LangGraph.

Retrieves isolated personal context from SQLite for the specific active user.
Keeps personal user preferences strictly separated from verified company facts.
"""

from typing import Dict, Any, List
from src.agents.state import AgentState
from src.memory_store import SQLiteMemoryStore


def memory_agent_node(state: AgentState, memory_store: SQLiteMemoryStore = None) -> Dict[str, Any]:
    """
    Retrieves active user memories relevant to the question for state['user_id'].
    
    Args:
        state: Current AgentState.
        memory_store: Optional injected SQLiteMemoryStore (useful for testing).
        
    Returns:
        Partial state update containing 'relevant_memories' and 'execution_steps'.
    """
    user_id = state.get("user_id", "").strip()
    question = state.get("question", "").strip()
    execution_steps = list(state.get("execution_steps", []))

    if not user_id:
        execution_steps.append("MemoryAgent: Anonymous/no user_id provided. Zero memories retrieved.")
        return {
            "relevant_memories": [],
            "execution_steps": execution_steps
        }

    store = memory_store or SQLiteMemoryStore()

    # Retrieve relevant memories strictly scoped to user_id
    recalled_records = store.retrieve_relevant_memories(
        user_id=user_id,
        query=question,
        top_k=3
    )

    formatted_memories = [r.format_as_context() for r in recalled_records]
    step_log = (
        f"MemoryAgent: Retrieved {len(formatted_memories)} isolated memories for user '{user_id}' from SQLite."
    )
    execution_steps.append(step_log)

    return {
        "relevant_memories": formatted_memories,
        "execution_steps": execution_steps
    }
