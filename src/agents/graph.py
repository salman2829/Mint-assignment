"""LangGraph StateGraph builder for Asterline Multi-Agent Research Assistant.

Orchestrates:
  START -> memory_agent -> research_agent -> verification_agent -> END
"""

from typing import Optional
from langgraph.graph import StateGraph, START, END

from src.agents.state import AgentState
from src.agents.memory_agent import memory_agent_node
from src.agents.research_agent import research_agent_node
from src.agents.verification_agent import verification_agent_node
from src.memory_store import SQLiteMemoryStore


def create_research_assistant_graph(llm=None, memory_store: Optional[SQLiteMemoryStore] = None):
    """
    Constructs and compiles the 3-agent LangGraph workflow.
    
    Args:
        llm: Optional language model instance (for mocking or custom models).
        memory_store: Optional custom SQLiteMemoryStore instance.
        
    Returns:
        Compiled StateGraph runnable.
    """
    builder = StateGraph(AgentState)

    # Wrap nodes to allow dependency injection
    def _memory_node(state: AgentState):
        return memory_agent_node(state, memory_store=memory_store)

    def _research_node(state: AgentState):
        return research_agent_node(state)

    def _verification_node(state: AgentState):
        return verification_agent_node(state, llm=llm)

    # Register nodes
    builder.add_node("memory_agent", _memory_node)
    builder.add_node("research_agent", _research_node)
    builder.add_node("verification_agent", _verification_node)

    # Define linear execution flow
    builder.add_edge(START, "memory_agent")
    builder.add_edge("memory_agent", "research_agent")
    builder.add_edge("research_agent", "verification_agent")
    builder.add_edge("verification_agent", END)

    return builder.compile()


def run_assistant(
    question: str,
    user_id: str = "default_user",
    llm=None,
    memory_store: Optional[SQLiteMemoryStore] = None
) -> AgentState:
    """
    Convenience function to execute the multi-agent graph on a user question.
    """
    app = create_research_assistant_graph(llm=llm, memory_store=memory_store)

    initial_state: AgentState = {
        "question": question,
        "user_id": user_id,
        "retrieved_evidence": [],
        "relevant_memories": [],
        "verification_findings": {},
        "final_answer": "",
        "execution_steps": [f"Orchestrator: Initialized request for user '{user_id}'."],
    }

    result = app.invoke(initial_state)
    return result
