"""Research Agent Node for LangGraph.

Specializes strictly in searching the local ChromaDB vector store and returning
authentic evidence chunks. Does not hallucinate or invent company facts.
"""

from typing import Dict, Any, List
from src.agents.state import AgentState
from src.vector_store import search_knowledge_base
from src.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME

# Minimum semantic similarity threshold to filter out out-of-domain queries
MIN_RELEVANCE_THRESHOLD = 0.35


def research_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves candidate document chunks from ChromaDB for the user's question.
    Filters out chunks below the relevance threshold to prevent false-positive grounding
    on out-of-domain queries.
    
    Args:
        state: Current AgentState.
        
    Returns:
        Partial state update containing 'retrieved_evidence' and 'execution_steps'.
    """
    question = state.get("question", "").strip()
    execution_steps = list(state.get("execution_steps", []))

    if not question:
        execution_steps.append("ResearchAgent: No question provided. Retrieved 0 chunks.")
        return {
            "retrieved_evidence": [],
            "execution_steps": execution_steps
        }

    # Query local ChromaDB for top 4 candidates
    raw_chunks = search_knowledge_base(
        query=question,
        top_k=4,
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
        model_name=EMBEDDING_MODEL_NAME,
    )

    # Filter by minimum semantic relevance threshold
    relevant_chunks = [c for c in raw_chunks if c.get("similarity_score", 0.0) >= MIN_RELEVANCE_THRESHOLD]

    if not relevant_chunks:
        step_log = (
            f"ResearchAgent: All {len(raw_chunks)} candidate chunks fell below the minimum "
            f"relevance threshold ({MIN_RELEVANCE_THRESHOLD}). Treating as insufficient evidence."
        )
    else:
        step_log = (
            f"ResearchAgent: Retrieved {len(relevant_chunks)} relevant chunks (threshold >= {MIN_RELEVANCE_THRESHOLD}) "
            f"from ChromaDB using '{EMBEDDING_MODEL_NAME}'."
        )
    execution_steps.append(step_log)

    return {
        "retrieved_evidence": relevant_chunks,
        "execution_steps": execution_steps
    }
