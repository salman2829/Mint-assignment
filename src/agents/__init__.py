"""LangGraph Multi-Agent Orchestration Package.

Contains:
- state.py: Shared TypedDict AgentState
- research_agent.py: Evidence retriever from local ChromaDB
- memory_agent.py: Isolated personal context retriever from SQLite
- verification_agent.py: Grounding, citation validation, and conflict detector
- graph.py: StateGraph orchestration pipeline
"""
