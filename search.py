"""Command-line search test for the Asterline Knowledge Base (Phase 1).

Accepts a question, queries the local ChromaDB vector store, and displays
the top 3 relevant chunks with full metadata, text, and scores.

Usage:
    python search.py "What is Asterline Labs and where are its offices?"
    python search.py --reindex
"""

import sys
import argparse
from pathlib import Path

from src.config import CORPUS_PATH, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME
from src.loader import load_and_validate_corpus
from src.vector_store import (
    build_or_update_collection,
    search_knowledge_base,
    get_chroma_client,
)


def ensure_index_ready(force_reindex: bool = False):
    """Ensures that the ChromaDB collection exists and contains all 75 chunks."""
    client = get_chroma_client(CHROMA_PERSIST_DIR)
    needs_indexing = force_reindex
    
    try:
        col = client.get_collection(name=CHROMA_COLLECTION_NAME)
        if col.count() == 0:
            needs_indexing = True
    except Exception:
        needs_indexing = True

    if needs_indexing:
        print("[Setup] Loading and validating original corpus.jsonl...")
        records = load_and_validate_corpus(CORPUS_PATH)
        print(f"[Setup] Loaded {len(records)} verified records from '{CORPUS_PATH.name}'.")
        build_or_update_collection(
            records=records,
            persist_dir=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME,
            model_name=EMBEDDING_MODEL_NAME,
            force_reindex=force_reindex,
        )


def run_search(query: str, top_k: int = 3):
    """Executes a search and displays nicely formatted results."""
    ensure_index_ready()

    print("\n" + "=" * 80)
    print(f" SEARCH QUERY: '{query}'")
    print(f" RETRIEVING TOP {top_k} RELEVANT CHUNKS (Local Sentence Transformers + ChromaDB)")
    print("=" * 80 + "\n")

    results = search_knowledge_base(
        query=query,
        top_k=top_k,
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
        model_name=EMBEDDING_MODEL_NAME,
    )

    if not results:
        print("No matching chunks found.")
        return

    for rank, item in enumerate(results, start=1):
        print(f"[{rank}] Chunk ID: {item['chunk_id']} | Doc ID: {item['doc_id']}")
        print(f"    Title:       {item['title']}")
        print(f"    Category:    {item['category']} | Version: {item['version']} | Published: {item['published']}")
        print(f"    Owner:       {item['owner']}")
        print(f"    Similarity:  {item['similarity_score']} (Cosine Distance: {item['distance']})")
        print(f"    Source URL:  {item['source_url']}")
        print("    Content:")
        # Indent content for clean reading
        for line in item['text'].split("\n"):
            print(f"      {line}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Search Asterline Knowledge Base (Local ChromaDB)")
    parser.add_argument("query", nargs="*", help="Question or search query string")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to return (default: 3)")
    parser.add_argument("--reindex", action="store_true", help="Force rebuilding the local vector index from corpus.jsonl")

    args = parser.parse_args()

    if args.reindex:
        print("[Search CLI] Initiating forced re-indexing...")
        ensure_index_ready(force_reindex=True)
        if not args.query:
            print("[Search CLI] Re-indexing complete.")
            return

    query_str = " ".join(args.query).strip() if args.query else ""
    if not query_str:
        # Default representative question for automated testing
        query_str = "What is Asterline Labs and where are its offices located?"
        print(f"[Search CLI] No query provided. Using test question: '{query_str}'")

    run_search(query_str, top_k=args.top_k)


if __name__ == "__main__":
    main()
