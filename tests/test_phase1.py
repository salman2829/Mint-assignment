"""Automated tests for Phase 1: Ingestion, Validation, Vector Store, and Search."""

import pytest
from pathlib import Path

from src.config import CORPUS_PATH, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME
from src.loader import load_and_validate_corpus, CorpusRecord
from src.vector_store import (
    build_or_update_collection,
    search_knowledge_base,
    get_chroma_client,
)


def test_corpus_file_integrity():
    """Verify that corpus.jsonl exists and contains exactly 75 records from 25 documents."""
    assert CORPUS_PATH.exists(), f"corpus.jsonl does not exist at {CORPUS_PATH}"
    
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    assert len(lines) == 75, f"Expected 75 records, but found {len(lines)}"


def test_loader_schema_and_metadata_preservation():
    """Verify that all records parse correctly and preserve every required field."""
    records = load_and_validate_corpus(CORPUS_PATH)
    assert len(records) == 75

    doc_ids = set(r.doc_id for r in records)
    assert len(doc_ids) == 25, f"Expected 25 unique doc_ids, got {len(doc_ids)}"

    for r in records:
        assert isinstance(r, CorpusRecord)
        assert r.chunk_id, "chunk_id must not be empty"
        assert r.doc_id, "doc_id must not be empty"
        assert r.title, "title must not be empty"
        assert r.category, "category must not be empty"
        assert r.owner, "owner must not be empty"
        assert r.source_url, "source_url must not be empty"
        assert r.published, "published must not be empty"
        assert r.version, "version must not be empty"
        assert len(r.text.strip()) > 0, "text must not be empty"

        # Check metadata dict conversion
        meta = r.to_metadata_dict()
        assert meta["chunk_id"] == r.chunk_id
        assert meta["doc_id"] == r.doc_id
        assert meta["title"] == r.title
        assert meta["category"] == r.category


def test_vector_indexing_and_persistence(tmp_path: Path):
    """Verify that records are indexed locally with Sentence Transformers into ChromaDB."""
    test_db_dir = tmp_path / "test_chroma"
    records = load_and_validate_corpus(CORPUS_PATH)
    
    count = build_or_update_collection(
        records=records,
        persist_dir=test_db_dir,
        collection_name="test_collection",
        model_name=EMBEDDING_MODEL_NAME,
        force_reindex=True,
    )
    assert count == 75

    # Verify persistent client can re-open collection and find 75 records
    client = get_chroma_client(test_db_dir)
    col = client.get_collection("test_collection")
    assert col.count() == 75


def test_semantic_search_retrieval(tmp_path: Path):
    """Verify that top-3 search returns relevant chunks with scores and metadata."""
    test_db_dir = tmp_path / "test_chroma"
    records = load_and_validate_corpus(CORPUS_PATH)
    
    build_or_update_collection(
        records=records,
        persist_dir=test_db_dir,
        collection_name="test_collection",
        model_name=EMBEDDING_MODEL_NAME,
    )

    query = "Where are Asterline Labs offices located?"
    results = search_knowledge_base(
        query=query,
        top_k=3,
        persist_dir=test_db_dir,
        collection_name="test_collection",
        model_name=EMBEDDING_MODEL_NAME,
    )

    assert len(results) == 3
    # Top result should be company overview mentioning Pune and Dublin
    top_chunk = results[0]
    assert "company" in top_chunk["doc_id"]
    assert "Pune" in top_chunk["text"] or "Dublin" in top_chunk["text"]
    assert top_chunk["similarity_score"] > 0.0
    assert "distance" in top_chunk
