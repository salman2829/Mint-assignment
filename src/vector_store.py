"""Local vector store management with ChromaDB and Sentence Transformers.

Stores document embeddings and metadata in a persistent local ChromaDB collection.
No remote or cloud embedding APIs are used.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.loader import CorpusRecord
from src.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME


_CACHED_MODEL: Optional[SentenceTransformer] = None


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Loads and caches the local SentenceTransformer embedding model."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None or getattr(_CACHED_MODEL, "_model_name_or_path", "") != model_name:
        print(f"[VectorStore] Loading local embedding model: '{model_name}'...")
        _CACHED_MODEL = SentenceTransformer(model_name)
        print("[VectorStore] Local embedding model loaded successfully.")
    return _CACHED_MODEL


def get_chroma_client(persist_dir: Path = CHROMA_PERSIST_DIR) -> chromadb.PersistentClient:
    """Returns a persistent ChromaDB client saving to the specified directory."""
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def build_or_update_collection(
    records: List[CorpusRecord],
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL_NAME,
    force_reindex: bool = False,
) -> int:
    """
    Embeds corpus records using Sentence Transformers and stores them in ChromaDB.
    
    Args:
        records: List of validated CorpusRecord objects.
        persist_dir: Directory to save persistent ChromaDB files.
        collection_name: Name of the ChromaDB collection.
        model_name: SentenceTransformers model identifier.
        force_reindex: If True, recreates the collection from scratch.
        
    Returns:
        Number of documents indexed in the collection.
    """
    client = get_chroma_client(persist_dir)
    
    # If forcing re-index, delete existing collection
    if force_reindex:
        try:
            client.delete_collection(name=collection_name)
            print(f"[VectorStore] Deleted existing collection '{collection_name}' for clean re-index.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )

    current_count = collection.count()
    if current_count == len(records) and not force_reindex:
        print(f"[VectorStore] Collection '{collection_name}' is already up-to-date with {current_count} chunks.")
        return current_count

    print(f"[VectorStore] Indexing {len(records)} chunks into ChromaDB...")
    model = get_embedding_model(model_name)

    ids = [r.chunk_id for r in records]
    documents = [r.text for r in records]
    metadatas = [r.to_metadata_dict() for r in records]

    # Generate embeddings locally in batches
    print("[VectorStore] Generating embeddings locally via SentenceTransformers...")
    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()

    # Upsert into ChromaDB
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    total_count = collection.count()
    print(f"[VectorStore] Successfully indexed {total_count} chunks into '{collection_name}'.")
    return total_count


def search_knowledge_base(
    query: str,
    top_k: int = 3,
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> List[Dict[str, Any]]:
    """
    Searches the knowledge base using local embeddings and cosine similarity.
    
    Args:
        query: User question or search phrase.
        top_k: Number of top chunks to return.
        persist_dir: Path to ChromaDB storage.
        collection_name: Name of ChromaDB collection.
        model_name: Local embedding model name.
        
    Returns:
        List of dicts containing chunk details and similarity score.
    """
    client = get_chroma_client(persist_dir)
    collection = client.get_collection(name=collection_name)

    model = get_embedding_model(model_name)
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    formatted_results = []
    if results and results.get("ids") and results["ids"][0]:
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):
            # With cosine distance: similarity = 1 - distance
            cosine_dist = distances[i]
            similarity_score = max(0.0, 1.0 - cosine_dist)

            meta = metas[i]
            formatted_results.append({
                "chunk_id": ids[i],
                "doc_id": meta.get("doc_id", ""),
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
                "owner": meta.get("owner", ""),
                "source_url": meta.get("source_url", ""),
                "published": meta.get("published", ""),
                "version": meta.get("version", ""),
                "text": docs[i],
                "distance": round(cosine_dist, 4),
                "similarity_score": round(similarity_score, 4),
            })

    return formatted_results
