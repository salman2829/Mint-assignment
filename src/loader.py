"""Document loader and validator for Asterline Knowledge Base (corpus.jsonl).

Reads the original corpus.jsonl in read-only mode, preserves all metadata fields,
and validates records using Pydantic.
"""

import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field, ValidationError


class CorpusRecord(BaseModel):
    """Schema for a single chunk record in corpus.jsonl."""
    chunk_id: str = Field(..., description="Unique chunk identifier, e.g., company:0")
    doc_id: str = Field(..., description="Parent document identifier, e.g., company")
    title: str = Field(..., description="Document title")
    category: str = Field(..., description="Category / topic domain")
    owner: str = Field(..., description="Document owner or team")
    source_url: str = Field(..., description="URI / URL to knowledge base source")
    published: str = Field(..., description="Publication or effective date, e.g., 20260301")
    version: str = Field(..., description="Document version number")
    text: str = Field(..., min_length=1, description="Chunk textual content")

    def to_metadata_dict(self) -> dict:
        """Convert metadata fields into a clean dictionary for vector storage."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "category": self.category,
            "owner": self.owner,
            "source_url": self.source_url,
            "published": self.published,
            "version": str(self.version),
        }


def load_and_validate_corpus(file_path: Path) -> List[CorpusRecord]:
    """
    Loads and strictly validates records from corpus.jsonl.
    
    Args:
        file_path: Absolute or relative Path to corpus.jsonl.
        
    Returns:
        List of validated CorpusRecord instances.
        
    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If file is empty or records fail validation.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found at: {path.resolve()}")

    records: List[CorpusRecord] = []
    unique_chunk_ids = set()
    line_number = 0

    # Open strictly in read-only mode ('r') with UTF-8 encoding
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            line_number += 1
            try:
                data = json.loads(line_str)
                record = CorpusRecord(**data)
                
                # Check for uniqueness of chunk_id
                if record.chunk_id in unique_chunk_ids:
                    raise ValueError(f"Duplicate chunk_id '{record.chunk_id}' found at line {line_number}")
                unique_chunk_ids.add(record.chunk_id)
                records.append(record)

            except json.JSONDecodeError as err:
                raise ValueError(f"Malformed JSON on line {line_number}: {err}") from err
            except ValidationError as err:
                raise ValueError(f"Schema validation error on line {line_number}: {err}") from err

    if not records:
        raise ValueError(f"Corpus file {path} contains no valid records.")

    return records


if __name__ == "__main__":
    from src.config import CORPUS_PATH
    try:
        loaded = load_and_validate_corpus(CORPUS_PATH)
        docs = set(r.doc_id for r in loaded)
        categories = set(r.category for r in loaded)
        print(f"Loaded and validated {len(loaded)} chunks across {len(docs)} documents.")
        print(f"Categories ({len(categories)}): {', '.join(sorted(categories))}")
    except Exception as e:
        print(f"Error loading corpus: {e}")
