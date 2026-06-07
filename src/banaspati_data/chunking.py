from __future__ import annotations

import hashlib
import re


def _fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap harus lebih kecil daripada chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def _recursive_chunks(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap harus lebih kecil daripada chunk_size")
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(part) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_fixed_chunks(part, size, overlap))
            continue
        candidate = f"{current}\n{part}".strip()
        if len(candidate) <= size:
            current = candidate
        else:
            chunks.append(current)
            prefix = current[-overlap:].strip() if overlap else ""
            current = f"{prefix}\n{part}".strip()
            if len(current) > size:
                current = part
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(
    documents: list[dict],
    strategy: str = "recursive",
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    splitter = {"fixed": _fixed_chunks, "recursive": _recursive_chunks}.get(strategy)
    if splitter is None:
        raise ValueError(f"Strategi tidak dikenal: {strategy}")

    output: list[dict] = []
    for document in documents:
        # Keep raw image assets for audit/vision processing, but do not embed
        # non-semantic placeholders such as "[Gambar 1 pada halaman 2]".
        if document["metadata"].get("content_type") == "image":
            continue
        for index, text in enumerate(splitter(document["text"], chunk_size, overlap)):
            chunk_key = f"{document['id']}|{strategy}|{chunk_size}|{overlap}|{index}"
            metadata = {
                **document["metadata"],
                "parent_id": document["id"],
                "chunk_index": index,
                "chunk_strategy": strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": overlap,
            }
            output.append(
                {
                    "id": hashlib.sha1(chunk_key.encode()).hexdigest(),
                    "text": text,
                    "metadata": metadata,
                }
            )
    return output
