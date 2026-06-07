from __future__ import annotations

from pathlib import Path


def build_chroma(
    chunks: list[dict],
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    batch_size: int = 64,
) -> int:
    if not chunks:
        raise ValueError(
            "File chunk kosong. Jalankan perintah prepare pada folder dataset yang berisi dokumen."
        )

    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(
        path=str(persist_dir), settings=Settings(anonymized_telemetry=False)
    )
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [item["text"] for item in batch]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        metadatas = [
            {key: value for key, value in item["metadata"].items() if value is not None}
            for item in batch
        ]
        collection.add(
            ids=[item["id"] for item in batch],
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
    return collection.count()


def query_chroma(
    query: str,
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    top_k: int = 5,
) -> list[dict]:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(
        path=str(persist_dir), settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(collection_name)
    if collection.count() == 0:
        raise ValueError("Collection Chroma kosong. Bangun ulang index dari file chunk yang tidak kosong.")

    model = SentenceTransformer(model_name)
    embedding = model.encode([query], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=embedding, n_results=top_k)
    return [
        {
            "text": document,
            "metadata": metadata,
            "distance": distance,
        }
        for document, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
