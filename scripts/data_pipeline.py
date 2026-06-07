#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banaspati_data.chunking import chunk_documents
from banaspati_data.extractors import extract_directory
from banaspati_data.io_utils import read_jsonl, write_jsonl
from banaspati_data.vector_store import build_chroma, query_chroma

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def experiment(documents: list[dict], output_dir: Path) -> list[dict]:
    configurations = [
        ("fixed", 500, 100),
        ("recursive", 500, 100),
        ("recursive", 1000, 150),
        ("recursive", 1500, 200),
    ]
    report = []
    for strategy, size, overlap in configurations:
        chunks = chunk_documents(documents, strategy, size, overlap)
        lengths = [len(item["text"]) for item in chunks]
        sources = {item["metadata"]["source_file"] for item in chunks}
        row = {
            "strategy": strategy,
            "chunk_size": size,
            "overlap": overlap,
            "chunk_count": len(chunks),
            "mean_chars": round(statistics.mean(lengths), 2) if lengths else 0,
            "median_chars": statistics.median(lengths) if lengths else 0,
            "max_chars": max(lengths, default=0),
            "source_coverage": len(sources),
        }
        report.append(row)
        write_jsonl(output_dir / f"chunks_{strategy}_{size}_{overlap}.jsonl", chunks)
    (output_dir / "chunking_experiment.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline data Multimodal RAG BANASPATI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Ekstrak dokumen dan jalankan eksperimen chunking")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=Path("artifacts"))
    prepare.add_argument("--ocr", action="store_true")

    index = subparsers.add_parser("index", help="Buat vector database Chroma")
    index.add_argument("--chunks", type=Path, required=True)
    index.add_argument("--persist-dir", type=Path, default=Path("artifacts/chroma"))
    index.add_argument("--collection", default="banaspati")
    index.add_argument("--model", default=DEFAULT_MODEL)

    query = subparsers.add_parser("query", help="Uji retrieval dari Chroma")
    query.add_argument("question")
    query.add_argument("--persist-dir", type=Path, default=Path("artifacts/chroma"))
    query.add_argument("--collection", default="banaspati")
    query.add_argument("--model", default=DEFAULT_MODEL)
    query.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()
    try:
        if args.command == "prepare":
            documents = extract_directory(
                args.input, use_ocr=args.ocr, assets_dir=args.output / "images"
            )
            args.output.mkdir(parents=True, exist_ok=True)
            write_jsonl(args.output / "extracted_documents.jsonl", documents)
            report = experiment(documents, args.output)
            print(json.dumps({"extracted_units": len(documents), "experiments": report}, indent=2))
        elif args.command == "index":
            if not args.chunks.is_file():
                raise FileNotFoundError(f"File chunk tidak ditemukan: {args.chunks}")
            count = build_chroma(
                read_jsonl(args.chunks), args.persist_dir, args.collection, args.model
            )
            print(json.dumps({"indexed_chunks": count, "persist_dir": str(args.persist_dir)}))
        else:
            results = query_chroma(
                args.question, args.persist_dir, args.collection, args.model, args.top_k
            )
            print(json.dumps(results, indent=2, ensure_ascii=False))
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
