# BANASPATI - Multimodal RAG System

**Bubur Panas Personal Assistant** — chatbot berbasis Multimodal RAG untuk menjawab pertanyaan akademik berdasarkan dokumen resmi departemen.

Praktikum Modul 5 — Kecerdasan Buatan dan Pembelajaran Mesin (LAB KCKS, ITS).

## Arsitektur Sistem

```
Dokumen (PDF/DOCX/TXT)
       │
       ▼
┌──────────────────┐
│  Extractor        │  PyMuPDF + python-docx + Tesseract OCR
│  (extractors.py)  │  → teks, tabel, gambar per halaman
└────────┬─────────┘
         │  extracted_documents.jsonl
         ▼
┌──────────────────┐
│  Chunker          │  Recursive / Fixed sliding window
│  (chunking.py)    │  → chunks dengan metadata sumber
└────────┬─────────┘
         │  chunks_recursive_1000_150.jsonl
         ▼
┌──────────────────┐
│  Embedder         │  paraphrase-multilingual-MiniLM-L12-v2
│  (vector_store.py)│  → 384-dim embeddings (cosine)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  ChromaDB         │  HNSW index, persistent storage
│  (artifacts/chroma)│
└────────┬─────────┘
         │  top-k retrieval
         ▼
┌──────────────────┐
│  LLM / SLM        │  Max 9B params (API atau local)
│  + Prompt + Context│  → jawaban berbasis retrieved context
└──────────────────┘
```

## Struktur Project

```
.
├── scripts/
│   ├── data_pipeline.py          # CLI utama (prepare / index / query)
│   └── download_data.sh          # Download dataset dari Google Drive
├── src/banaspati_data/
│   ├── __init__.py
│   ├── extractors.py             # Ekstraksi PDF/DOCX/TXT + OCR
│   ├── chunking.py               # Strategi chunking (fixed/recursive)
│   ├── vector_store.py           # ChromaDB build & query
│   └── io_utils.py               # JSONL read/write
├── notebooks/
│   └── 01_data_processing.ipynb  # Notebook dokumentasi pipeline
├── tests/
│   └── test_data_pipeline.py     # Unit tests
├── docs/
│   └── DATA_ENGINEERING.md       # Dokumentasi teknis data pipeline
├── data/
│   ├── database/                 # Dokumen sumber (8 file)
│   ├── banaspati_eval_questions.csv  # 10 soal evaluasi
│   └── downloads/                # File zip mentah
├── artifacts/                    # Output pipeline (generated)
│   ├── extracted_documents.jsonl
│   ├── chunks_*.jsonl
│   └── chroma/                   # Vector database
├── requirements.txt
├── Soal.md                       # Spesifikasi praktikum
└── README.md
```

## Cara Pakai

### 1. Setup Environment

```bash
# Clone repo
git clone <repo-url>
cd Modul-5-Artificial-Inteligent-Chatbot

# Buat virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
bash scripts/download_data.sh
```

Ini akan mengunduh dan mengekstrak:
- **8 dokumen akademik** (PDF/DOCX) ke `data/database/`
- **10 soal evaluasi** (CSV) ke `data/banaspati_eval_questions.csv`

Dataset berasal dari Google Drive (link di Soal.md).

### 3. Pipeline Data

```bash
# Step 1: Ekstrak dokumen + eksperimen chunking
python scripts/data_pipeline.py prepare --input data/database --output artifacts

# Tambahkan --ocr untuk OCR gambar (perlu Tesseract binary)
python scripts/data_pipeline.py prepare --input data/database --output artifacts --ocr

# Step 2: Bangun vector database
python scripts/data_pipeline.py index \
  --chunks artifacts/chunks_recursive_1000_150.jsonl \
  --persist-dir artifacts/chroma

# Step 3: Uji retrieval
python scripts/data_pipeline.py query "Apa syarat kelulusan mahasiswa?" \
  --persist-dir artifacts/chroma
```

### 4. Jalankan Tests

```bash
python -m pytest tests/ -v
```

## Argumen CLI

### `prepare` — Ekstrak dan chunk dokumen

| Argumen | Wajib | Default | Deskripsi |
|---------|-------|---------|-----------|
| `--input` | Ya | — | Folder berisi dokumen sumber |
| `--output` | Tidak | `artifacts` | Folder output |
| `--ocr` | Tidak | `false` | Aktifkan OCR pada gambar |

### `index` — Bangun vector database

| Argumen | Wajib | Default | Deskripsi |
|---------|-------|---------|-----------|
| `--chunks` | Ya | — | File `.jsonl` hasil chunking |
| `--persist-dir` | Tidak | `artifacts/chroma` | Folder penyimpanan ChromaDB |
| `--collection` | Tidak | `banaspati` | Nama collection |
| `--model` | Tidak | `paraphrase-multilingual-MiniLM-L12-v2` | Model embedding |

### `query` — Cari konteks dari vector database

| Argumen | Wajib | Default | Deskripsi |
|---------|-------|---------|-----------|
| `question` | Ya | — | Pertanyaan pencarian |
| `--persist-dir` | Tidak | `artifacts/chroma` | Folder ChromaDB |
| `--collection` | Tidak | `banaspati` | Nama collection |
| `--model` | Tidak | `paraphrase-multilingual-MiniLM-L12-v2` | Model embedding |
| `--top-k` | Tidak | `5` | Jumlah hasil teratas |

## Konfigurasi Chunking

Pipeline menjalankan 4 eksperimen chunking otomatis:

| Strategi | Chunk Size | Overlap | Keterangan |
|----------|-----------|---------|------------|
| `fixed` | 500 | 100 | Baseline, sliding window |
| `recursive` | 500 | 100 | Sentence-aware, kecil |
| `recursive` | 1000 | 150 | **Rekomendasi** — balance context/precision |
| `recursive` | 1500 | 200 | Context panjang, lebih sedikit chunks |

**Rekomendasi:** `recursive_1000_150` — chunk cukup besar untuk menangkap konteks paragraf penuh, tapi tidak terlalu besar sampai noise.

## Model Embedding

**`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**

- Dimensi: 384
- Multilingual (Indonesia + Inggris)
- Ukuran kecil (~120MB), cepat untuk inference
- Cosine similarity via HNSW index di ChromaDB

## Evaluasi

### Metrik Akurasi/Kualitas
- **RAGAS**: faithfulness, answer relevancy, context precision, context recall
- **LLM-as-a-Judge**: Gemini 3 Flash via Google AI Studio
  - Rubrik: correctness, faithfulness, relevance, completeness, source support, hallucination detection

### Metrik Inferensi
- Retrieval latency
- Generation latency
- End-to-end latency
- Token usage
- Throughput (tokens/sec)
- Estimasi biaya (API models)
- Resource usage — RAM/VRAM (local models)
- TTFT jika mendukung streaming

### File Evaluasi
- `data/banaspati_eval_questions.csv` — 10 soal uji dengan reference answer dan expected source

## OCR (Opsional)

OCR memerlukan **Tesseract binary** terinstall di sistem:

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Jalankan prepare dengan OCR
python scripts/data_pipeline.py prepare --input data/database --output artifacts --ocr
```

OCR dilakukan pada gambar yang ter-embed di PDF/DOCX:
- Upscale 3x untuk gambar < 2500px
- Contrast enhancement 1.5x
- Tesseract `--psm 6` (single text block)

## Troubleshooting

**"ModuleNotFoundError: No module named 'banaspati_data'"**
- Pastikan menjalankan dari root project, bukan dari `scripts/`
- Atau: `export PYTHONPATH=src:$PYTHONPATH`

**"File chunk tidak ditemukan"**
- Jalankan `prepare` dulu sebelum `index`
- Pastikan path ke file `.jsonl` benar

**"Failed to send telemetry event"**
- Sudah difix dengan pin `posthog>=3,<4` di `requirements.txt`
- Jalankan `pip install "posthog>=3,<4"` jika masih muncul

**Tesseract not found**
- Install Tesseract binary (lihat bagian OCR di atas)
- Atau jalankan tanpa flag `--ocr`

## Dokumentasi Lanjutan

- [DATA_ENGINEERING.md](docs/DATA_ENGINEERING.md) — Detail teknis pipeline
- [notebooks/01_data_processing.ipynb](notebooks/01_data_processing.ipynb) — Notebook walkthrough

## License

Tugas praktikum — Kecerdasan Buatan dan Pembelajaran Mesin, ITS.
