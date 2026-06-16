# BANASPATI — Multimodal RAG System

**Bubur Panas Personal Assistant** — Chatbot berbasis Multimodal RAG (Retrieval-Augmented Generation) untuk menjawab pertanyaan akademik berdasarkan dokumen resmi Departemen Teknologi Informasi, ITS.

Praktikum Modul 5 — Kecerdasan Buatan dan Pembelajaran Mesin (LAB KCKS, ITS).

---

## Daftar Isi

- [Arsitektur Sistem](#arsitektur-sistem)
- [Struktur Project](#struktur-project)
- [Prasyarat & Setup](#prasyarat--setup)
- [Cara Pakai](#cara-pakai)
- [API Manager & Rotasi Kunci](#api-manager--rotasi-kunci)
- [Pipeline Data (CLI)](#pipeline-data-cli)
- [Konfigurasi Chunking](#konfigurasi-chunking)
- [Model Embedding](#model-embedding)
- [Evaluasi & Metrik](#evaluasi--metrik)
- [OCR (Tesseract)](#ocr-tesseract)
- [Daftar Scripts Utilitas](#daftar-scripts-utilitas)
- [Troubleshooting](#troubleshooting)
- [Dokumentasi Lanjutan](#dokumentasi-lanjutan)
- [Pembagian Tugas Kelompok](#pembagian-tugas-kelompok)
- [License](#license)

---

## Arsitektur Sistem

```
Dokumen (PDF / DOCX / TXT)
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
│ (artifacts/chroma)│
└────────┬─────────┘
         │  Hybrid Retrieval (k=2000 → rerank → top 10)
         ▼
┌──────────────────┐
│  Gemini 2.5 Flash │  API (<9B params, Free Tier)
│  + Anti-Halusinasi│  + Rotasi 3 API Keys otomatis
│  Prompt + Context │  → jawaban berbasis retrieved context
└──────────────────┘
```

---

## Struktur Project

```
.
├── .env                              # ⛔ (HIDDEN) API Keys Gemini (3 keys)
├── .gitignore
├── README.md
├── requirements.txt                  # Dependensi Python
├── Soal.md                           # Spesifikasi praktikum
│
├── data/                             # ⛔ (HIDDEN) Seluruh folder data
│   ├── database/                     #    8 dokumen sumber (PDF/DOCX)
│   │   ├── Data Dosen.pdf
│   │   ├── Jadwal Perkuliahan.docx   #    ⚠ Berisi gambar (perlu OCR)
│   │   ├── Kalender-Akademik-ITS-Thn-Akademik-2025-2026.pdf
│   │   ├── Kurikulum.pdf             #    ~26 MB, 800+ halaman
│   │   ├── Nilai snbt 2025.pdf
│   │   ├── Peraturan Akademik.pdf
│   │   ├── Sosialisasi Magang dan Prestasi DTI.pdf
│   │   └── Visi Misi Departemen.pdf
│   ├── banaspati_eval_questions.csv  #    10 soal evaluasi + ground truth
│   └── downloads/                    #    File zip mentah
│
├── artifacts/                        # ⛔ (HIDDEN) Output pipeline (generated)
│   ├── extracted_documents.jsonl     #    Hasil ekstraksi dokumen
│   ├── chunks_*.jsonl                #    Hasil chunking (4 eksperimen)
│   ├── chroma/                       #    Vector database ChromaDB
│   ├── evaluation_results.csv        #    Hasil evaluasi 10 soal
│   └── images/                       #    Gambar yang diekstrak dari PDF
│
├── src/banaspati_data/               # Source code utama
│   ├── __init__.py
│   ├── extractors.py                 # Ekstraksi PDF/DOCX/TXT + OCR
│   ├── chunking.py                   # Strategi chunking (fixed/recursive)
│   ├── vector_store.py               # ChromaDB build & query
│   ├── io_utils.py                   # JSONL read/write
│   └── api_manager.py               # ✅ Rotasi API Key + Rate Limiter
│
├── notebooks/
│   ├── 01_data_processing.ipynb      # Pipeline data (Orang 1)
│   ├── 02_rag_pipeline_sandbox.ipynb # Demo Chatbot RAG (Orang 2)
│   └── 03_evaluation_metrics.ipynb   # Evaluasi & Metrik (Orang 3)
│
├── scripts/                          # Skrip utilitas & automation
│   ├── data_pipeline.py              # CLI utama (prepare / index / query)
│   ├── download_data.sh              # Download dataset dari Google Drive
│   ├── run_eval_03.py                # Skrip evaluasi otomatis (10 soal)
│   ├── generate_mock_results.py      # Fallback evaluasi tanpa API
│   ├── reindex_jadwal_ocr_v2.py      # OCR untuk Jadwal Perkuliahan.docx
│   └── ...                           # Skrip patch & utilitas lainnya
│
├── tests/
│   └── test_data_pipeline.py         # Unit tests
│
└── docs/
    └── DATA_ENGINEERING.md           # Dokumentasi teknis data pipeline
```

---

## Prasyarat & Setup

### File yang Disembunyikan (`.gitignore`)

Beberapa file/folder **tidak ikut ter-push ke GitHub** karena alasan keamanan dan ukuran. Anda harus menyiapkannya secara manual:

| File / Folder | Keterangan | Cara Mendapatkan |
|---|---|---|
| `.env` | 3 API Keys Google Gemini | Buat manual (lihat langkah di bawah) |
| `data/` | 8 dokumen akademik + 10 soal evaluasi | Jalankan `scripts/download_data.sh` |
| `artifacts/` | Vector DB, chunks, hasil evaluasi | Di-generate otomatis via notebook `01_data_processing.ipynb` |
| `venv/` | Virtual environment Python | Buat manual via `python -m venv venv` |

### Prasyarat Sistem

| Software | Versi Minimum | Keterangan |
|---|---|---|
| **Python** | 3.10+ | Direkomendasikan 3.13 |
| **Tesseract OCR** | 5.x | **Wajib** untuk memproses `Jadwal Perkuliahan.docx` |
| **Git** | 2.x | Untuk clone repository |

### 1. Clone Repository

```bash
git clone https://github.com/HikariReiziq/Modul-5-Artificial-Inteligent-Chatbot.git
cd Modul-5-Artificial-Inteligent-Chatbot
```

### 2. Buat Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependensi utama:
- `langchain`, `langchain-huggingface`, `langchain-chroma` — Framework RAG
- `chromadb` — Vector database
- `sentence-transformers` — Model embedding
- `google-genai`, `langchain-google-genai` — Google Gemini API
- `ragas` — Framework evaluasi RAG
- `pytesseract`, `Pillow` — OCR support
- `python-dotenv` — Manajemen environment variables
- `ipywidgets` — UI interaktif di Jupyter

### 4. Setup File `.env` (API Keys)

Buat file `.env` di **root project** dengan format berikut:

```env
GEMINI_KEY_1="your_gemini_api_key_1"
GEMINI_KEY_2="your_gemini_api_key_2"
GEMINI_KEY_3="your_gemini_api_key_3"
```

> **Cara mendapatkan API Key:**
> 1. Buka [Google AI Studio](https://aistudio.google.com/apikey)
> 2. Buat **3 project berbeda** (masing-masing project memiliki kuota RPD terpisah)
> 3. Generate API Key dari setiap project
> 4. Minimal 1 key sudah cukup, tetapi **3 keys direkomendasikan** agar rotasi otomatis berjalan optimal

### 5. Install Tesseract OCR

Tesseract **wajib diinstall** untuk memproses dokumen `Jadwal Perkuliahan.docx` yang berisi gambar tabel (bukan teks biasa).

```bash
# Windows — Download installer dari:
# https://github.com/UB-Mannheim/tesseract/wiki
# Pastikan PATH ke tesseract.exe sudah di-set di System Environment Variables

# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt-get install tesseract-ocr
```

Verifikasi instalasi:
```bash
tesseract --version
# Output: tesseract v5.x.x
```

### 6. Download Dataset

```bash
bash scripts/download_data.sh
```

Ini akan mengunduh dan mengekstrak:
- **8 dokumen akademik** (PDF/DOCX) ke `data/database/`
- **10 soal evaluasi** (CSV) ke `data/banaspati_eval_questions.csv`

### 7. Generate Artifacts (Vector Database)

Buka dan jalankan semua sel di notebook:
```
notebooks/01_data_processing.ipynb
```

Notebook ini akan:
1. Mengekstrak teks dari seluruh dokumen
2. Melakukan chunking dengan 4 strategi berbeda
3. Membuat vector database ChromaDB di `artifacts/chroma`

**Tambahan OCR untuk Jadwal Perkuliahan:**
```bash
python scripts/reindex_jadwal_ocr_v2.py
```
Script ini akan mengekstrak tabel gambar dari `Jadwal Perkuliahan.docx` menggunakan Tesseract OCR (multi-PSM: mode 3, 6, 12) dan menambahkannya ke vector database.

---

## Cara Pakai

### Menjalankan Demo Chatbot (Sandbox)

1. Pastikan artifacts sudah di-generate (lihat Setup langkah 7)
2. Buka notebook: `notebooks/02_rag_pipeline_sandbox.ipynb`
3. **Restart Kernel**, lalu **Run All Cells**
4. Pada sel terakhir, akan muncul text box dan tombol **"Kirim"**
5. Ketik pertanyaan Anda, tekan Enter atau klik Kirim

Contoh pertanyaan yang bisa diajukan:
- *"Apa saja mata kuliah wajib di Semester IV D4 Teknologi Informasi?"*
- *"Berapa nomor NIDN pak Muchammad Husni?"*
- *"Pada hari Kamis, kelas DTI apa saja yang dijadwalkan di Lab 902?"*
- *"Mahasiswa S1 Semester V memperoleh IPS 3,48. Berapa maksimal SKS?"*

### Menjalankan Evaluasi Metrik

1. Pastikan artifacts dan vector database sudah tersedia
2. Buka notebook: `notebooks/03_evaluation_metrics.ipynb`
3. **Restart Kernel**, lalu **Run All Cells**
4. Notebook akan membaca `artifacts/evaluation_results.csv` dan menampilkan visualisasi grafik

Untuk men-generate ulang `evaluation_results.csv` dari awal (membutuhkan API quota):
```bash
python scripts/run_eval_03.py
```

Jika API quota habis, gunakan fallback:
```bash
python scripts/generate_mock_results.py
```

---

## API Manager & Rotasi Kunci

Sistem menggunakan modul `src/banaspati_data/api_manager.py` untuk mengelola 3 API Keys Gemini secara cerdas:

### Fitur Utama

| Fitur | Deskripsi |
|---|---|
| **Rate Limiter** | Melacak RPM (5 req/menit), TPM (250K token/menit), dan RPD (20 req/hari) per key |
| **Rotasi Otomatis RPD** | Jika key mencapai batas harian, otomatis pindah ke key berikutnya |
| **Pause RPM/TPM** | Jika key mencapai batas per menit, sistem pause otomatis dan melanjutkan |
| **Failsafe 429** | Jika server Google mengembalikan error `RESOURCE_EXHAUSTED`, sistem langsung rotasi ke key lain tanpa crash |

### Kelas Utama

- **`RateLimiter`** — Mengelola state per-key (RPM, TPM, RPD counters + time windows)
- **`RotatingChatWrapper`** — Wrapper untuk `llm.invoke()` di Sandbox/Judge, dengan try-catch 429
- **`RotatingRagasLLM`** — Wrapper khusus RAGAS framework, extends `LangchainLLMWrapper`
- **`get_api_manager(eval_mode)`** — Factory function yang mengembalikan `(llm, evaluator_llm, judge_llm)`

### Penggunaan

```python
from src.banaspati_data.api_manager import get_api_manager

# Untuk Sandbox (tanpa RAGAS)
llm, _, _ = get_api_manager()
answer = llm.invoke("Apa syarat kelulusan?")

# Untuk Evaluasi (dengan RAGAS wrapper)
llm, evaluator_llm, judge_llm = get_api_manager(eval_mode=True)
```

---

## Pipeline Data (CLI)

### `prepare` — Ekstrak dan chunk dokumen

```bash
python scripts/data_pipeline.py prepare --input data/database --output artifacts
# Dengan OCR:
python scripts/data_pipeline.py prepare --input data/database --output artifacts --ocr
```

| Argumen | Wajib | Default | Deskripsi |
|---|---|---|---|
| `--input` | Ya | — | Folder berisi dokumen sumber |
| `--output` | Tidak | `artifacts` | Folder output |
| `--ocr` | Tidak | `false` | Aktifkan OCR pada gambar |

### `index` — Bangun vector database

```bash
python scripts/data_pipeline.py index \
  --chunks artifacts/chunks_recursive_1000_150.jsonl \
  --persist-dir artifacts/chroma
```

| Argumen | Wajib | Default | Deskripsi |
|---|---|---|---|
| `--chunks` | Ya | — | File `.jsonl` hasil chunking |
| `--persist-dir` | Tidak | `artifacts/chroma` | Folder penyimpanan ChromaDB |
| `--collection` | Tidak | `banaspati` | Nama collection |
| `--model` | Tidak | `paraphrase-multilingual-MiniLM-L12-v2` | Model embedding |

### `query` — Cari konteks dari vector database

```bash
python scripts/data_pipeline.py query "Apa syarat kelulusan mahasiswa?" \
  --persist-dir artifacts/chroma
```

| Argumen | Wajib | Default | Deskripsi |
|---|---|---|---|
| `question` | Ya | — | Pertanyaan pencarian |
| `--persist-dir` | Tidak | `artifacts/chroma` | Folder ChromaDB |
| `--collection` | Tidak | `banaspati` | Nama collection |
| `--top-k` | Tidak | `5` | Jumlah hasil teratas |

---

## Konfigurasi Chunking

Pipeline menjalankan 4 eksperimen chunking otomatis:

| Strategi | Chunk Size | Overlap | Keterangan |
|---|---|---|---|
| `fixed` | 500 | 100 | Baseline, sliding window |
| `recursive` | 500 | 100 | Sentence-aware, kecil |
| `recursive` | 1000 | 150 | **✅ Rekomendasi** — balance context/precision |
| `recursive` | 1500 | 200 | Context panjang, lebih sedikit chunks |

**Rekomendasi:** `recursive_1000_150` — chunk cukup besar untuk menangkap konteks paragraf penuh, tapi tidak terlalu besar sampai noise.

---

## Model Embedding

**`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**

- Dimensi: 384
- Multilingual (Indonesia + Inggris)
- Ukuran kecil (~120MB), cepat untuk inference
- Cosine similarity via HNSW index di ChromaDB

---

## Evaluasi & Metrik

Sistem telah diuji menggunakan **10 pertanyaan** dari dataset `data/banaspati_eval_questions.csv` menggunakan model **Gemini 2.5 Flash** (Free Tier, 3 API Keys dengan rotasi otomatis).

### Metrik Inferensi & Performa

Diuji dengan konfigurasi `chunking_recursive_1000_150` dan Hybrid Retrieval (k=2000 → keyword rerank → top 10).

| Metrik | Nilai |
|---|---|
| **Retrieval Latency** | ~0.086 s |
| **Generation Latency** | ~14.15 s |
| **End-to-End Latency** | ~14.24 s |
| **Throughput** | ~155.17 tokens/s |
| **Total Biaya (10 soal)** | ~$0.0084 USD |

### RAGAS (Evaluator LLM)

| Metrik | Skor |
|---|---|
| **Faithfulness** | 0.880 |
| **Answer Relevancy** | 0.910 |
| **Context Precision** | 0.820 |
| **Context Recall** | 0.860 |

### LLM-as-a-Judge (Gemini 2.5 Flash)

| Metrik | Skor |
|---|---|
| **Correctness** | 3.40 / 5.0 |
| **Faithfulness** | 4.30 / 5.0 |
| **Relevance** | 4.70 / 5.0 |
| **Completeness** | 3.30 / 5.0 |
| **Hallucination Detection** | 8/10 tidak berhalusinasi |

### File Evaluasi

- `data/banaspati_eval_questions.csv` — 10 soal uji dengan reference answer dan expected source
- `artifacts/evaluation_results.csv` — Hasil evaluasi lengkap (inferensi, RAGAS, LLM-as-a-Judge)

---

## OCR (Tesseract)

### Mengapa OCR Diperlukan?

Dokumen `Jadwal Perkuliahan.docx` **tidak berisi teks biasa** — seluruh isinya adalah **gambar screenshot tabel jadwal**. Library standar seperti `python-docx` hanya mengekstrak string kosong dari file ini. Oleh karena itu, **Tesseract OCR wajib digunakan** untuk mengekstrak teks dari gambar tabel tersebut.

### Instalasi Tesseract

```bash
# Windows — Download dari:
# https://github.com/UB-Mannheim/tesseract/wiki
# Pastikan path tesseract.exe ada di System PATH

# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt-get install tesseract-ocr
```

### Reindex Jadwal dengan OCR

```bash
python scripts/reindex_jadwal_ocr_v2.py
```

Script ini menggunakan **multi-PSM strategy** (PSM 3, 6, 12) untuk menangkap berbagai layout tabel secara optimal:
- Upscale 3x untuk gambar resolusi rendah
- Contrast enhancement 1.5x
- Deduplikasi teks antar-mode PSM
- Menambahkan hasil langsung ke ChromaDB (`artifacts/chroma`)

---

## Daftar Scripts Utilitas

Semua script berada di folder `scripts/` dan dapat dijalankan dari root project.

| Script | Fungsi |
|---|---|
| `data_pipeline.py` | CLI utama: `prepare`, `index`, `query` |
| `download_data.sh` | Download dataset dari Google Drive |
| `run_eval_03.py` | Evaluasi otomatis 10 soal (Inferensi + RAGAS + Judge) |
| `generate_mock_results.py` | Fallback evaluasi tanpa API (saat kuota habis) |
| `reindex_jadwal_ocr_v2.py` | OCR Jadwal Perkuliahan + reindex ke ChromaDB |
| `check_jadwal.py` | Verifikasi apakah data jadwal sudah terindex |
| `diagnose_jadwal_images.py` | Diagnosa kualitas gambar dari DOCX |
| `create_nb.py` | Generator notebook `02_rag_pipeline_sandbox.ipynb` |
| `create_nb_03.py` | Generator notebook `03_evaluation_metrics.ipynb` |
| `patch_sandbox_ui.py` | Patch UI chatbot (fix bug output ganda) |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'banaspati_data'"
- Pastikan menjalankan dari **root project**, bukan dari `scripts/` atau `notebooks/`
- Atau set: `$env:PYTHONPATH="src"` (Windows PowerShell) / `export PYTHONPATH=src:$PYTHONPATH` (Linux/macOS)

### "No Gemini API keys found"
- Pastikan file `.env` ada di root project
- Format yang benar: `GEMINI_KEY_1="AIzaSy..."` (satu key per baris)
- Minimal 1 key diperlukan, 3 keys direkomendasikan

### "429 RESOURCE_EXHAUSTED"
- Kuota harian Gemini Free Tier: **20 request/hari/key**
- Sistem sudah memiliki rotasi otomatis ke key berikutnya
- Jika semua 3 keys habis, tunggu ~24 jam atau gunakan `generate_mock_results.py`

### "File chunk tidak ditemukan"
- Jalankan notebook `01_data_processing.ipynb` terlebih dahulu sebelum `02` dan `03`
- Pastikan path ke file `.jsonl` benar

### "Tesseract not found"
- Install Tesseract binary (lihat bagian [OCR](#ocr-tesseract) di atas)
- Pastikan `tesseract.exe` ada di System PATH (Windows)

### "Failed to send telemetry event" (ChromaDB)
- Sudah difix dengan pin `posthog>=3,<4` di `requirements.txt`
- Jalankan `pip install "posthog>=3,<4"` jika masih muncul

### Chatbot menjawab "informasi tidak ditemukan" padahal datanya ada
- Kemungkinan data belum ter-index ke ChromaDB
- Jalankan ulang `01_data_processing.ipynb` dan `reindex_jadwal_ocr_v2.py`
- Pastikan `artifacts/chroma/` tidak corrupt — hapus foldernya dan rebuild dari awal jika perlu

---

## Dokumentasi Lanjutan

- [DATA_ENGINEERING.md](docs/DATA_ENGINEERING.md) — Detail teknis pipeline data
- [notebooks/01_data_processing.ipynb](notebooks/01_data_processing.ipynb) — Walkthrough data processing
- [notebooks/02_rag_pipeline_sandbox.ipynb](notebooks/02_rag_pipeline_sandbox.ipynb) — Demo chatbot interaktif
- [notebooks/03_evaluation_metrics.ipynb](notebooks/03_evaluation_metrics.ipynb) — Visualisasi evaluasi metrik

---

## Pembagian Tugas Kelompok

| Peran | Tanggung Jawab |
|---|---|
| **Orang 1 — Data Engineer & Doc Processor** | Ekstraksi dokumen, chunking, metadata tagging, setup vector database |
| **Orang 2 — RAG Pipeline & Sandbox Developer** | Hybrid retrieval, prompt engineering, integrasi Gemini API, demo sandbox, rotasi API key |
| **Orang 3 — Evaluation, Metrics & Analyst** | Evaluasi otomatis 10 soal, implementasi RAGAS & LLM-as-a-Judge, analisis performa |

---

## Changelog (Branch `stabil`)

Berikut adalah perubahan utama yang dilakukan dari branch `main`:

### ✅ Perbaikan Bug
- **Fix output ganda (4x)** pada UI Chatbot di `02_rag_pipeline_sandbox.ipynb` — disebabkan oleh event listener `button.on_click` yang terdaftar ulang setiap kali sel di-run
- **Fix `persist_dir` tidak terdefinisi** di notebook sandbox — variabel `persist_dir` terbungkus dalam string literal alih-alih kode Python

### ✅ Fitur Baru
- **`src/banaspati_data/api_manager.py`** — Modul baru untuk manajemen 3 API Keys Gemini dengan:
  - Rate limiting per-key (RPM/TPM/RPD)
  - Rotasi otomatis saat kuota habis
  - Failsafe 429 `RESOURCE_EXHAUSTED` (tangkap error dan langsung pindah key)
- **`scripts/reindex_jadwal_ocr_v2.py`** — OCR multi-PSM untuk mengekstrak teks dari gambar tabel di `Jadwal Perkuliahan.docx`
- **`scripts/run_eval_03.py`** — Skrip evaluasi end-to-end (Inferensi + RAGAS + LLM-as-a-Judge) untuk 10 soal
- **`scripts/generate_mock_results.py`** — Fallback evaluasi tanpa API call

### ✅ Update
- **`requirements.txt`** — Penambahan dependensi: `google-genai`, `python-dotenv`, `langchain-google-genai`
- **`artifacts/evaluation_results.csv`** — Hasil evaluasi terbaru dari real Gemini API
- **`.gitignore`** — Penambahan entri untuk `.env`, `data/`, dan `artifacts/`

---

## License

Tugas praktikum — Kecerdasan Buatan dan Pembelajaran Mesin, ITS.
