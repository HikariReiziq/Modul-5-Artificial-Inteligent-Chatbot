# Data Engineering BANASPATI

## Tujuan dan Batasan

Pipeline ini mengubah database dokumen akademik yang heterogen menjadi chunk
bermetadata dan vector database yang siap dipakai oleh pipeline RAG. Jawaban
tidak dibuat pada tahap ini; keluaran utama adalah konteks yang dapat
ditelusuri kembali ke sumbernya.

Format yang didukung adalah PDF, DOCX, TXT, dan Markdown. Untuk PDF dan DOCX,
pipeline mengekstrak teks, tabel, serta keberadaan gambar. Teks di dalam gambar
dapat diekstrak menggunakan OCR opsional.

## Keputusan Teknis

### Ekstraksi multimodal

- **PyMuPDF** dipilih untuk PDF karena dapat mengambil teks per halaman, tabel,
  gambar, dan nomor halaman.
- **python-docx** dipilih untuk DOCX karena dapat mengambil paragraf, heading,
  tabel, dan media tertanam.
- Gambar tanpa OCR tetap direkam sebagai unit konten. Ini mencegah modalitas
  gambar hilang dari audit, tetapi tidak dimasukkan ke vector database karena
  placeholder gambar tidak memberi informasi semantik untuk retrieval.
- OCR dibuat opsional karena hasil dan latensinya bergantung kualitas scan dan
  instalasi Tesseract. Sebelum OCR, gambar kecil diperbesar dan kontrasnya
  ditingkatkan agar teks tabel/jadwal lebih mudah terbaca.

### Metadata

Setiap unit dan chunk menyimpan `source_file`, `file_name`, `file_type`,
`page`, `section`, dan `content_type`. Chunk juga menyimpan `parent_id`,
`chunk_index`, `chunk_strategy`, `chunk_size`, serta `chunk_overlap`.
Metadata ini memungkinkan jawaban menampilkan sumber dan halaman.

### Eksperimen chunking

Pipeline membandingkan:

| Strategi | Ukuran | Overlap | Tujuan |
|---|---:|---:|---|
| fixed | 500 | 100 | baseline sederhana |
| recursive | 500 | 100 | menguji chunk kecil berbasis batas kalimat |
| recursive | 1000 | 150 | kandidat utama, seimbang antara konteks dan presisi |
| recursive | 1500 | 200 | menguji konteks lebih panjang |

`recursive_1000_150` menjadi rekomendasi awal, bukan klaim konfigurasi final.
Konfigurasi terbaik harus dipilih berdasarkan retrieval evaluation pada
pertanyaan evaluasi, misalnya hit-rate/recall@k dan context precision.

### Embedding dan vector database

Model default adalah `paraphrase-multilingual-MiniLM-L12-v2` karena mendukung
bahasa Indonesia, ringan, terbuka, dan cocok untuk semantic retrieval.
ChromaDB digunakan karena persisten, menyimpan metadata, dan menyediakan
cosine similarity melalui indeks HNSW.

## Artefak

Perintah `prepare` menghasilkan:

- `extracted_documents.jsonl`: hasil ekstraksi sebelum chunking;
- `images/`: gambar asli yang diekstrak dari PDF/DOCX;
- `chunks_<strategy>_<size>_<overlap>.jsonl`: seluruh kandidat chunk;
- `chunking_experiment.json`: ringkasan statistik eksperimen.

Setiap baris JSONL memiliki struktur:

```json
{
  "id": "stable-sha1-id",
  "text": "isi konteks",
  "metadata": {
    "source_file": "kurikulum/kurikulum.pdf",
    "page": 12,
    "content_type": "text",
    "chunk_index": 0
  }
}
```

## Cara Menjalankan dan Menginterpretasi

```bash
python scripts/data_pipeline.py prepare --input data/database --output artifacts --ocr
python scripts/data_pipeline.py index --chunks artifacts/chunks_recursive_1000_150.jsonl
python scripts/data_pipeline.py query "pertanyaan uji"
```

Periksa bahwa hasil query relevan, `source_file` dan `page` tersedia, serta
jarak cosine hasil relevan lebih kecil daripada hasil tidak relevan. Setelah
dataset asli tersedia, gunakan minimal 10 pertanyaan evaluasi untuk menentukan
konfigurasi chunking final.

## Limitasi

- Tabel PDF yang kompleks dapat kehilangan struktur merge-cell.
- Gambar tanpa teks tidak memiliki caption semantik; peningkatan berikutnya
  adalah image captioning dengan vision model maksimal 9B.
- Pemotongan berbasis karakter tidak identik dengan token model generatif.
- Statistik chunking belum cukup untuk memilih konfigurasi terbaik tanpa
  evaluasi retrieval pada pertanyaan dan ground truth.
