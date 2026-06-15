"""
Script untuk mengekstrak gambar dari Jadwal Perkuliahan.docx menggunakan
Tesseract OCR, lalu mengindeks hasilnya ke dalam ChromaDB.

Catatan: File .docx ini berisi gambar-gambar jadwal (screenshot/foto),
sehingga diperlukan OCR untuk membaca teksnya.
"""
import sys
import io
sys.path.insert(0, '.')

from docx import Document
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import warnings
warnings.filterwarnings('ignore')

# Path ke Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

JADWAL_PATH = Path("data/database/Jadwal Perkuliahan.docx")


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocessing gambar untuk meningkatkan akurasi OCR.
    """
    # Konversi ke grayscale
    image = image.convert("L")
    
    # Perbesar jika terlalu kecil
    if max(image.size) < 2000:
        scale = 3
        image = image.resize(
            (image.width * scale, image.height * scale),
            Image.Resampling.LANCZOS
        )
    elif max(image.size) < 3000:
        scale = 2
        image = image.resize(
            (image.width * scale, image.height * scale),
            Image.Resampling.LANCZOS
        )
    
    # Tingkatkan kontras
    image = ImageEnhance.Contrast(image).enhance(2.0)
    
    # Sharpening
    image = image.filter(ImageFilter.SHARPEN)
    
    return image


def extract_images_from_docx(path: Path):
    """
    Ekstrak semua gambar dari file .docx.
    Returns: list of (image_bytes, image_index)
    """
    doc = Document(path)
    images = []
    
    for idx, rel in enumerate(doc.part.rels.values(), start=1):
        if "image" in rel.reltype:
            image_bytes = rel.target_part.blob
            images.append((image_bytes, idx))
    
    return images


def ocr_image(image_bytes: bytes, image_idx: int) -> str:
    """
    Jalankan OCR pada gambar dan kembalikan teksnya.
    """
    image = Image.open(io.BytesIO(image_bytes))
    processed = preprocess_image(image)
    
    # OCR config: --psm 6 = Assume uniform block of text
    # --oem 3 = LSTM engine (terbaik untuk teks campuran)
    config = "--psm 6 --oem 3"
    
    try:
        text = pytesseract.image_to_string(processed, lang="eng", config=config)
        text = text.strip()
        return text
    except Exception as e:
        print(f"  Warning: OCR gagal untuk gambar {image_idx}: {e}")
        return ""


def main():
    print("=" * 60)
    print("Re-indexing Jadwal Perkuliahan.docx (OCR Mode)")
    print("=" * 60)
    
    # Ekstrak gambar dari docx
    print(f"\n[1/4] Mengekstrak gambar dari: {JADWAL_PATH}")
    images = extract_images_from_docx(JADWAL_PATH)
    print(f"      Ditemukan {len(images)} gambar.")
    
    if not images:
        print("ERROR: Tidak ada gambar ditemukan di file .docx!")
        return
    
    # Jalankan OCR pada setiap gambar
    print("\n[2/4] Menjalankan OCR pada setiap gambar...")
    ocr_results = []
    for image_bytes, idx in images:
        print(f"      OCR gambar {idx}/{len(images)}...", end=" ", flush=True)
        text = ocr_image(image_bytes, idx)
        if text and len(text) > 20:
            ocr_results.append({
                "text": text,
                "image_index": idx,
                "char_count": len(text)
            })
            print(f"OK ({len(text)} karakter)")
        else:
            print(f"SKIP (terlalu sedikit teks: '{text[:50] if text else '(kosong)'}')")
    
    print(f"\n      Total {len(ocr_results)} gambar berhasil di-OCR dari {len(images)} gambar.")
    
    if not ocr_results:
        print("ERROR: Semua gambar gagal di-OCR!")
        return
    
    # Tampilkan sample
    print("\n=== SAMPLE OCR OUTPUT (Gambar pertama) ===")
    if ocr_results:
        print(ocr_results[0]["text"][:500])
    
    # Load ChromaDB
    print("\n[3/4] Menghubungkan ke ChromaDB & menyiapkan embedding...")
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document as LCDocument
    
    embed_model = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
    vectorstore = Chroma(
        persist_directory='artifacts/chroma',
        collection_name='banaspati',
        embedding_function=embed_model
    )
    
    # Hapus chunks Jadwal Perkuliahan lama
    print("      Menghapus chunks Jadwal Perkuliahan lama...")
    try:
        old_results = vectorstore.get(where={"source_file": "Jadwal Perkuliahan.docx"})
        if old_results and old_results['ids']:
            vectorstore.delete(ids=old_results['ids'])
            print(f"      Dihapus {len(old_results['ids'])} chunks lama.")
        else:
            print("      Tidak ada chunks lama ditemukan.")
    except Exception as e:
        print(f"      Warning: {e}")
    
    # Tambahkan hasil OCR ke ChromaDB
    print(f"\n[4/4] Menambahkan {len(ocr_results)} chunks OCR ke ChromaDB...")
    docs_to_add = []
    for result in ocr_results:
        docs_to_add.append(LCDocument(
            page_content=result["text"],
            metadata={
                "source_file": "Jadwal Perkuliahan.docx",
                "file_name": "Jadwal Perkuliahan.docx",
                "file_type": "docx",
                "content_type": "image_ocr",
                "image_index": result["image_index"]
            }
        ))
    
    vectorstore.add_documents(docs_to_add)
    print(f"      Selesai! {len(docs_to_add)} chunks berhasil diindeks.")
    
    # Verifikasi retrieval
    print("\n" + "=" * 60)
    print("VERIFIKASI RETRIEVAL")
    print("=" * 60)
    test_queries = [
        "mata kuliah hari senin",
        "jadwal kuliah selasa rabu",
        "ruangan kuliah semester"
    ]
    for q in test_queries:
        docs = vectorstore.similarity_search(q, k=5)
        jadwal_hits = [d for d in docs if 'Jadwal' in d.metadata.get('source_file', '')]
        print(f"\nQuery '{q}': {len(jadwal_hits)}/5 hasil dari Jadwal Perkuliahan")
        if jadwal_hits:
            print(f"  Preview: {jadwal_hits[0].page_content[:200]}")


if __name__ == "__main__":
    main()
