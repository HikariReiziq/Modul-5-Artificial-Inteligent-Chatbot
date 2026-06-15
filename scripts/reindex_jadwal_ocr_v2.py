"""
Script OCR V2: Menggunakan berbagai Page Segmentation Modes (PSM) Tesseract
untuk memastikan seluruh tata letak jadwal yang rumit terekstrak dengan maksimal.
Hasil teks dari berbagai PSM akan digabung menjadi konteks kaya untuk RAG.
"""
import sys, io
sys.path.insert(0, '.')

from docx import Document
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import warnings
warnings.filterwarnings('ignore')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

JADWAL_PATH = Path("data/database/Jadwal Perkuliahan.docx")


def preprocess_image(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    scale = 3
    large = gray.resize((gray.width * scale, gray.height * scale), Image.Resampling.LANCZOS)
    large = ImageEnhance.Contrast(large).enhance(2.5)
    return large


def extract_images_from_docx(path: Path):
    doc = Document(path)
    return [(rel.target_part.blob, idx) for idx, rel in enumerate(doc.part.rels.values(), 1) if "image" in rel.reltype]


def ocr_image_multi_psm(image_bytes: bytes, image_idx: int) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    processed = preprocess_image(image)
    
    combined_text = []
    # PSM 3: Automatic segmentation
    # PSM 6: Uniform block (bagus untuk baris teks padat)
    # PSM 12: Sparse text with OSD (menangkap teks yang tercecer di berbagai kolom)
    for psm in [3, 6, 12]:
        try:
            text = pytesseract.image_to_string(processed, lang="eng", config=f"--psm {psm} --oem 3")
            combined_text.append(f"--- Ekstraksi Layout V{psm} ---\n{text.strip()}")
        except Exception:
            pass
            
    return "\n\n".join(combined_text)


def main():
    print("Mengekstrak & OCR Jadwal Perkuliahan (Multi-PSM)...")
    images = extract_images_from_docx(JADWAL_PATH)
    
    ocr_results = []
    for image_bytes, idx in images:
        text = ocr_image_multi_psm(image_bytes, idx)
        if text and len(text) > 50:
            # Berikan metadata eksplisit agar lebih mudah dicari LLM
            header = f"[Dokumen: Jadwal Perkuliahan Fakultas, Bagian/Gambar: {idx}]\n"
            # Tambahkan keyword pembantu
            keywords = "Keyword: Senin, Selasa, Rabu, Kamis, Jumat, Jadwal, Kuliah, SKS, Dosen, Ruangan, Kelas\n"
            
            ocr_results.append({
                "text": header + keywords + text,
                "image_index": idx
            })
            print(f"Gambar {idx}: OK ({len(text)} karakter)")
            
    print("\nMenghubungkan ke ChromaDB...")
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document as LCDocument
    
    embed_model = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
    vectorstore = Chroma(
        persist_directory='artifacts/chroma',
        collection_name='banaspati',
        embedding_function=embed_model
    )
    
    print("Menghapus chunks Jadwal Perkuliahan lama...")
    try:
        old_results = vectorstore.get(where={"source_file": "Jadwal Perkuliahan.docx"})
        if old_results and old_results['ids']:
            vectorstore.delete(ids=old_results['ids'])
    except Exception:
        pass
        
    print(f"Menambahkan {len(ocr_results)} chunks komprehensif ke ChromaDB...")
    docs_to_add = [
        LCDocument(
            page_content=res["text"],
            metadata={
                "source_file": "Jadwal Perkuliahan.docx",
                "file_name": "Jadwal Perkuliahan.docx",
                "file_type": "docx",
                "content_type": "image_ocr_multi",
                "image_index": res["image_index"]
            }
        ) for res in ocr_results
    ]
    vectorstore.add_documents(docs_to_add)
    print("Selesai!")

    # Test
    print("\nTest Retrieval: 'mata kuliah hari senin'")
    docs = vectorstore.similarity_search("mata kuliah hari senin", k=5)
    for d in docs:
        if 'Jadwal' in d.metadata.get('source_file', ''):
            print("=> FOUND Jadwal Chunk!")
            print(d.page_content[:200] + "...\n")

if __name__ == "__main__":
    main()
