"""
Diagnostic: simpan gambar dari Jadwal Perkuliahan.docx ke disk
agar bisa diperiksa kualitasnya secara visual, dan coba berbagai
konfigurasi OCR untuk menemukan yang terbaik.
"""
import sys, io
sys.path.insert(0, '.')

from docx import Document
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

JADWAL_PATH = Path("data/database/Jadwal Perkuliahan.docx")
OUT_DIR = Path("artifacts/jadwal_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

doc = Document(JADWAL_PATH)
images = [(rel.target_part.blob, idx) for idx, rel in enumerate(doc.part.rels.values(), 1) if "image" in rel.reltype]

print(f"Ditemukan {len(images)} gambar")

for image_bytes, idx in images:
    image = Image.open(io.BytesIO(image_bytes))
    print(f"\nGambar {idx}: ukuran={image.size}, mode={image.mode}, bytes={len(image_bytes)}")
    
    # Simpan gambar asli
    out_path = OUT_DIR / f"jadwal_original_{idx}.png"
    image.save(out_path)
    print(f"  Disimpan ke: {out_path}")
    
    # Coba beberapa PSM mode
    gray = image.convert("L")
    # Scale up 3x for better OCR
    scale = 3
    large = gray.resize((gray.width * scale, gray.height * scale), Image.Resampling.LANCZOS)
    large = ImageEnhance.Contrast(large).enhance(2.5)
    
    for psm in [3, 6, 11, 12]:
        config = f"--psm {psm} --oem 3"
        try:
            text = pytesseract.image_to_string(large, lang="eng", config=config)
            text = text.strip()
            print(f"  PSM {psm}: {len(text)} chars | Preview: {repr(text[:120])}")
        except Exception as e:
            print(f"  PSM {psm}: ERROR - {e}")

print("\n\nGambar sudah disimpan di:", OUT_DIR.absolute())
print("Silakan buka folder tersebut untuk memeriksa kualitas gambar secara visual.")
