import json

file_path = "notebooks/02_rag_pipeline_sandbox.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            # Hapus baris yang salah format (mengandung backslash-n yang hardcoded dan kutip ganda di depan)
            if line.startswith('"persist_dir =') and 'chroma' in line:
                # Ganti dengan baris Python yang benar
                new_source.append("persist_dir = '../artifacts/chroma' if os.path.basename(os.getcwd()) == 'notebooks' else 'artifacts/chroma'\n")
            else:
                new_source.append(line)
        cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

# Verifikasi hasil
with open(file_path, "r", encoding="utf-8") as f:
    nb2 = json.load(f)
for cell in nb2["cells"]:
    if cell["cell_type"] == "code":
        for line in cell["source"]:
            if 'persist_dir' in line:
                print("RESULT:", repr(line))
