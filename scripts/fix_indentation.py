import json

# Fix 1: Perbaiki IndentationError - persist_dir yang salah format
file_path = "notebooks/02_rag_pipeline_sandbox.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        for i, line in enumerate(cell["source"]):
            # Hapus leading whitespace '    ' yang salah pada baris persist_dir
            if line.startswith("    \"persist_dir =") and "chroma" in line:
                # Tulis ulang baris dengan benar (tanpa indentasi ekstra)
                new_source.append("\"persist_dir = '../artifacts/chroma_new' if os.path.basename(os.getcwd()) == 'notebooks' else 'artifacts/chroma_new'\\n\",\n")
            else:
                new_source.append(line)
        cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Fixed: IndentationError in 02_rag_pipeline_sandbox.ipynb")
print("persist_dir now correctly points to 'artifacts/chroma_new'")
