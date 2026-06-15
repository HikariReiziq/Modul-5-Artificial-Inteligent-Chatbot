import json

file_path = "notebooks/02_rag_pipeline_sandbox.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        for i, line in enumerate(source):
            if "persist_dir =" in line and "artifacts/chroma'" in line:
                source[i] = "    \"persist_dir = '../artifacts/chroma_new' if os.path.basename(os.getcwd()) == 'notebooks' else 'artifacts/chroma_new'\\n\",\n"

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Sandbox notebook successfully patched to use chroma_new")
