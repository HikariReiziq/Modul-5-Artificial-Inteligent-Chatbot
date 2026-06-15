import json

# Update 02_rag_pipeline_sandbox.ipynb: chroma_new -> chroma
file_path = "notebooks/02_rag_pipeline_sandbox.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            if "chroma_new" in line:
                line = line.replace("chroma_new", "chroma")
            new_source.append(line)
        cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated 02_rag_pipeline_sandbox.ipynb: chroma_new -> chroma")
