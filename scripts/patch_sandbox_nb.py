import json

file_path = "notebooks/02_rag_pipeline_sandbox.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        
        # 1. Modify cell edcfea24 (API Keys initialization)
        if cell.get("id") == "edcfea24" or any("KEY_1 = os.environ.get" in line for line in source):
            new_source = []
            for line in source:
                if "KEY_1 =" in line or "KEY_2 =" in line or "# API Keys" in line:
                    pass # remove old api key stuff
                else:
                    new_source.append(line)
            # Add sys.path modification to import src
            new_source.append("\nimport sys\n")
            new_source.append("if '../' not in sys.path:\n")
            new_source.append("    sys.path.append('../')\n")
            cell["source"] = new_source
            
        # 2. Modify cell 6e252a7d (LLM instantiation)
        elif cell.get("id") == "6e252a7d" or any("llm1 =" in line for line in source):
            new_source = [
                "from src.banaspati_data.api_manager import get_api_manager\n",
                "llm, _, _ = get_api_manager()\n"
            ]
            cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Sandbox notebook successfully patched to use api_manager.py")
