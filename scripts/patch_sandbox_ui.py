import json
import os

NOTEBOOK_PATH = "notebooks/02_rag_pipeline_sandbox.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_ui_cell_source = [
    "# === SANDBOX UI CHATBOT ===\n",
    "import ipywidgets as widgets\n",
    "from IPython.display import display, Markdown, clear_output\n",
    "\n",
    "# Gunakan global state yang aman untuk mencegah multiple bindings\n",
    "if 'banaspati_ui_initialized' not in globals():\n",
    "    out = widgets.Output()\n",
    "    text_input = widgets.Text(description=\"Tanya:\", layout=widgets.Layout(width=\"80%\"))\n",
    "    button = widgets.Button(description=\"Kirim\", button_style=\"success\")\n",
    "    \n",
    "    def on_submit(_):\n",
    "        if button.disabled:\n",
    "            return\n",
    "        button.disabled = True\n",
    "        button.description = \"Loading...\"\n",
    "        try:\n",
    "            with out:\n",
    "                out.clear_output()\n",
    "                q = text_input.value.strip()\n",
    "                if not q: return\n",
    "                display(Markdown(f\"**Pertanyaan:** {q}\"))\n",
    "                ans = banaspati_answer(q)\n",
    "                display(Markdown(f\"**BANASPATI:** {ans}\"))\n",
    "        finally:\n",
    "            button.disabled = False\n",
    "            button.description = \"Kirim\"\n",
    "            text_input.value = \"\" # Auto clear input\n",
    "\n",
    "    button.on_click(on_submit)\n",
    "    text_input.on_submit(on_submit) # Dukung enter key\n",
    "    banaspati_ui_initialized = True\n",
    "\n",
    "# Tampilkan UI\n",
    "clear_output(wait=True)\n",
    "display(widgets.VBox([widgets.HBox([text_input, button]), out]))\n"
]

# Find and replace the UI cell
# We'll look for the cell containing 'text_input = widgets.Text'
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source_str = "".join(cell["source"])
        if "button.on_click(on_submit)" in source_str and "def banaspati_answer" in source_str:
            # We need to split the cell into two! One for banaspati_answer, one for UI
            # Let's just find the split point.
            # Actually, let's just replace the sandbox widget part of the string.
            lines = cell["source"]
            new_lines = []
            for line in lines:
                if line.startswith("# Sandbox Widget") or line.startswith("out = widgets.Output()"):
                    break
                new_lines.append(line)
            
            # Now append the new UI code
            new_lines.extend([line for line in new_ui_cell_source])
            cell["source"] = new_lines
            print(f"Updated cell {i}")
            break

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
