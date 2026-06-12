import nbformat as nbf
nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell('# RAG Pipeline & Sandbox BANASPATI (Orang 2)\n\nNotebook ini mengimplementasikan sistem retrieval dan LLM generation dengan model open-weight <9B menggunakan HuggingFace API.'))

nb.cells.append(nbf.v4.new_code_cell('''import os, warnings
warnings.filterwarnings('ignore')

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import ipywidgets as widgets
from IPython.display import display, Markdown

# Token HuggingFace dari Orang 2
os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'MASUKKAN_HF_TOKEN_ANDA_DISINI'
'''))

nb.cells.append(nbf.v4.new_markdown_cell('## 1. Load Vector Database (ChromaDB)'))

nb.cells.append(nbf.v4.new_code_cell('''embed_model = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
vectorstore = Chroma(persist_directory='../artifacts', collection_name='banaspati_docs', embedding_function=embed_model)
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
'''))

nb.cells.append(nbf.v4.new_markdown_cell('## 2. Setup Generator LLM (<9B Parameter)'))

nb.cells.append(nbf.v4.new_code_cell('''llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    max_new_tokens=512,
    temperature=0.1,
    repetition_penalty=1.1,
    return_full_text=False
)
'''))

nb.cells.append(nbf.v4.new_markdown_cell('## 3. Prompt Engineering (Anti-Halusinasi)'))

nb.cells.append(nbf.v4.new_code_cell('''template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Anda adalah BANASPATI (Bubur Panas Personal Assistant), asisten akademik yang ahli.
Tugas Anda adalah menjawab pertanyaan pengguna BERDASARKAN KONTEKS yang diberikan saja.
Jika informasi tidak ada di dalam konteks, Anda WAJIB menjawab dengan persis kalimat: "informasi tidak ditemukan".
Jangan pernah mengarang jawaban atau menggunakan pengetahuan dari luar dokumen.

Konteks Dokumen:
{context}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
'''))

nb.cells.append(nbf.v4.new_markdown_cell('## 4. Fungsi Utama dan Sandbox UI'))

nb.cells.append(nbf.v4.new_code_cell('''def banaspati_answer(question):
    docs = retriever.invoke(question)
    context_text = "\\n\\n".join([f"[Sumber: {d.metadata.get('source_file', 'Unknown')}, Hal: {d.metadata.get('page', 'Unknown')}]\\n{d.page_content}" for d in docs])
    
    print("="*50)
    print("DOKUMEN REFERENSI YANG DI-RETRIEVE:")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc.metadata.get('source_file', '')} (Hal {doc.metadata.get('page', '')})")
        print(doc.page_content[:200] + "...")
    print("="*50 + "\\n")
    
    final_prompt = prompt.format(context=context_text, question=question)
    answer = llm.invoke(final_prompt)
    return answer

# Sandbox Widget
out = widgets.Output()
text_input = widgets.Text(description="Tanya:", layout=widgets.Layout(width="80%"))
button = widgets.Button(description="Kirim", button_style="success")

def on_submit(_):
    with out:
        out.clear_output()
        q = text_input.value
        display(Markdown(f"**Pertanyaan:** {q}"))
        ans = banaspati_answer(q)
        display(Markdown(f"**BANASPATI:** {ans}"))

button.on_click(on_submit)
display(widgets.HBox([text_input, button]), out)
'''))

with open('notebooks/02_rag_pipeline_sandbox.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
