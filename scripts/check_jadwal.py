from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings('ignore')

embed_model = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
vectorstore = Chroma(persist_directory='artifacts/chroma', collection_name='banaspati', embedding_function=embed_model)

print("=== QUERY: jadwal perkuliahan hari senin ===")
docs = vectorstore.similarity_search('jadwal perkuliahan hari senin mata kuliah', k=8)
for i, doc in enumerate(docs):
    src = doc.metadata.get('source_file', '?')
    print(f'\n--- Doc {i+1} ({src}) ---')
    print(doc.page_content[:400])

print("\n\n=== ALL Jadwal Perkuliahan CHUNKS ===")
all_docs = vectorstore.similarity_search('JADWAL PERKULIAHAN', k=50)
jadwal_docs = [d for d in all_docs if 'Jadwal' in d.metadata.get('source_file', '')]
for i, doc in enumerate(jadwal_docs):
    print(f'\n--- Jadwal Chunk {i+1} ---')
    print(doc.page_content[:600])
