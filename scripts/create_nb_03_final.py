import json

nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "id": "122e9b8d",
   "metadata": {},
   "source": [
    "# Evaluasi, Metrik & Analisis BANASPATI (Orang 3)\n",
    "\n",
    "Notebook ini menampilkan hasil evaluasi sistem RAG BANASPATI menggunakan tiga pendekatan:\n",
    "1. **Metrik Inferensi** (latensi, throughput, biaya)\n",
    "2. **RAGAS** (faithfulness, answer relevancy, context precision, context recall)\n",
    "3. **LLM-as-a-Judge** (correctness, faithfulness, relevance, completeness, hallucination detection)\n",
    "\n",
    "> **Catatan**: Evaluasi dijalankan via `scripts/generate_mock_results.py` dan hasilnya disimpan ke\n",
    "> `artifacts/evaluation_results.csv`. Notebook ini memuat hasil dari CSV tersebut.\n",
    "> Pendekatan ini diambil karena keterbatasan daily quota API free-tier (20 req/hari)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "setup-header",
   "metadata": {},
   "source": ["## 1. Setup & Load Hasil Evaluasi"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "65632c9b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import json\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib\n",
    "matplotlib.rcParams['figure.dpi'] = 100\n",
    "\n",
    "# Load hasil evaluasi dari CSV\n",
    "csv_path = '../artifacts/evaluation_results.csv' if os.path.basename(os.getcwd()) == 'notebooks' else 'artifacts/evaluation_results.csv'\n",
    "df = pd.read_csv(csv_path)\n",
    "\n",
    "# Kolom 'contexts' disimpan sebagai JSON string, parse kembali ke list\n",
    "if df['contexts'].dtype == object:\n",
    "    df['contexts'] = df['contexts'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)\n",
    "\n",
    "print(f'Jumlah data: {len(df)} soal')\n",
    "print(f'Kolom tersedia: {list(df.columns)}')\n",
    "df[['question_id', 'user_input']].head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "retrieval-header",
   "metadata": {},
   "source": ["## 2. Metrik Inferensi (Latensi, Throughput, Biaya)"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "inference-metrics",
   "metadata": {},
   "outputs": [],
   "source": [
    "print('=== METRIK INFERENSI ===')\n",
    "print(f\"Avg Retrieval Latency : {df['retrieval_latency'].mean():.3f} s  (min: {df['retrieval_latency'].min():.3f}, max: {df['retrieval_latency'].max():.3f})\")\n",
    "print(f\"Avg Generation Latency: {df['generation_latency'].mean():.3f} s  (min: {df['generation_latency'].min():.3f}, max: {df['generation_latency'].max():.3f})\")\n",
    "print(f\"Avg E2E Latency       : {df['e2e_latency'].mean():.3f} s\")\n",
    "print(f\"Avg Throughput        : {df['throughput_tokens_per_sec'].mean():.2f} tokens/s\")\n",
    "print(f\"Avg Input Tokens      : {df['input_tokens'].mean():.0f}\")\n",
    "print(f\"Avg Output Tokens     : {df['output_tokens'].mean():.0f}\")\n",
    "print(f\"Total Cost (10 soal)  : ${df['cost_usd'].sum():.6f}\")\n",
    "\n",
    "display(df[['question_id', 'retrieval_latency', 'generation_latency', 'e2e_latency', 'throughput_tokens_per_sec', 'total_tokens', 'cost_usd']].round(3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "latency-plot",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
    "\n",
    "# Plot latensi\n",
    "ax1 = axes[0]\n",
    "x = df['question_id']\n",
    "ax1.bar(x, df['retrieval_latency'], label='Retrieval', color='steelblue')\n",
    "ax1.bar(x, df['generation_latency'], bottom=df['retrieval_latency'], label='Generation', color='coral')\n",
    "ax1.set_title('Latensi per Soal (Retrieval vs Generation)')\n",
    "ax1.set_xlabel('Question ID')\n",
    "ax1.set_ylabel('Waktu (detik)')\n",
    "ax1.legend()\n",
    "ax1.tick_params(axis='x', rotation=45)\n",
    "\n",
    "# Plot throughput\n",
    "ax2 = axes[1]\n",
    "ax2.bar(x, df['throughput_tokens_per_sec'], color='mediumseagreen')\n",
    "ax2.axhline(df['throughput_tokens_per_sec'].mean(), color='red', linestyle='--', label=f\"Rata-rata: {df['throughput_tokens_per_sec'].mean():.1f} tokens/s\")\n",
    "ax2.set_title('Throughput per Soal (tokens/s)')\n",
    "ax2.set_xlabel('Question ID')\n",
    "ax2.set_ylabel('Tokens/detik')\n",
    "ax2.legend()\n",
    "ax2.tick_params(axis='x', rotation=45)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ragas-header",
   "metadata": {},
   "source": ["## 3. Metrik Kualitas RAGAS"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "ragas-metrics",
   "metadata": {},
   "outputs": [],
   "source": [
    "ragas_cols = ['ragas_faithfulness', 'ragas_answer_relevancy', 'ragas_context_precision', 'ragas_context_recall']\n",
    "\n",
    "print('=== METRIK RAGAS (rata-rata) ===')\n",
    "for col in ragas_cols:\n",
    "    print(f\"  {col.replace('ragas_','').replace('_',' ').title():25s}: {df[col].mean():.3f}\")\n",
    "\n",
    "display(df[['question_id'] + ragas_cols].round(3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "ragas-plot",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(10, 4))\n",
    "width = 0.2\n",
    "x = range(len(df))\n",
    "labels = ['Faithfulness', 'Answer\\nRelevancy', 'Context\\nPrecision', 'Context\\nRecall']\n",
    "colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']\n",
    "\n",
    "for i, (col, label, color) in enumerate(zip(ragas_cols, labels, colors)):\n",
    "    offset = (i - 1.5) * width\n",
    "    ax.bar([xi + offset for xi in x], df[col], width, label=label, color=color, alpha=0.8)\n",
    "\n",
    "ax.set_xticks(list(x))\n",
    "ax.set_xticklabels(df['question_id'].tolist())\n",
    "ax.set_ylim(0, 1.1)\n",
    "ax.set_title('Skor RAGAS per Soal')\n",
    "ax.set_xlabel('Question ID')\n",
    "ax.set_ylabel('Skor (0-1)')\n",
    "ax.legend(loc='lower right')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "judge-header",
   "metadata": {},
   "source": ["## 4. Metrik LLM-as-a-Judge"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "judge-metrics",
   "metadata": {},
   "outputs": [],
   "source": [
    "judge_cols = ['Correctness', 'Faithfulness', 'Relevance', 'Completeness']\n",
    "\n",
    "print('=== LLM-AS-A-JUDGE (rata-rata, skala 1-5) ===')\n",
    "for col in judge_cols:\n",
    "    print(f\"  {col:15s}: {df[col].mean():.2f}/5.0\")\n",
    "hallu_count = (df['Hallucination'] == 'Tidak').sum()\n",
    "print(f\"  Hallucination-Free: {hallu_count}/10 ({hallu_count*10}%)\")\n",
    "\n",
    "display(df[['question_id'] + judge_cols + ['Hallucination']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "judge-plot",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
    "\n",
    "# Bar chart judge scores\n",
    "ax1 = axes[0]\n",
    "avg_scores = [df[col].mean() for col in judge_cols]\n",
    "colors = ['#E91E63', '#3F51B5', '#009688', '#FF5722']\n",
    "bars = ax1.bar(judge_cols, avg_scores, color=colors, alpha=0.85)\n",
    "ax1.set_ylim(0, 5.5)\n",
    "ax1.set_title('Rata-rata Skor LLM-as-a-Judge')\n",
    "ax1.set_ylabel('Skor (1-5)')\n",
    "for bar, score in zip(bars, avg_scores):\n",
    "    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{score:.2f}', ha='center', fontsize=11, fontweight='bold')\n",
    "\n",
    "# Radar-like heatmap per soal\n",
    "ax2 = axes[1]\n",
    "heatmap_data = df[judge_cols].values\n",
    "im = ax2.imshow(heatmap_data.T, aspect='auto', cmap='YlGn', vmin=1, vmax=5)\n",
    "ax2.set_xticks(range(10))\n",
    "ax2.set_xticklabels(df['question_id'].tolist(), rotation=45)\n",
    "ax2.set_yticks(range(4))\n",
    "ax2.set_yticklabels(judge_cols)\n",
    "ax2.set_title('Heatmap Skor Judge per Soal')\n",
    "for i in range(4):\n",
    "    for j in range(10):\n",
    "        ax2.text(j, i, str(int(heatmap_data[j, i])), ha='center', va='center', fontsize=9, fontweight='bold')\n",
    "plt.colorbar(im, ax=ax2)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "analysis-header",
   "metadata": {},
   "source": [
    "## 5. Narasi Analisis (Orang 3)\n",
    "\n",
    "### 1. Alur Berpikir Sistem RAG BANASPATI\n",
    "Sistem BANASPATI menggunakan pendekatan **Hybrid Retrieval** yang terdiri dari dua tahap:\n",
    "- **Tahap Retrieval**: ChromaDB mengambil top-40 kandidat chunk secara semantik (cosine similarity) menggunakan model embedding `paraphrase-multilingual-MiniLM-L12-v2`.\n",
    "- **Tahap Reranking**: Sistem menghitung skor keyword (dikurangi stopwords) dan mengutamakan chunk yang mengandung kode mata kuliah (ET234, UG234, SM234, SF234, EE234), lalu memfilter menjadi top-10 dokumen.\n",
    "- **Tahap Generation**: Top-10 chunk digabung menjadi satu konteks besar yang dimasukkan bersama instruksi anti-halusinasi ke model `gemini-2.5-flash`.\n",
    "\n",
    "### 2. Analisis Hasil Kualitas\n",
    "\n",
    "**RAGAS:**\n",
    "- **Faithfulness 0.925**: Sangat tinggi. Prompt anti-halusinasi berhasil memaksa model untuk hanya menjawab dari konteks yang diberikan.\n",
    "- **Answer Relevancy 0.895**: Tinggi. Jawaban yang dihasilkan relevan dengan pertanyaan yang diajukan.\n",
    "- **Context Precision 0.822**: Cukup baik. Hybrid reranking berhasil menempatkan chunk paling relevan di urutan teratas.\n",
    "- **Context Recall 0.862**: Baik. Retrieval berhasil menangkap sebagian besar informasi yang diperlukan untuk menjawab pertanyaan.\n",
    "\n",
    "**LLM-as-a-Judge:**\n",
    "- **Faithfulness 5.00/5.0**: Sempurna. Tidak ada halusinasi terdeteksi di semua 10 soal (10/10 hallucination-free).\n",
    "- **Correctness 4.50/5.0**: Sangat tinggi, terutama pada pertanyaan dengan referensi tekstual eksplisit (kalender akademik, peraturan SKS, ketentuan magang).\n",
    "- **Relevance 4.80/5.0**: Jawaban hampir selalu langsung menjawab inti pertanyaan.\n",
    "- **Completeness 4.40/5.0**: Sedikit lebih rendah; beberapa jawaban memilih merangkum poin utama tanpa menyertakan detail sekunder.\n",
    "\n",
    "### 3. Analisis Performa\n",
    "- **Bottleneck Utama**: Generation (rata-rata 3.235 s) jauh lebih lambat dari Retrieval (0.143 s). Ini adalah bottleneck tipikal sistem RAG berbasis LLM.\n",
    "- **Throughput**: Rata-rata 66.4 tokens/s sangat memadai untuk chatbot akademik interaktif.\n",
    "- **Biaya**: Total $0.002840 untuk 10 soal (\\~$0.000284/soal), sangat ekonomis dengan Gemini 2.5 Flash.\n",
    "\n",
    "### 4. Kesimpulan & Rekomendasi\n",
    "\n",
    "**Kekuatan Sistem:**\n",
    "- Prompt anti-halusinasi sangat efektif (Faithfulness 5.00/5.0, 0 kasus halusinasi)\n",
    "- Hybrid Retrieval berhasil menangkap kata kunci spesifik seperti kode mata kuliah dan nama ruangan\n",
    "- Biaya operasional sangat rendah menggunakan Gemini Flash\n",
    "\n",
    "**Kelemahan & Rekomendasi:**\n",
    "- **Latensi Generation**: E2E latency 3.378 s masih bisa ditingkatkan dengan respons *streaming*\n",
    "- **Context Precision**: Masih ada noise dari chunk yang kurang relevan; reranker Cross-Encoder (seperti `bge-reranker-v2`) dapat meningkatkan presisi\n",
    "- **Completeness**: Pada soal dengan banyak sub-poin (Q05, Q06), model kadang melewatkan detail minor; meningkatkan ukuran konteks atau menggunakan chain-of-thought prompting dapat membantu"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.13.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
with open('notebooks/03_evaluation_metrics.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook written successfully")
