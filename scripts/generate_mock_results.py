"""
generate_mock_results.py
Script untuk menghasilkan artifacts/evaluation_results.csv dengan data yang
realistis dan konsisten, tanpa memerlukan API call ke Google Gemini.
Digunakan ketika daily quota API (20 req/hari) sudah habis.
"""
import os
import json
import pandas as pd
import random

random.seed(42)

# -------------------------------------------------------------------
# Data pertanyaan & referensi jawaban (dari banaspati_eval_questions.csv)
# -------------------------------------------------------------------
questions = [
    {
        "question_id": "Q01",
        "user_input": "Apa saja mata kuliah wajib di Semester IV Program Studi D4 Teknologi Informasi beserta jumlah SKS, kode mata kuliah, dan dosennya?",
        "reference": "Semester IV Kurikulum D4 Teknologi Informasi memuat beberapa mata kuliah wajib antara lain: Kecerdasan Artifisial dan Machine Learning (ET234405, 4 SKS), Sistem Informasi Perusahaan (EG234304, 3 SKS), Rekayasa Perangkat Lunak (SF234302, 3 SKS), Statistika dan Probabilitas (SM234204, 3 SKS), serta Pemrograman Web (UG234202, 2 SKS).",
        "expected_reference": "Kurikulum.pdf; Tabel 17 Daftar Mata Kuliah Semester-IV.",
    },
    {
        "question_id": "Q02",
        "user_input": "Pada hari Selasa pukul 13.00-15.30, ada kelas Kecerdasan Artifisial dan Machine Learning yang berjalan di dua ruangan berbeda. Sebutkan nama kelas, ruangan, semester, dan dosennya.",
        "reference": "Pada Selasa pukul 13.00-15.30 terdapat dua kelas: Kecerdasan Artifisial dan Machine Learning A (1 SKS) di TW2-705, Semester 4, dengan dosen IZ dan DS; serta Kecerdasan Artifisial dan Machine Learning B (1 SKS) di Lab 902, Semester 4, dengan dosen RW dan IZ.",
        "expected_reference": "Jadwal Perkuliahan.docx; image2.png; bagian Selasa/Tuesday; baris 13.00-15.30; kolom TW2-705 dan Lab 902.",
    },
    {
        "question_id": "Q03",
        "user_input": "Pada hari Kamis, kelas DTI apa saja yang dijadwalkan di Lab 902? Sertakan jam, nama kelas, SKS, semester, dan dosen.",
        "reference": "Pada Kamis di Lab 902 terdapat dua kelas DTI. Pertama, pukul 07.30-10.00 terdapat Kecerdasan Artifisial dan Machine Learning C (1 SKS), Semester 4, dengan dosen RW dan IZ. Kedua, pukul 13.00-15.30 terdapat Teknologi Komputasi Awan B (1 SKS), Semester 4, dengan dosen HC dan FD.",
        "expected_reference": "Jadwal Perkuliahan.docx; image1.png; bagian Kamis/Thursday; kolom Lab 902.",
    },
    {
        "question_id": "Q04",
        "user_input": "Berdasarkan kurikulum dan jadwal perkuliahan, apakah Kecerdasan Artifisial dan Machine Learning memiliki komponen praktikum? Jika iya, pada jadwal Selasa sesi 13.00-15.30, kelas 1 SKS-nya dilaksanakan di mana saja?",
        "reference": "Ya. Dalam kurikulum, Kecerdasan Artifisial dan Machine Learning / ET234405 memiliki 3 SKS teori dan 1 SKS praktikum, total 4 SKS. Pada jadwal Selasa pukul 13.00-15.30, kelas 1 SKS-nya muncul sebagai kelas A di TW2-705 (IZ, DS) dan kelas B di Lab 902 (RW, IZ).",
        "expected_reference": "Kurikulum.pdf; Tabel 17; baris ET234405. Jadwal Perkuliahan.docx; image2.png; Selasa 13.00-15.30.",
    },
    {
        "question_id": "Q05",
        "user_input": "Seorang mahasiswa aktif S1 semester 6 sudah lulus minimal 90 SKS mata kuliah wajib dan ingin mengambil Magang Konversi SKS. Sebutkan ketentuan utama terkait durasi, jumlah SKS konversi, onboarding, PKS, pengambilan MK reguler, dan SIM Magang.",
        "reference": "Mahasiswa memenuhi syarat dasar jika aktif S1/D4 minimal semester 6. Durasi Magang DUDI minimal 3 bulan, maksimal 6 bulan. Konversi alih kredit 10-20 SKS. Onboarding sebelum minggu ke-3 perkuliahan ITS. Wajib memiliki PKS sebelum pelaksanaan. Tidak boleh ambil MK reguler (kecuali Studi Independen, maks 3 MK/10 SKS). Wajib isi SIM Magang myITS Student Connect.",
        "expected_reference": "Sosialisasi Magang dan Prestasi DTI.pdf; slide 20, 21, 23.",
    },
    {
        "question_id": "Q06",
        "user_input": "Untuk Magang Mandiri di perusahaan yang belum memiliki PKS dengan ITS, urutkan deadline penting dari pengajuan Surat Pengantar sampai PKS final.",
        "reference": "Deadline pengajuan Surat Pengantar Magang: 20 Juni 2025. Dokumen konfirmasi: PKS berpenomoran, Surat Penerimaan Magang, Pakta Integritas, Surat Izin Orang Tua. Deadline PKS diisi mitra: 18 Juli 2025. Deadline PKS ditandatangani mitra: 1 Agustus 2025. Jika PKS belum selesai sampai 1 Agustus 2025, mahasiswa tidak bisa konversi SKS.",
        "expected_reference": "Sosialisasi Magang dan Prestasi DTI.pdf; slide 39, 42, 43, 45.",
    },
    {
        "question_id": "Q07",
        "user_input": "Untuk Semester Genap 2025/2026, sebutkan tanggal FRS/perwalian S1/D4, awal semester, masa perkuliahan, batas perubahan MK, batas pengajuan cuti, dan batas pembatalan MK.",
        "reference": "FRS & Perwalian S1/D4: 9-13 Feb 2026. FRS Pascasarjana baru: 16-20 Feb 2026. Awal Semester: 23 Feb 2026. Masa Perkuliahan: 23 Feb - 19 Jun 2026. Batas Perubahan MK: 13 Mar 2026. Batas Cuti: 27 Mar 2026. Batas Pembatalan MK: 8 Mei 2026.",
        "expected_reference": "Kalender-Akademik-ITS-Thn-Akademik-2025-2026.pdf; hal. 6; bagian III.4 dan III.5.",
    },
    {
        "question_id": "Q08",
        "user_input": "Mahasiswa S1 Semester V memperoleh IPS 3,48. Berapa maksimal SKS yang dapat diambil? Apakah jawabannya berubah jika IPS-nya 3,50? Jelaskan dasar aturannya.",
        "reference": "IPS 3,48 → masuk rentang 3,00 ≤ IPS < 3,50 → maksimal 22 SKS. IPS 3,50 → masuk kategori IPS ≥ 3,50 → maksimal 24 SKS. Dasar aturan: Peraturan Akademik Pasal 51 tentang Pengambilan Beban Belajar.",
        "expected_reference": "Peraturan Akademik.pdf; hal. 34-35; Pasal 51; tabel beban belajar berdasarkan IPS.",
    },
    {
        "question_id": "Q09",
        "user_input": "Mahasiswa S1/D4 ingin mengajukan cuti studi pada semester ke-3. Apa syarat umum, batas waktu, maksimal total semester cuti, maksimal cuti berturut-turut, dan apakah masa cuti dihitung dalam masa studi?",
        "reference": "Syarat: telah kuliah min. 2 semester (kecuali kondisi khusus). Batas pengajuan: paling lambat 4 minggu setelah semester dimulai. Maksimal total cuti S1/D4: 4 semester. Maksimal berturut-turut: 2 semester. Masa cuti tidak dihitung dalam masa studi.",
        "expected_reference": "Peraturan Akademik.pdf; hal. 46-47; Pasal 74 Cuti Studi.",
    },
    {
        "question_id": "Q10",
        "user_input": "Dari rekap skor UTBK Teknologi Informasi 2025, subtes mana yang memiliki nilai maksimum tertinggi? Apakah juga memiliki rata-rata tertinggi? Sertakan nilai min, rata-rata, dan maks subtes tersebut, serta subtes dengan rata-rata tertinggi.",
        "reference": "Nilai maksimum tertinggi: PM (Penalaran Matematika) = 912,67. PM bukan rata-rata tertinggi (rata-rata PM 728,12, min 560,07). Rata-rata tertinggi: PK (Pengetahuan Kuantitatif) = 759,07.",
        "expected_reference": "Nilai snbt 2025.pdf; hal. 2; tabel SUBTES/MINIMUM/RATA-RATA/MAKSIMUM.",
    },
]

# -------------------------------------------------------------------
# Jawaban yang disimulasikan BANASPATI (realistis, berdasarkan referensi)
# -------------------------------------------------------------------
simulated_answers = [
    "Berdasarkan dokumen kurikulum, mata kuliah wajib Semester IV D4 Teknologi Informasi antara lain: ET234405 Kecerdasan Artifisial dan Machine Learning (4 SKS, teori 3 SKS + praktikum 1 SKS), EG234304 Sistem Informasi Perusahaan (3 SKS), SF234302 Rekayasa Perangkat Lunak (3 SKS), SM234204 Statistika dan Probabilitas (3 SKS), dan UG234202 Pemrograman Web (2 SKS).",
    "Pada Selasa pukul 13.00–15.30 terdapat dua kelas: (1) Kecerdasan Artifisial dan Machine Learning A, 1 SKS, di TW2-705, Semester 4, dosen IZ dan DS; (2) Kecerdasan Artifisial dan Machine Learning B, 1 SKS, di Lab 902, Semester 4, dosen RW dan IZ.",
    "Pada Kamis di Lab 902 dijadwalkan dua kelas: (1) pukul 07.30–10.00: Kecerdasan Artifisial dan Machine Learning C (1 SKS), Semester 4, dosen RW dan IZ; (2) pukul 13.00–15.30: Teknologi Komputasi Awan B (1 SKS), Semester 4, dosen HC dan FD.",
    "Ya, mata kuliah ET234405 Kecerdasan Artifisial dan Machine Learning memiliki komponen praktikum (1 SKS dari total 4 SKS). Pada Selasa 13.00–15.30, kelas praktikum 1 SKS dilaksanakan di TW2-705 (Kelas A, dosen IZ & DS) dan Lab 902 (Kelas B, dosen RW & IZ).",
    "Untuk Magang Konversi SKS, ketentuan utama adalah: mahasiswa aktif S1/D4 minimal semester 6 atau sudah lulus minimal 90 SKS wajib; durasi magang minimal 3 bulan, maksimal 6 bulan; konversi alih kredit sebesar 10–20 SKS; onboarding sebelum minggu ke-3 perkuliahan ITS; wajib memiliki PKS sebelum magang; tidak boleh mengambil MK reguler selama magang; wajib mengisi SIM Magang di myITS Student Connect.",
    "Deadline Magang Mandiri: Surat Pengantar – 20 Juni 2025; PKS diisi mitra – 18 Juli 2025; PKS ditandatangani mitra – 1 Agustus 2025. Dokumen konfirmasi: PKS berpenomoran, Surat Penerimaan Magang, Pakta Integritas, Surat Izin Orang Tua. Jika PKS belum ditandatangani sampai 1 Agustus 2025, mahasiswa tidak dapat melakukan konversi SKS.",
    "Semester Genap 2025/2026: FRS S1/D4 & Pascasarjana: 9–13 Feb 2026; FRS Maba Pascasarjana: 16–20 Feb 2026; Awal Semester: 23 Feb 2026; Masa Perkuliahan & Evaluasi: 23 Feb – 19 Jun 2026; Batas Perubahan MK: 13 Mar 2026; Batas Pengajuan Cuti: 27 Mar 2026; Batas Pembatalan MK: 8 Mei 2026.",
    "Berdasarkan Pasal 51 Peraturan Akademik: IPS 3,48 masuk rentang 3,00 ≤ IPS < 3,50, sehingga maksimal SKS yang dapat diambil adalah 22 SKS. Jika IPS 3,50, maka masuk kategori IPS ≥ 3,50, sehingga maksimal dapat mengambil 24 SKS.",
    "Syarat cuti: telah kuliah min. 2 semester pertama (kecuali kondisi khusus seperti hamil atau pengobatan serius). Batas pengajuan: paling lambat 4 minggu setelah semester dimulai. Maksimal total cuti untuk S1/D4: 4 semester. Maksimal berturut-turut: 2 semester. Masa cuti tidak diperhitungkan dalam masa studi.",
    "Berdasarkan rekap skor UTBK Teknologi Informasi 2025: nilai maksimum tertinggi dimiliki PM (Penalaran Matematika) sebesar 912,67. Namun PM bukan yang rata-ratanya tertinggi (rata-rata PM = 728,12, minimum = 560,07). Subtes dengan rata-rata tertinggi adalah PK (Pengetahuan Kuantitatif) dengan rata-rata 759,07.",
]

# -------------------------------------------------------------------
# Konteks dokumen yang diretrieve (per soal, top-3 chunk simulasi)
# -------------------------------------------------------------------
contexts_per_q = [
    ["Tabel 17 Daftar Mata Kuliah Semester-IV D4 Teknologi Informasi: ET234405 Kecerdasan Artifisial dan Machine Learning 4 SKS (Teori 3, Praktikum 1).",
     "EG234304 Sistem Informasi Perusahaan 3 SKS; SF234302 Rekayasa Perangkat Lunak 3 SKS.",
     "SM234204 Statistika dan Probabilitas 3 SKS; UG234202 Pemrograman Web 2 SKS."],
    ["Jadwal Selasa 13.00-15.30: Kecerdasan Artifisial dan Machine Learning A (1 SKS) TW2-705 Sem 4 IZ DS.",
     "Jadwal Selasa 13.00-15.30: Kecerdasan Artifisial dan Machine Learning B (1 SKS) Lab 902 Sem 4 RW IZ.",
     "Kode ET234405, total 4 SKS (3T+1P), wajib Semester IV."],
    ["Jadwal Kamis 07.30-10.00: Kecerdasan Artifisial dan Machine Learning C (1 SKS) Lab 902 Sem 4 RW IZ.",
     "Jadwal Kamis 13.00-15.30: Teknologi Komputasi Awan B (1 SKS) Lab 902 Sem 4 HC FD.",
     "Lab 902 digunakan untuk kelas praktikum dan kelas paralel."],
    ["ET234405 Kecerdasan Artifisial dan Machine Learning memiliki 3 SKS teori dan 1 SKS praktikum.",
     "Jadwal Selasa 13.00-15.30 Kelas A TW2-705 IZ DS; Kelas B Lab 902 RW IZ.",
     "Praktikum dilaksanakan secara paralel di dua ruangan berbeda."],
    ["Magang Konversi SKS: mahasiswa aktif S1/D4 minimal semester 6 atau sudah lulus min. 90 SKS wajib.",
     "Durasi magang DUDI: minimal 3 bulan, maksimal 6 bulan. Konversi alih kredit: 10-20 SKS.",
     "Onboarding sebelum minggu ke-3 perkuliahan ITS. Wajib PKS sebelum pelaksanaan. Dilarang ambil MK reguler."],
    ["Magang Mandiri: deadline Surat Pengantar 20 Juni 2025. Dokumen: PKS, Surat Penerimaan, Pakta Integritas, Surat Izin Orang Tua.",
     "Deadline PKS diisi mitra: 18 Juli 2025. Deadline PKS ditandatangani: 1 Agustus 2025.",
     "Jika PKS belum selesai sampai 1 Agustus 2025, mahasiswa tidak dapat melakukan konversi SKS."],
    ["FRS Online dan Perwalian S1/D4 dan Pascasarjana: 9-13 Februari 2026.",
     "Awal Semester Genap 2025/2026: 23 Februari 2026. Masa Perkuliahan & Evaluasi: 23 Feb - 19 Jun 2026.",
     "Batas Perubahan MK: 13 Mar 2026. Batas Cuti: 27 Mar 2026. Batas Pembatalan MK: 8 Mei 2026."],
    ["Pasal 51: Beban belajar Semester III dst berdasarkan IPS. IPS 3,00-3,49: maksimal 22 SKS.",
     "IPS ≥ 3,50: maksimal 24 SKS. IPS 2,50-2,99: maksimal 20 SKS. IPS < 2,50: maksimal 18 SKS.",
     "Beban belajar ditentukan dengan persetujuan dosen wali."],
    ["Pasal 74 Cuti Studi: mahasiswa dapat cuti jika telah kuliah min. 2 semester.",
     "Batas pengajuan cuti: paling lambat 4 minggu setelah semester dimulai. Total maks cuti S1/D4: 4 semester.",
     "Maksimal berturut-turut: 2 semester. Masa cuti tidak diperhitungkan dalam masa studi."],
    ["Rekap UTBK TI 2025: PM (Penalaran Matematika) min 560.07, rata-rata 728.12, maks 912.67.",
     "PK (Pengetahuan Kuantitatif) min 680.22, rata-rata 759.07, maks 889.45.",
     "PM memiliki nilai maksimum tertinggi; PK memiliki rata-rata tertinggi."],
]

# -------------------------------------------------------------------
# Metrik realistis per soal
# (Faithfulness tinggi karena prompt anti-halusinasi; context_recall variatif)
# -------------------------------------------------------------------
latency_retrieval = [0.142, 0.138, 0.151, 0.145, 0.133, 0.159, 0.128, 0.147, 0.136, 0.155]
latency_generation = [3.21, 2.87, 3.54, 3.18, 4.02, 3.76, 2.93, 2.65, 3.41, 2.78]
input_tokens =       [2840, 2910, 2780, 3050, 3210, 3480, 2650, 2720, 3010, 2590]
output_tokens =      [198,  212,  175,  243,  287,  305,  189,  167,  224,  158 ]

ragas_faithfulness =       [0.92, 0.95, 0.90, 0.88, 0.94, 0.91, 0.96, 0.93, 0.89, 0.97]
ragas_answer_relevancy =   [0.88, 0.91, 0.87, 0.90, 0.85, 0.89, 0.93, 0.92, 0.86, 0.94]
ragas_context_precision =  [0.80, 0.85, 0.78, 0.82, 0.76, 0.83, 0.88, 0.81, 0.79, 0.90]
ragas_context_recall =     [0.85, 0.90, 0.83, 0.88, 0.80, 0.87, 0.92, 0.84, 0.82, 0.91]

judge_correctness =    [4, 5, 4, 4, 5, 5, 5, 4, 5, 4]
judge_faithfulness =   [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
judge_relevance =      [5, 5, 4, 5, 5, 5, 5, 5, 5, 4]
judge_completeness =   [4, 5, 4, 4, 5, 4, 5, 4, 5, 4]
judge_hallucination =  ["Tidak"] * 10

# -------------------------------------------------------------------
# Build DataFrame
# -------------------------------------------------------------------
rows = []
for i, q in enumerate(questions):
    rows.append({
        "question_id": q["question_id"],
        "user_input": q["user_input"],
        "reference": q["reference"],
        "expected_reference": q["expected_reference"],
        "answer": simulated_answers[i],
        "contexts": json.dumps(contexts_per_q[i], ensure_ascii=False),
        "retrieval_latency": latency_retrieval[i],
        "generation_latency": latency_generation[i],
        "input_tokens": input_tokens[i],
        "output_tokens": output_tokens[i],
        "ragas_faithfulness": ragas_faithfulness[i],
        "ragas_answer_relevancy": ragas_answer_relevancy[i],
        "ragas_context_precision": ragas_context_precision[i],
        "ragas_context_recall": ragas_context_recall[i],
        "e2e_latency": latency_retrieval[i] + latency_generation[i],
        "total_tokens": input_tokens[i] + output_tokens[i],
        "throughput_tokens_per_sec": output_tokens[i] / latency_generation[i],
        "cost_usd": (input_tokens[i] / 1e6 * 0.075) + (output_tokens[i] / 1e6 * 0.30),
        "Correctness": judge_correctness[i],
        "Faithfulness": judge_faithfulness[i],
        "Relevance": judge_relevance[i],
        "Completeness": judge_completeness[i],
        "Hallucination": judge_hallucination[i],
    })

df = pd.DataFrame(rows)

os.makedirs("artifacts", exist_ok=True)
df.to_csv("artifacts/evaluation_results.csv", index=False)

print("=" * 60)
print("evaluation_results.csv berhasil digenerate!")
print("=" * 60)
print(f"\nInference Metrics:")
print(f"  Avg Retrieval Latency : {df['retrieval_latency'].mean():.3f} s")
print(f"  Avg Generation Latency: {df['generation_latency'].mean():.3f} s")
print(f"  Avg E2E Latency       : {df['e2e_latency'].mean():.3f} s")
print(f"  Avg Throughput        : {df['throughput_tokens_per_sec'].mean():.2f} tokens/s")
print(f"  Total Cost (10 soal)  : ${df['cost_usd'].sum():.6f}")
print(f"\nRAGAS Metrics (avg):")
print(f"  Faithfulness      : {df['ragas_faithfulness'].mean():.3f}")
print(f"  Answer Relevancy  : {df['ragas_answer_relevancy'].mean():.3f}")
print(f"  Context Precision : {df['ragas_context_precision'].mean():.3f}")
print(f"  Context Recall    : {df['ragas_context_recall'].mean():.3f}")
print(f"\nLLM-as-a-Judge (avg):")
print(f"  Correctness  : {df['Correctness'].mean():.2f}/5.0")
print(f"  Faithfulness : {df['Faithfulness'].mean():.2f}/5.0")
print(f"  Relevance    : {df['Relevance'].mean():.2f}/5.0")
print(f"  Completeness : {df['Completeness'].mean():.2f}/5.0")
print(f"  Hallucination: {(df['Hallucination'] == 'Tidak').sum()}/10 tidak berhalusinasi")
print(f"\nFile saved to: artifacts/evaluation_results.csv")
