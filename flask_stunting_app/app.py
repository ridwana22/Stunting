from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# --- MEMUAT MODEL DAN ARTEFAK LAINNYA ---
# Muat semua file yang dibutuhkan saat aplikasi pertama kali dijalankan
try:
    model = joblib.load('virtual/best_stunting_model.joblib')
    scaler = joblib.load('virtual/scaler.joblib')
    le = joblib.load('virtual/label_encoder.joblib')
    print("✅ Model dan artefak berhasil dimuat.")
except FileNotFoundError as e:
    print(f"❌ Error: File model tidak ditemukan. Pastikan file berada di direktori yang benar. {e}")
    model = scaler = le = None
except Exception as e:
    print(f"❌ Error saat memuat model: {e}")
    model = scaler = le = None

# --- FUNGSI SIMULASI LLM UNTUK SARAN ---
def get_llm_advice(prediction_status):
    """
    Mensimulasikan panggilan ke LLM (seperti Gemini) untuk mendapatkan saran
    berdasarkan hasil prediksi.
    """
    if prediction_status == "Stunting":
        pencegahan_list = [
            "Pastikan asupan gizi seimbang, kaya protein hewani (telur, ikan, ayam), zat besi, dan zinc.",
            "Lanjutkan Inisiasi Menyusu Dini (IMD) dan berikan ASI eksklusif hingga bayi berusia 6 bulan.",
            "Berikan Makanan Pendamping ASI (MPASI) yang berkualitas dan beragam mulai usia 6 bulan.",
            "Rutin pantau pertumbuhan dan perkembangan anak di Posyandu atau fasilitas kesehatan terdekat.",
            "Jaga kebersihan lingkungan dan sanitasi untuk mencegah infeksi berulang yang dapat mengganggu penyerapan gizi."
        ]
        penanganan_list = [
            "Segera konsultasikan dengan dokter anak atau ahli gizi untuk mendapatkan asesmen dan rencana tatalaksana gizi yang tepat.",
            "Dokter mungkin akan merekomendasikan pemberian Pangan Olahan untuk Keperluan Medis Khusus (PKMK) jika diperlukan.",
            "Stimulasi psikososial dan motorik anak melalui permainan edukatif untuk mendukung perkembangan otaknya.",
            "Atasi penyakit penyerta jika ada, seperti infeksi cacing atau anemia, sesuai anjuran dokter.",
            "Perbaiki pola makan seluruh keluarga untuk menciptakan lingkungan yang mendukung perbaikan gizi anak."
        ]
    else: # Jika "Tidak Stunting"
        pencegahan_list = [
            "Kerja bagus! Terus pertahankan pola asuh dan asupan gizi yang sudah baik.",
            "Pastikan anak mendapatkan imunisasi lengkap dan tepat waktu untuk melindunginya dari berbagai penyakit.",
            "Lanjutkan pemantauan berat badan dan tinggi badan secara berkala untuk deteksi dini masalah pertumbuhan.",
            "Ciptakan lingkungan yang bersih dan aman untuk bermain dan bereksplorasi.",
            "Berikan stimulasi sesuai usia untuk mendukung perkembangan kognitif dan motorik yang optimal."
        ]
        penanganan_list = [
            "Saat ini tidak diperlukan penanganan khusus karena status gizi anak baik.",
            "Fokus pada upaya pencegahan untuk memastikan tren pertumbuhan positif ini terus berlanjut.",
            "Jika ada kekhawatiran sekecil apa pun mengenai pertumbuhan atau kesehatan anak, jangan ragu untuk berkonsultasi dengan tenaga kesehatan."
        ]
        
    return {"pencegahan": pencegahan_list, "penanganan": penanganan_list}


# --- ROUTING APLIKASI ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Pastikan model sudah termuat
        if not all([model, scaler, le]):
             return "Error: Model tidak dapat digunakan. Silakan periksa log server.", 500

        try:
            # 1. Ambil data dari form dan konversi ke tipe numerik
            jenis_kelamin = int(request.form['jenis_kelamin'])
            umur_bulan = int(request.form['umur_bulan'])
            tinggi_cm = float(request.form['tinggi_cm'])
            berat_kg = float(request.form['berat_kg'])

            # 2. Hitung fitur turunan (IMT)
            tinggi_m = tinggi_cm / 100
            imt = berat_kg / (tinggi_m ** 2) if tinggi_m > 0 else 0

            # 3. Buat DataFrame sesuai format yang dibutuhkan model
            data_baru = pd.DataFrame({
                'Jenis Kelamin': [jenis_kelamin],
                'Umur (bulan)': [umur_bulan],
                'Tinggi Badan (cm)': [tinggi_cm],
                'Berat Badan (kg)': [berat_kg],
                'IMT': [imt]
            })

            # 4. Lakukan scaling pada data
            data_baru_scaled = scaler.transform(data_baru)

            # 5. Lakukan prediksi
            prediksi_angka = model.predict(data_baru_scaled)
            prediksi_prob = model.predict_proba(data_baru_scaled)
            
            # 6. Terjemahkan hasil prediksi
            hasil_prediksi = le.inverse_transform(prediksi_angka)[0]
            probabilitas = np.max(prediksi_prob) * 100

            # 7. Dapatkan saran dari "LLM"
            saran_llm = get_llm_advice(hasil_prediksi)

            # 8. Kirim semua hasil ke halaman result.html
            return render_template(
                'result.html', 
                prediction=hasil_prediksi, 
                probability=f"{probabilitas:.2f}",
                saran_pencegahan=saran_llm["pencegahan"],
                saran_penanganan=saran_llm["penanganan"]
            )

        except Exception as e:
            # Jika terjadi error saat proses, kembali ke halaman utama dengan pesan error
            return render_template('index.html', error=f"Terjadi kesalahan: {e}")

    # Jika metodenya GET, tampilkan halaman utama
    return render_template('index.html', error=None)

# Menjalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)