from flask import Flask, render_template, request, flash, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import os
import json

# --- Tambahan untuk Integrasi LLM & Caching ---
import google.generativeai as genai
from dotenv import load_dotenv
from flask_caching import Cache ### BARU: Impor Cache

# Muat environment variables dari file .env
load_dotenv()

# --- Inisialisasi Aplikasi dan Cache ---
app = Flask(__name__)
# ### BARU: Konfigurasi kunci rahasia untuk flash messages (Manajemen Error)
app.secret_key = os.urandom(24) 
# ### BARU: Konfigurasi Cache (sederhana, berbasis memori)
cache = Cache(config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 3600 # Cache selama 1 jam (3600 detik)
})
cache.init_app(app)

# --- KONFIGURASI API GEMINI ---
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY tidak ditemukan di file .env")
    genai.configure(api_key=api_key)
    llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("✅ Konfigurasi Google Gemini berhasil.")
except Exception as e:
    print(f"❌ Error konfigurasi Gemini: {e}")
    llm_model = None

# --- MEMUAT MODEL MACHINE LEARNING ---
try:
    model = joblib.load('virtual/best_stunting_model.joblib')
    scaler = joblib.load('virtual/scaler.joblib')
    le = joblib.load('virtual/label_encoder.joblib')
    print("✅ Model Machine Learning berhasil dimuat.")
except Exception as e:
    print(f"❌ Error saat memuat model ML: {e}")
    model = scaler = le = None

# --- FUNGSI PEMANGGILAN LLM (YANG SEBENARNYA) ---
### DIUBAH: Fungsi sekarang di-cache dan menerima data anak untuk personalisasi
@cache.memoize()
def get_llm_advice(prediction_status, umur_bulan, jenis_kelamin_teks):
    """
    Membuat prompt personal, memanggil API Gemini, dan mem-parsing hasilnya.
    Hasil dari fungsi ini akan di-cache.
    """
    if not llm_model:
        return {
            "pencegahan": ["Layanan LLM tidak tersedia saat ini."],
            "penanganan": ["Silakan periksa konfigurasi API Key."]
        }

    # ### DIUBAH: Prompt sekarang lebih personal
    prompt = f"""
    Anda adalah seorang ahli gizi anak yang memberikan saran untuk orang tua.
    Seorang pengguna baru saja mendapatkan hasil prediksi status gizi anaknya.

    Data Anak:
    - Umur: {umur_bulan} bulan
    - Jenis Kelamin: {jenis_kelamin_teks}
    - Hasil Prediksi: "{prediction_status}"

    Tugas Anda:
    Berikan saran yang spesifik, praktis, dan relevan dengan mempertimbangkan usia anak tersebut.
    Sajikan jawaban Anda HANYA dalam format JSON yang valid dengan dua kunci utama: "pencegahan" dan "penanganan".
    Setiap kunci harus berisi sebuah array (list) dari string, di mana setiap string adalah satu poin saran.
    Gunakan bahasa yang empatik dan mudah dimengerti oleh orang tua.
    """
    
    try:
        response = llm_model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        advice = json.loads(cleaned_response)
        return advice
    except Exception as e:
        print(f"❌ Error saat memanggil LLM atau parsing JSON: {e}")
        # ### DIUBAH: Memberikan fallback jika LLM error
        return {
            "pencegahan": ["Terjadi kesalahan saat mencoba mendapatkan saran dari AI. Fokus pada gizi seimbang dan pantau pertumbuhan anak di Posyandu."],
            "penanganan": ["Jika Anda khawatir, segera konsultasikan dengan dokter anak atau ahli gizi terdekat."]
        }


# --- ROUTING APLIKASI ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # ### DIUBAH: Manajemen Error yang lebih baik dengan blok try-except besar
        try:
            if not all([model, scaler, le]):
                 flash("Error: Model prediktor tidak siap. Silakan hubungi administrator.", "error")
                 return redirect(url_for('index'))

            # 1. Ambil & validasi data dari form
            jenis_kelamin = int(request.form['jenis_kelamin'])
            umur_bulan = int(request.form['umur_bulan'])
            tinggi_cm = float(request.form['tinggi_cm'])
            berat_kg = float(request.form['berat_kg'])

            # 2. Hitung fitur turunan (IMT)
            tinggi_m = tinggi_cm / 100
            imt = berat_kg / (tinggi_m ** 2) if tinggi_m > 0 else 0

            # 3. Buat DataFrame
            data_baru = pd.DataFrame({
                'Jenis Kelamin': [jenis_kelamin], 'Umur (bulan)': [umur_bulan],
                'Tinggi Badan (cm)': [tinggi_cm], 'Berat Badan (kg)': [berat_kg], 'IMT': [imt]
            })

            # 4. Lakukan scaling dan prediksi
            data_baru_scaled = scaler.transform(data_baru)
            prediksi_angka = model.predict(data_baru_scaled)
            prediksi_prob = model.predict_proba(data_baru_scaled)
            hasil_prediksi = le.inverse_transform(prediksi_angka)[0]
            probabilitas = np.max(prediksi_prob) * 100

            # 5. Panggil fungsi LLM dengan data personal
            jenis_kelamin_teks = 'Laki-laki' if jenis_kelamin == 1 else 'Perempuan'
            saran_llm = get_llm_advice(hasil_prediksi, umur_bulan, jenis_kelamin_teks)
            
            # ### BARU: Siapkan data untuk visualisasi
            user_data_for_chart = {
                "umur": umur_bulan,
                "tinggi": tinggi_cm
            }

            return render_template(
                'result.html', 
                prediction=hasil_prediksi, 
                probability=f"{probabilitas:.2f}",
                saran_pencegahan=saran_llm.get("pencegahan", []),
                saran_penanganan=saran_llm.get("penanganan", []),
                user_data=user_data_for_chart ### BARU: Kirim data ke template
            )
        # ### DIUBAH: Menangkap error input dan memberikan pesan yang ramah
        except ValueError:
            flash("Data yang Anda masukkan tidak valid. Harap periksa kembali semua isian.", "error")
            return redirect(url_for('index'))
        except Exception as e:
            flash(f"Terjadi kesalahan tak terduga: {e}", "error")
            return redirect(url_for('index'))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
