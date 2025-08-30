from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Muat environment variables dari file .env
load_dotenv()

# --- KONFIGURASI API GEMINI ---
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY tidak ditemukan di file .env")
    genai.configure(api_key=api_key)
    # Inisialisasi model Generatif
    llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("✅ Konfigurasi Google Gemini berhasil.")
except Exception as e:
    print(f"❌ Error konfigurasi Gemini: {e}")
    llm_model = None

# Inisialisasi aplikasi Flask
app = Flask(__name__)

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
def get_llm_advice(prediction_status):
    """
    Membuat prompt, memanggil API Gemini, dan mem-parsing hasilnya.
    """
    if not llm_model:
        return {
            "pencegahan": ["Layanan LLM tidak tersedia saat ini."],
            "penanganan": ["Silakan periksa konfigurasi API Key."]
        }

    # -- Prompt Engineering: Merancang perintah untuk LLM --
    # Meminta output dalam format JSON membuat respons lebih mudah diolah.
    prompt = f"""
    Anda adalah seorang asisten kesehatan virtual yang ahli dalam gizi anak dan stunting.
    Seorang pengguna baru saja mendapatkan hasil prediksi status gizi anaknya.
    
    Hasil Prediksi: "{prediction_status}"
    
    Tugas Anda:
    Berikan saran yang jelas, ringkas, dan mudah dipahami mengenai upaya pencegahan dan rekomendasi penanganan.
    Sajikan jawaban Anda HANYA dalam format JSON yang valid dengan dua kunci utama: "pencegahan" dan "penanganan".
    Setiap kunci harus berisi sebuah array (list) dari string, di mana setiap string adalah satu poin saran.
    
    Contoh jika hasilnya 'Stunting':
    {{
        "pencegahan": ["Pastikan asupan gizi seimbang.", "Lanjutkan ASI eksklusif hingga 6 bulan."],
        "penanganan": ["Segera konsultasikan dengan dokter anak.", "Ikuti anjuran pemberian Pangan Olahan Medis Khusus jika direkomendasikan."]
    }}
    
    Sekarang, berikan saran untuk hasil prediksi di atas.
    """
    
    try:
        # Mengirim prompt ke Gemini
        response = llm_model.generate_content(prompt)
        
        # Membersihkan dan mem-parsing output JSON dari LLM
        # Terkadang LLM membungkus JSON dalam backticks (```json ... ```)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        advice = json.loads(cleaned_response)
        return advice
    except Exception as e:
        print(f"❌ Error saat memanggil LLM atau parsing JSON: {e}")
        return {
            "pencegahan": ["Terjadi kesalahan saat mencoba mendapatkan saran dari AI."],
            "penanganan": ["Silakan coba lagi nanti."]
        }


# --- ROUTING APLIKASI ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not all([model, scaler, le]):
             return "Error: Model ML tidak dapat digunakan. Silakan periksa log server.", 500

        try:
            jenis_kelamin = int(request.form['jenis_kelamin'])
            umur_bulan = int(request.form['umur_bulan'])
            tinggi_cm = float(request.form['tinggi_cm'])
            berat_kg = float(request.form['berat_kg'])

            tinggi_m = tinggi_cm / 100
            imt = berat_kg / (tinggi_m ** 2) if tinggi_m > 0 else 0

            data_baru = pd.DataFrame({
                'Jenis Kelamin': [jenis_kelamin], 'Umur (bulan)': [umur_bulan],
                'Tinggi Badan (cm)': [tinggi_cm], 'Berat Badan (kg)': [berat_kg], 'IMT': [imt]
            })

            data_baru_scaled = scaler.transform(data_baru)
            prediksi_angka = model.predict(data_baru_scaled)
            prediksi_prob = model.predict_proba(data_baru_scaled)
            hasil_prediksi = le.inverse_transform(prediksi_angka)[0]
            probabilitas = np.max(prediksi_prob) * 100

            # Panggil fungsi LLM yang sudah terintegrasi
            saran_llm = get_llm_advice(hasil_prediksi)

            return render_template(
                'result.html', 
                prediction=hasil_prediksi, 
                probability=f"{probabilitas:.2f}",
                saran_pencegahan=saran_llm.get("pencegahan", []),
                saran_penanganan=saran_llm.get("penanganan", [])
            )
        except Exception as e:
            return render_template('index.html', error=f"Terjadi kesalahan: {e}")

    return render_template('index.html', error=None)

if __name__ == '__main__':
    app.run(debug=True)
