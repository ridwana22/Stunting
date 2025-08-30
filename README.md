🩺 Deteksi Dini Stunting dengan Machine Learning & Gemini AI
Aplikasi web ini adalah sebuah purwarupa (prototype) yang dirancang untuk membantu melakukan deteksi dini stunting pada anak-anak menggunakan model Machine Learning. Aplikasi ini kemudian diintegrasikan dengan Google Gemini (LLM) untuk memberikan saran pencegahan dan penanganan yang dinamis dan personal.

Aplikasi ini dibangun menggunakan Flask sebagai kerangka kerja backend dan dirancang dengan antarmuka yang ramah pengguna.

✨ Fitur Utama
Prediksi Berbasis Machine Learning: Menggunakan model yang telah dilatih (berdasarkan dataset relevan) untuk memprediksi status stunting berdasarkan input data anak (jenis kelamin, umur, tinggi, berat).

Saran Dinamis dari LLM: Terintegrasi dengan Google Gemini (gemini-1.5-flash-latest) untuk memberikan rekomendasi pencegahan dan penanganan yang dipersonalisasi sesuai dengan data dan hasil prediksi anak.

Visualisasi Data: Menampilkan posisi pertumbuhan anak pada kurva standar WHO (median) untuk memberikan konteks visual yang lebih baik kepada pengguna.

Caching: Mengimplementasikan sistem cache untuk menyimpan respons dari LLM, sehingga mengurangi latensi dan menghemat biaya API untuk permintaan yang berulang.

Antarmuka Interaktif: Dilengkapi dengan indikator loading dan validasi input di sisi klien untuk meningkatkan pengalaman pengguna.

Manajemen Error: Sistem penanganan error yang baik untuk memberikan umpan balik yang jelas jika terjadi masalah pada input atau koneksi API.

🛠️ Teknologi yang Digunakan
Backend: Python, Flask

Machine Learning: Scikit-learn, Pandas, Joblib

Large Language Model (LLM): Google Gemini API (google-generativeai)

Frontend: HTML, CSS, JavaScript
-- Visualisasi: Chart.js

Lainnya: python-dotenv (untuk manajemen API Key), Flask-Caching
