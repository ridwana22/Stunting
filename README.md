# 🩺 Deteksi Dini Stunting dengan Machine Learning & Gemini AI

Aplikasi web ini merupakan **prototype** yang dirancang untuk membantu **deteksi dini stunting pada anak-anak** menggunakan model **Machine Learning**. Sistem ini kemudian diintegrasikan dengan **Google Gemini (LLM)** untuk memberikan **saran pencegahan dan penanganan yang dinamis serta personal**.

Aplikasi dibangun dengan **Flask** sebagai backend dan dirancang dengan antarmuka yang **sederhana, interaktif, dan ramah pengguna**.

---

## ✨ Fitur Utama

* **Prediksi Berbasis Machine Learning**
  Menggunakan model terlatih (berdasarkan dataset relevan) untuk memprediksi status stunting berdasarkan data anak (jenis kelamin, umur, tinggi, berat).

* **Saran Dinamis dari LLM**
  Terintegrasi dengan **Google Gemini (gemini-1.5-flash-latest)** untuk memberikan rekomendasi pencegahan dan penanganan yang dipersonalisasi sesuai dengan data serta hasil prediksi anak.

* **Visualisasi Data**
  Menampilkan posisi pertumbuhan anak pada kurva standar **WHO (median)** untuk memberikan konteks visual yang lebih jelas kepada pengguna.

* **Caching**
  Mengimplementasikan sistem cache untuk menyimpan respons dari LLM, sehingga dapat mengurangi latensi dan menghemat biaya API untuk permintaan yang berulang.

* **Antarmuka Interaktif**
  Dilengkapi dengan indikator loading, validasi input di sisi klien, serta desain yang ramah pengguna untuk meningkatkan pengalaman penggunaan aplikasi.

* **Manajemen Error**
  Sistem penanganan error yang baik untuk memberikan umpan balik yang jelas jika terjadi masalah pada input atau koneksi API.

---

## 🛠️ Teknologi yang Digunakan

* **Backend**: Python, Flask
* **Machine Learning**: Scikit-learn, Pandas, Joblib
* **Large Language Model (LLM)**: Google Gemini API (`google-generativeai`)
* **Frontend**: HTML, CSS, JavaScript

  * **Visualisasi**: Chart.js
* **Lainnya**: `python-dotenv` (manajemen API Key), `Flask-Caching`

---

Mau saya buatkan juga **struktur folder + cara install & run project** biar README ini lebih lengkap seperti proyek GitHub profesional?
