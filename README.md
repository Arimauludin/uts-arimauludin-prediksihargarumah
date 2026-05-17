# 🏡 Web Prediksi Harga Rumah - UTS Kecerdasan Buatan

Aplikasi web berbasis **Flask** yang dirancang untuk mensimulasikan dan membandingkan hasil prediksi harga rumah menggunakan 5 algoritma Machine Learning yang berbeda. Aplikasi ini juga dilengkapi dengan penyimpanan riwayat prediksi secara lokal menggunakan database SQLite.

---

## 🚀 Fitur Utama
* **Form Prediksi Interaktif:** Mengalkulasi estimasi harga rumah berdasarkan luas tanah, luas bangunan, jumlah kamar tidur, kamar mandi, dan lokasi.
* **Simulasi & Perbandingan 5 Model ML:** Menyediakan komparasi hasil prediksi dari algoritma *Linear Regression, Random Forest, Gradient Boosting (Patokan Utama), Decision Tree,* dan *SVR*.
* **Grafik Performa Dinamis:** Visualisasi perbandingan harga antar model pada halaman khusus.
* **Sistem Database Otomatis:** Setiap data input dan hasil prediksi disimpan secara aman ke database SQLite (`predictions.db`).
* **Desain Responsive:** Menggunakan template HTML yang modern dan nyaman dilihat di berbagai perangkat.

---

## 📁 Struktur Project
```text
UTS-PREDIKSI-HARGA-RUMAH/
├── .venv-1/               # Virtual Environment Python
├── dataset/               # Folder penyimpanan dataset pendukung
├── models/                # Tempat menyimpan serialized model (.pkl / .h5)
├── static/                # File CSS, JavaScript, dan Gambar untuk Web
├── templates/             # Template HTML (index.html, prediksi.html, dll)
├── .gitignore             # File konfigurasi pengabaian Git (Virtual Env & DB)
├── app.py                 # File utama aplikasi Flask & database SQLite
├── Procfile               # Konfigurasi deployment untuk platform Render
├── requirements.txt       # Daftar dependency / library Python
└── README.md              # Dokumentasi project