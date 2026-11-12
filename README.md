# Analisis Performa Penjualan, Perilaku Pelanggan, dan Distribusi Geografis pada Brazilian E-Commerce Olist

## Deskripsi Proyek

Proyek ini bertujuan untuk menganalisis data Brazilian E-Commerce Olist guna memahami performa pesanan dan perilaku pelanggan melalui proses analisis data, visualisasi, serta pembuatan dashboard interaktif menggunakan Streamlit.

## Pertanyaan Bisnis

1. Bagaimana perkembangan jumlah pesanan terkirim setiap bulan dan periode mana yang menunjukkan peningkatan atau penurunan aktivitas pesanan?
2. Bagaimana karakteristik pelanggan berdasarkan segmentasi RFM (Recency, Frequency, Monetary), dan segmen pelanggan mana yang perlu diprioritaskan dalam strategi retensi?

## Struktur Proyek

```text
submission/
├── dashboard/
│   ├── main_data.csv
│   └── dashboard.py
├── data/
│   ├── customers_dataset.csv
│   ├── geolocation_dataset.csv
│   ├── order_items_dataset.csv
│   ├── order_payments_dataset.csv
│   ├── order_reviews_dataset.csv
│   ├── orders_dataset.csv
│   ├── products_dataset.csv
│   ├── sellers_dataset.csv
│   └── product_category_name_translation.csv
├── notebook.ipynb
├── README.md
├── requirements.txt
└── url.txt
```

## Setup Environment

### 1. Setup Environment - Anaconda

Buat environment baru menggunakan Python:

```bash
conda create --name main-ds python=3.11
```

Aktifkan environment:

```bash
conda activate main-ds
```

### 2. Setup Environment - Shell/Terminal

Jika tidak menggunakan Anaconda, buat virtual environment dengan perintah berikut.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Pastikan terminal berada di direktori utama proyek (`submission/`), kemudian jalankan:

```bash
pip install -r requirements.txt
```

Perintah tersebut akan menginstal seluruh library yang diperlukan untuk menjalankan dashboard.

## Menjalankan Dashboard

Setelah seluruh dependensi berhasil diinstal, jalankan aplikasi Streamlit menggunakan perintah:

```bash
streamlit run dashboard/dashboard.py
```

Dashboard akan terbuka secara otomatis melalui browser. Jika tidak terbuka, salin alamat lokal yang muncul pada terminal ke browser.

## Dashboard Online

Dashboard juga dapat diakses melalui tautan yang tercantum pada file `url.txt` di dalam folder submission.
