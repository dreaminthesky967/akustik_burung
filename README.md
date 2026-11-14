# Analisis Akustik Suara Burung

## Deskripsi

Proyek ini digunakan untuk melakukan analisis sinyal audio suara burung menggunakan Python. Analisis meliputi ekstraksi fitur akustik, pengolahan sinyal, visualisasi data, serta analisis statistik untuk membandingkan karakteristik suara antar sampel.

## Fitur
'''
- Membaca file audio (.wav)
- Ekstraksi fitur akustik menggunakan Librosa
- Visualisasi waveform dan spectrogram
- Analisis statistik menggunakan SciPy dan Scikit\-Posthocs
- Visualisasi hasil dengan Matplotlib dan Seaborn
- Monitoring proses menggunakan TQDM
'''
## Struktur Proyek

akustik\_burung/

│

├── data/

│   ├── audio/

│   └── hasil/

│

├── src/

│   ├── preprocessing.py

│   ├── feature\_extraction.py

│   ├── analysis.py

│   └── visualization.py

│

├── requirements.txt

├── README.md

└── main.py

## Persyaratan

- Python 3.11

## Membuat Virtual Environment

'''
Windows:

py \-3.11 \-m venv env

env\\Scripts\\activate
'''

## Instalasi Dependensi
'''

pip install \-r requirements.txt

atau

pip install numpy pandas scipy librosa soundfile scikit\-learn scikit\-posthocs matplotlib seaborn tqdm
'''

## Menjalankan Program
'''

python3 run_pipeline.py

'''
## Library yang Digunakan

- NumPy
- Pandas
- SciPy
- Librosa
- SoundFile
- Scikit\-Learn
- Scikit\-Posthocs
- Matplotlib
- Seaborn
- TQDM

## Output

Program dapat menghasilkan:

- Data fitur akustik
- Grafik waveform
- Grafik spectrogram
- Hasil analisis statistik
- Visualisasi perbandingan antar sampel suara burung

## Author

Nama: 

Mata Kuliah: Sinyal dan Sistem

Program Studi: Sistem Komputer

