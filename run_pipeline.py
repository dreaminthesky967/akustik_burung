"""
Pipeline Utama — Analisis Window STFT pada Sinyal Akustik Burung (Topik 1).

Desain eksperimen:
    4 jenis window  × 4 ukuran N = 16 konfigurasi (tanpa overlap/duplikasi)
    Evaluasi        : 16 × 5-fold × 3 seed = 240 evaluasi fold
    Baseline        : hann_N2048_ov50 (default librosa), termasuk dalam grid
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TAHAPAN = [
    ("Minggu 1: Persiapan Dataset & Representasi Sinyal", "src.week1_prepare"),
    ("Minggu 1: Ekstraksi Fitur & Resolusi STFT",         "src.week1_build_features"),
    ("Minggu 2: Eksperimen Klasifikasi & Karakter Sistem", "src.week2_experiment"),
    ("Minggu 2: Pengujian Hipotesis H1 & H2",             "src.week2_hypotheses"),
    ("Minggu 2: Uji Filter Bandpass & SNR Sinyal (H3)",   "src.week2_snr"),
    ("Minggu 3: Laporan, Spektrogram & Evaluasi Akhir",   "src.week3_report"),
]


def jalankan_tahap(nama: str, modul: str) -> None:
    """Jalankan modul dan laporkan status."""
    print(f"\n{'─' * 60}")
    print(f"  [MENJALANKAN] {nama}")
    print(f"{'─' * 60}")
    hasil = subprocess.run([sys.executable, "-m", modul], cwd=ROOT)
    if hasil.returncode != 0:
        print(f"  [GAGAL] {nama} berhenti dengan error ({hasil.returncode}).")
        sys.exit(hasil.returncode)


if __name__ == "__main__":
    print("=" * 65)
    print("  PIPELINE ANALISIS SINYAL & SISTEM AKUSTIK BURUNG (TOPIK 1)")
    print("  16 Konfigurasi Window STFT (4 window × 4 ukuran)")
    print("=" * 65)

    for nama, modul in TAHAPAN:
        jalankan_tahap(nama, modul)

    print("\n" + "=" * 65)
    print("  [SELESAI] Semua tahapan berhasil dijalankan!")
    print("  Hasil terstruktur tersimpan pada folder:")
    print("    - results/minggu_1/ (figures & tables)")
    print("    - results/minggu_2/ (figures & tables)")
    print("    - results/minggu_3/ (figures & tables)")
    print("=" * 65)
