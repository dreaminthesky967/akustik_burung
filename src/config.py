"""
Konfigurasi parameter terpusat untuk Minggu 1:
Verifikasi Dataset, Praproses Sinyal, Desain Filter, dan Pipeline Ekstraksi Fitur.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# Pengaturan Path Direktori Otomatis
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
META_DIR = BASE_DIR / "data" / "metadata"
FIG_DIR = BASE_DIR / "outputs" / "figures"
TABLE_DIR = BASE_DIR / "outputs" / "tables"

for folder in [RAW_DATA_DIR, META_DIR, FIG_DIR, TABLE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Week1Config:
    """Parameter standar sesuai dokumen proposal Topik 1 (Kelompok 1)."""

    # Parameter Audio & Praproses
    target_fs: int = 22050             # Standar sample rate (22,05 kHz)
    pre_emphasis_coef: float = 0.97    # y[n] = x[n] - 0,97 * x[n-1]

    # Parameter Bandpass Filter (1 - 10 kHz)
    lowcut_hz: float = 1000.0
    highcut_hz: float = 10000.0
    iir_order: int = 4                 # Butterworth orde 4
    fir_numtaps: int = 101             # FIR orde setara (101 taps / orde 100)

    # Parameter STFT & Ekstraksi Fitur
    window_sizes: Tuple[int, ...] = (256, 512, 1024, 2048)
    window_types: Tuple[str, ...] = ("boxcar", "hann", "hamming", "blackman")
    n_mels: int = 64                   # Mel-spectrogram 64 band
    n_mfcc: int = 13                   # MFCC 13 koefisien

    # Parameter Validasi Silang (5-fold x 3 seed)
    n_folds: int = 5
    random_seeds: Tuple[int, ...] = (42, 100, 2026)


CONFIG = Week1Config()