"""
Praproses sinyal dan ekstraksi fitur MFCC untuk 16 konfigurasi window STFT.

Desain eksperimen:
    4 jenis window  : rectangular, Hann, Hamming, Blackman
    4 ukuran window : 256, 512, 1024, 2048 sampel
    Total           : 4 × 4 = 16 konfigurasi (tanpa duplikasi / overlap)
    Overlap tetap   : 50% untuk semua konfigurasi
    Baseline        : hann_N2048_ov50 (default librosa), sudah termasuk
                      dalam 16 konfigurasi grid

Modul ini hanya berisi fungsi murni (tanpa side-effect I/O) agar
bisa dipanggil paralel dari week1_build_features.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt, lfilter

# ------------------------------------------------------------------ konstanta
TARGET_SR  : int = 22_050       # sample rate kerja (Hz)
N_MFCC     : int = 13           # jumlah koefisien MFCC
N_MELS     : int = 40           # jumlah filter mel (aman untuk N ≥ 256)
FEATURE_DIM: int = N_MFCC * 4   # (mean, std) × (MFCC, Δ-MFCC) = 52

# Pemetaan nama pendek → nama librosa/scipy
WINDOWS: dict[str, str] = {
    "rect"    : "boxcar",
    "hann"    : "hann",
    "hamming" : "hamming",
    "blackman": "blackman",
}
SIZES: tuple[int, ...] = (256, 512, 1024, 2048)


# ================================================================== konfigurasi

@dataclass(frozen=True)
class KonfigurasiWindow:
    """
    Satu konfigurasi STFT: jenis window, ukuran N, dan overlap.

    Contoh penamaan: hann_N1024_ov50
    """

    window : str          # kunci di WINDOWS (rect/hann/hamming/blackman)
    n      : int          # ukuran window dalam sampel
    overlap: float = 0.5  # 50% overlap untuk semua konfigurasi

    @property
    def hop(self) -> int:
        """Jarak antar frame dalam sampel."""
        return int(self.n * (1 - self.overlap))

    @property
    def nama(self) -> str:
        """Identifier unik, dipakai sebagai kunci di file .npz dan kolom CSV."""
        return f"{self.window}_N{self.n}_ov{int(self.overlap * 100)}"


# 16 konfigurasi = 4 window × 4 ukuran (tanpa duplikasi)
CONFIGS = [KonfigurasiWindow(w, n) for w in WINDOWS for n in SIZES]

# Baseline: hann_N2048_ov50 (default librosa), sudah ada di dalam CONFIGS
BASELINE = next(c for c in CONFIGS if c.window == "hann" and c.n == 2048)

# Verifikasi: harus tepat 16 konfigurasi, tidak ada nama yang sama
assert len(CONFIGS) == 16, f"Harus 16 konfigurasi, bukan {len(CONFIGS)}"
assert len({c.nama for c in CONFIGS}) == 16, "Ada nama konfigurasi duplikat!"


# ================================================================== praproses

def praproses(y: np.ndarray, sr: int, bandpass: bool = True) -> np.ndarray:
    """
    Terapkan pre-emphasis lalu (opsional) filter bandpass 1–10 kHz.

    Parameters
    ----------
    y       : sinyal mono float32
    sr      : sample rate sinyal masukan (harus == TARGET_SR)
    bandpass: jika True, terapkan Butterworth orde 4 zero-phase

    Returns
    -------
    np.ndarray float32
    """
    # Pre-emphasis: y[n] = x[n] − 0,97·x[n−1]
    y = lfilter([1.0, -0.97], [1.0], y)

    if bandpass:
        sos = butter(4, [1_000, 10_000], btype="band", fs=sr, output="sos")
        y   = sosfiltfilt(sos, y)   # zero-phase: tidak menggeser fase

    return y.astype(np.float32)


# ================================================================== fitur

def hitung_mfcc_statistik(y: np.ndarray, cfg: KonfigurasiWindow) -> np.ndarray:
    """
    Hitung MFCC 13 + Δ-MFCC, lalu rangkum dengan (mean, std) sepanjang waktu.

    Returns
    -------
    np.ndarray shape (FEATURE_DIM,) = (52,)
    """
    mfcc = librosa.feature.mfcc(
        y          = y,
        sr         = TARGET_SR,
        n_mfcc     = N_MFCC,
        n_fft      = cfg.n,
        hop_length = cfg.hop,
        window     = WINDOWS[cfg.window],
        n_mels     = N_MELS,
    )
    assert mfcc.shape[1] >= 9, (
        f"Terlalu sedikit frame ({mfcc.shape[1]}) untuk delta (butuh ≥9)"
    )
    delta = librosa.feature.delta(mfcc, width=9, mode="nearest")

    return np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        delta.mean(axis=1),
        delta.std(axis=1),
    ])


def ekstrak_satu_klip(path: str, bandpass: bool = True) -> dict[str, np.ndarray]:
    """
    Muat, praproses, lalu hitung fitur untuk SEMUA 16 konfigurasi.

    Audio hanya dimuat dan dipraproses SEKALI, menghemat I/O disk.

    Returns
    -------
    dict  {cfg.nama: np.ndarray shape (52,)}  —  16 entri
    """
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    y    = praproses(y, TARGET_SR, bandpass)
    return {cfg.nama: hitung_mfcc_statistik(y, cfg) for cfg in CONFIGS}


def ukur_waktu_ekstraksi(paths: list[str], n_klip: int = 200) -> dict[str, float]:
    """
    Ukur waktu ekstraksi (ms/klip) tiap konfigurasi.

    Dijalankan pada SATU proses agar tidak bias dari paralelisme.

    Returns
    -------
    dict  {cfg.nama: ms_per_klip}  —  16 entri
    """
    sinyal = [
        praproses(librosa.load(p, sr=TARGET_SR)[0], TARGET_SR)
        for p in paths[:n_klip]
    ]
    hasil = {}
    for cfg in CONFIGS:
        t0 = time.perf_counter()
        for y in sinyal:
            hitung_mfcc_statistik(y, cfg)
        hasil[cfg.nama] = 1000 * (time.perf_counter() - t0) / len(sinyal)
    return hasil
