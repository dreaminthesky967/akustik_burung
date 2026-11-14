"""
fitur STFT, Mel-Spectrogram (64 band), dan MFCC  
"""

import time
from typing import Tuple
import numpy as np
import pandas as pd
import librosa

from src.config import CONFIG, TABLE_DIR


def save_stft_resolution_table(fs: int = CONFIG.target_fs) -> pd.DataFrame:
    """Menghitung dan menyimpan tabel trade-off resolusi frekuensi (df) dan waktu (dt)."""
    rows = []
    for n_fft in CONFIG.window_sizes:
        df_hz = fs / n_fft
        dt_ms = (n_fft / fs) * 1000.0
        rows.append({
            "ukuran_window_N": n_fft,
            "sample_rate_hz": fs,
            "delta_f_hz": round(df_hz, 2),
            "delta_t_ms": round(dt_ms, 2),
            "hop_50_sampel": n_fft // 2,
            "hop_75_sampel": n_fft // 4,
        })

    df_res = pd.DataFrame(rows)
    out_csv = TABLE_DIR / "minggu1_tabel_resolusi_stft.csv"
    df_res.to_csv(out_csv, index=False)
    print("\n=== TABEL TRADE-OFF RESOLUSI STFT (Delta f vs Delta t) ===")
    print(df_res.to_string(index=False))
    return df_res


def extract_features_single_clip(
    waveform: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 512,
    window_type: str = "hann",
    fs: int = CONFIG.target_fs,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Mengekstraksi fitur dari sinyal audio yang telah dipraproses:
    1. STFT dengan ukuran N dan jenis window tertentu.
    2. Mel-Spectrogram 64 band pada pita 1-10 kHz.
    3. MFCC 13 koefisien + Delta-MFCC, diringkas menjadi mean & std (52 fitur).
    """
    t_start = time.perf_counter()

    # 1. STFT
    stft_spec = librosa.stft(
        waveform, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window_type
    )
    power_spec = np.abs(stft_spec) ** 2

    # 2. Mel-Spectrogram (64 band)
    mel_spec = librosa.feature.melspectrogram(
        S=power_spec,
        sr=fs,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=CONFIG.n_mels,
        fmin=CONFIG.lowcut_hz,
        fmax=CONFIG.highcut_hz,
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # 3. MFCC (13 koefisien) + Delta-MFCC
    mfcc = librosa.feature.mfcc(S=log_mel_spec, sr=fs, n_mfcc=CONFIG.n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)

    # 4. Vektor fitur statistik per klip (Mean & Std sepanjang waktu = 52 dimensi)
    feature_vector = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta_mfcc, axis=1),
        np.std(delta_mfcc, axis=1),
    ]).astype(np.float32)

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return feature_vector, log_mel_spec, elapsed_ms