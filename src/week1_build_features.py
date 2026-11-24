"""
Minggu 1 — Ekstraksi Fitur 16 Konfigurasi Window STFT & Analisis Resolusi Sinyal.
Seluruh luaran Minggu 1 disimpan rapi di results/minggu_1/ (figures & tables).

Desain eksperimen:
    4 jenis window × 4 ukuran N = 16 konfigurasi (tanpa overlap/duplikasi)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.common import INTERIM, TAB_W1, periksa_fitur
from src.features import (
    CONFIGS, FEATURE_DIM, TARGET_SR, SIZES,
    ekstrak_satu_klip, ukur_waktu_ekstraksi,
)
from src import plots


def buat_tabel_resolusi(fs: int = TARGET_SR) -> pd.DataFrame:
    """Hitung trade-off resolusi STFT (Δf dan Δt) untuk tiap ukuran window N."""
    baris = []
    for n in SIZES:
        baris.append({
            "N"         : n,
            "delta_f_hz": round(fs / n, 2),
            "delta_t_ms": round(1000 * n / fs, 2),
        })
    return pd.DataFrame(baris)


if __name__ == "__main__":
    paths = pd.read_csv(INTERIM / "t1_manifest.csv")["path"].tolist()
    feature_path = INTERIM / "t1_features.npz"

    if not feature_path.exists():
        # Ekstraksi paralel (n_jobs=-2 = semua core kecuali satu)
        hasil = Parallel(n_jobs=-2)(
            delayed(ekstrak_satu_klip)(p)
            for p in tqdm(paths, desc="Ekstraksi fitur STFT", leave=False)
        )

        # Susun array per konfigurasi
        arrays = {
            cfg.nama: np.stack([r[cfg.nama] for r in hasil])
            for cfg in CONFIGS
        }

        # Verifikasi dimensi dan nilai
        for nama, X in arrays.items():
            periksa_fitur(X, len(paths), FEATURE_DIM)

        np.savez_compressed(feature_path, **arrays)

        # Benchmark waktu ekstraksi
        waktu = ukur_waktu_ekstraksi(paths)
        df_waktu = pd.DataFrame({
            "config"     : list(waktu.keys()),
            "ms_per_clip": list(waktu.values()),
            "feat_dim"   : FEATURE_DIM,
        })
        df_waktu.to_csv(TAB_W1 / "t1_time.csv", index=False)
    else:
        # Jika belum ada tabel waktu di minggu_1, ukur atau verifikasi
        if not (TAB_W1 / "t1_time.csv").exists():
            waktu = ukur_waktu_ekstraksi(paths)
            df_waktu = pd.DataFrame({
                "config"     : list(waktu.keys()),
                "ms_per_clip": list(waktu.values()),
                "feat_dim"   : FEATURE_DIM,
            })
            df_waktu.to_csv(TAB_W1 / "t1_time.csv", index=False)

    # Tabel resolusi STFT & plot resolusi sinyal
    df_res = buat_tabel_resolusi()
    df_res.to_csv(TAB_W1 / "t1_resolution.csv", index=False)
    df_res.to_csv(TAB_W1 / "minggu1_tabel_resolusi_stft.csv", index=False)
    plots.t1_resolusi_stft(df_res)

    print(
        f"[Minggu 1 - Fitur] Selesai: "
        f"16 konfigurasi × {len(paths)} klip = fitur tersedia di results/minggu_1/."
    )
