"""
Minggu 3 — Analitik Akhir, Matriks Konfusi, Representasi Waktu-Frekuensi, dan Laporan.
Seluruh luaran Minggu 3 disimpan rapi di results/minggu_3/ (figures & tables).

Menghasilkan:
    - Tabel metrik window (main lobe, sidelobe)
    - Matriks konfusi konfigurasi terbaik
    - F1 per kelas
    - Tabel ringkasan komparasi lengkap 16 konfigurasi
    - Spektrogram dan MFCC representasi sinyal & sistem
    - Trade-off akurasi vs biaya komputasi
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import get_window
from sklearn.metrics import confusion_matrix, f1_score

from src import plots
from src.common import INTERIM, TAB_W1, TAB_W2, TAB_W3, FIG_W2, FIG_W3, muat_fold
from src.features import WINDOWS
from src.week2_experiment import buat_model


def tabel_metrik_window(n: int = 256, pad: int = 2**14) -> pd.DataFrame:
    """Metrik analitik main lobe dan sidelobe untuk setiap window."""
    baris = []
    for label, nama in WINDOWS.items():
        resp = np.abs(np.fft.rfft(get_window(nama, n), pad))
        db   = 20 * np.log10(resp / resp.max() + 1e-12)
        null_pertama = int(np.argmax(np.diff(db) > 0))
        baris.append({
            "window"          : label,
            "mainlobe_bins"   : round(2 * null_pertama * n / pad, 2),
            "peak_sidelobe_db": round(float(db[null_pertama:].max()), 2),
        })
    return pd.DataFrame(baris)


def prediksi_out_of_fold(nama_config: str, seed: int = 0):
    """Hasilkan prediksi out-of-fold untuk evaluasi final."""
    y = pd.read_csv(INTERIM / "t1_manifest.csv")["label"].to_numpy()
    X = np.load(INTERIM / "t1_features.npz")[nama_config]
    pred = np.empty_like(y)
    for tr, te in muat_fold(INTERIM / "t1_folds.json")[seed]:
        pred[te] = buat_model(seed).fit(X[tr], y[tr]).predict(X[te])
    return y, pred


if __name__ == "__main__":
    skor     = pd.read_csv(TAB_W2 / "t1_fold_scores.csv")
    manifest = pd.read_csv(INTERIM / "t1_manifest.csv")
    waktu    = pd.read_csv(TAB_W1 / "t1_time.csv")

    # Konfigurasi terbaik berdasarkan rerata macro-F1
    config_terbaik = skor.groupby("config")["macro_f1"].mean().idxmax()
    f1_terbaik     = skor.groupby("config")["macro_f1"].mean()[config_terbaik]

    # 1. Tabel metrik window (mainlobe & sidelobe)
    df_metrik_w = tabel_metrik_window()
    df_metrik_w.to_csv(TAB_W3 / "t1_window_metrics.csv", index=False)

    # 2. Grafik trade-off akurasi vs waktu
    plots.t1_tradeoff_akurasi_waktu(skor, waktu, folder=FIG_W3, nama_file="w3_tradeoff")
    plots.t1_tradeoff_akurasi_waktu(skor, waktu, folder=FIG_W2, nama_file="w2_tradeoff")

    # 3. Prediksi out-of-fold & matriks konfusi
    y_asli, y_pred = prediksi_out_of_fold(config_terbaik)
    label_kelas    = sorted(set(y_asli))
    cm = confusion_matrix(y_asli, y_pred, labels=label_kelas, normalize="true")
    plots.matriks_konfusi(cm, label_kelas, f"w3_confusion_{config_terbaik}")

    # 4. F1 per kelas untuk konfigurasi terbaik
    df_f1_kelas = pd.DataFrame({
        "class": label_kelas,
        "f1"   : f1_score(
            y_asli, y_pred, labels=label_kelas, average=None, zero_division=0
        ),
    })
    df_f1_kelas.to_csv(
        TAB_W3 / f"t1_f1_per_class_{config_terbaik}.csv", index=False
    )
    plots.t1_f1_per_kelas(df_f1_kelas, config_terbaik)

    # 5. Tabel ringkasan akhir 16 konfigurasi (terurut dari terbaik)
    ringkasan_16 = (
        skor.groupby("config")
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
            accuracy_mean=("accuracy", "mean"),
        )
        .reset_index()
        .merge(waktu[["config", "ms_per_clip"]], on="config", how="left")
        .sort_values("macro_f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    ringkasan_16.index += 1
    ringkasan_16.index.name = "peringkat"
    ringkasan_16.to_csv(TAB_W3 / "t1_ringkasan_akhir_16konfigurasi.csv")

    # 6. Representasi waktu-frekuensi (STFT Spektrogram & MFCC)
    plots.t1_spektrogram_contoh(manifest, n=1024, window="hann")
    plots.t1_mfcc_contoh(manifest, config_nama=config_terbaik)

    print(
        f"[Minggu 3 - Report] Selesai: "
        f"Konfigurasi terbaik = {config_terbaik} (Macro-F1 = {f1_terbaik:.4f})."
    )
    print(f"                     Output tersimpan di results/minggu_3/ (figures & tables).")
