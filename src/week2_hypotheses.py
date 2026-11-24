"""
Minggu 2 — Pengujian Hipotesis H1 & H2 serta Visualisasi Hasil Percobaan STFT.
Seluruh luaran Minggu 2 disimpan rapi di results/minggu_2/ (figures & tables).

H1: Window 512–1024 sampel menghasilkan macro-F1 lebih tinggi dari 256 dan 2048.
H2: Window tapered (Hann, Blackman) mengungguli rectangular.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.common import TAB_W2, rank_biserial
from src import plots

KELAS_NOISE = "noise"


def rerata_per_blok(df: pd.DataFrame, masker: pd.Series, kolom: str) -> pd.Series:
    """Hitung rerata kolom per blok (seed, fold)."""
    return df[masker].groupby(["seed", "fold"])[kolom].mean()


def uji_kontras(
    df      : pd.DataFrame,
    masker_a: pd.Series,
    masker_b: pd.Series,
    kolom   : str,
    nama    : str,
) -> dict:
    """Wilcoxon berpasangan satu sisi: a > b pada 15 blok."""
    a = rerata_per_blok(df, masker_a, kolom)
    b = rerata_per_blok(df, masker_b, kolom)
    _, p = wilcoxon(a, b, alternative="greater")
    return {
        "contrast"       : nama,
        "metric"         : kolom,
        "mean_a"         : float(a.mean()),
        "mean_b"         : float(b.mean()),
        "diff"           : float(a.mean() - b.mean()),
        "p_raw"          : float(p),
        "r_rank_biserial": rank_biserial(a, b),
    }


def koreksi_holm(p_values: pd.Series) -> np.ndarray:
    """Koreksi Holm untuk multiple testing."""
    m = len(p_values)
    urutan = p_values.argsort().to_numpy()
    p_koreksi = np.empty(m)
    batas_kum = 0.0
    for rank, idx in enumerate(urutan):
        batas_kum = max(batas_kum, min(1.0, (m - rank) * p_values.iloc[idx]))
        p_koreksi[idx] = batas_kum
    return p_koreksi


if __name__ == "__main__":
    df = pd.read_csv(TAB_W2 / "t1_fold_scores.csv")

    # 16 konfigurasi (seluruhnya overlap 50%)
    grid = df[df.overlap == 0.5] if "overlap" in df.columns else df
    kolom_noise = f"f1_{KELAS_NOISE}"

    kontras = [
        uji_kontras(
            grid,
            grid.n.isin([512, 1024]),
            grid.n == 256,
            "macro_f1",
            "H1: 512-1024 > 256",
        ),
        uji_kontras(
            grid,
            grid.n.isin([512, 1024]),
            grid.n == 2048,
            "macro_f1",
            "H1: 512-1024 > 2048",
        ),
        uji_kontras(
            grid,
            grid.window.isin(["hann", "blackman"]),
            grid.window == "rect",
            "macro_f1",
            "H2: Window tapered > rect (F1)",
        ),
    ]
    if kolom_noise in grid.columns:
        kontras.append(
            uji_kontras(
                grid,
                grid.window.isin(["hann", "blackman"]),
                grid.window == "rect",
                kolom_noise,
                "H2: Window tapered > rect (Noise)",
            )
        )

    hasil = pd.DataFrame(kontras)
    hasil["p_holm"] = koreksi_holm(hasil["p_raw"])
    hasil.to_csv(TAB_W2 / "t1_contrasts.csv", index=False)

    # Rerata marginal per ukuran N dan per jenis window
    m_n = grid.groupby("n")["macro_f1"].mean().rename("macro_f1_mean")
    m_w = grid.groupby("window")["macro_f1"].mean().rename("macro_f1_mean")
    m_n.to_csv(TAB_W2 / "t1_marginal_n.csv")
    m_w.to_csv(TAB_W2 / "t1_marginal_window.csv")

    # Grafik evaluasi H1 & H2 (Minggu 2)
    plots.t1_heatmap_f1(df)
    plots.t1_kurva_f1_vs_n(df)
    plots.t1_boxplot_f1(df)

    print(
        f"[Minggu 2 - Hypotheses] Selesai: "
        f"{len(hasil)} kontras diuji & grafik tersimpan di results/minggu_2/."
    )
