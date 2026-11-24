"""
Minggu 2 — Analisis Peningkatan SNR melalui Filter Bandpass (H3).
Seluruh luaran Minggu 2 disimpan rapi di results/minggu_2/ (figures & tables).

H3: Bandpass 1–10 kHz menaikkan SNR kelas burung secara signifikan.
"""

from __future__ import annotations

import librosa
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.common import INTERIM, TAB_W2, rank_biserial
from src.features import TARGET_SR, praproses
from src import plots

KELAS_DIKECUALIKAN = {"noise", "other_birds"}
N_KLIP_SAMPEL = 1_000


def estimasi_snr(y: np.ndarray, ukuran_frame: int = 512) -> float:
    """
    Estimasi SNR berbasis kuartil frame energi.

    Frame dengan energi tertinggi dianggap sinyal,
    frame dengan energi terendah dianggap noise.
    """
    n_frame = len(y) // ukuran_frame
    if n_frame < 4:
        return 0.0

    daya = (
        y[:n_frame * ukuran_frame]
        .reshape(n_frame, ukuran_frame) ** 2
    ).mean(axis=1)

    q = max(n_frame // 4, 1)
    terurut = np.sort(daya)
    return float(10 * np.log10(
        (terurut[-q:].mean() + 1e-12) / (terurut[:q].mean() + 1e-12)
    ))


if __name__ == "__main__":
    target_csv = TAB_W2 / "t1_snr.csv"

    if target_csv.exists() and len(pd.read_csv(target_csv)) > 0:
        df_snr = pd.read_csv(target_csv)
    else:
        manifest  = pd.read_csv(INTERIM / "t1_manifest.csv")
        df_burung = manifest[~manifest["label"].isin(KELAS_DIKECUALIKAN)]
        df_sampel = df_burung.sample(
            min(N_KLIP_SAMPEL, len(df_burung)), random_state=0
        )

        baris = []
        for path in df_sampel["path"]:
            y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
            baris.append({
                "clip"      : path,
                "snr_before": estimasi_snr(praproses(y, TARGET_SR, bandpass=False)),
                "snr_after" : estimasi_snr(praproses(y, TARGET_SR, bandpass=True)),
            })

        df_snr = pd.DataFrame(baris)
        df_snr.to_csv(target_csv, index=False)

    # Uji Wilcoxon berpasangan
    _, p = wilcoxon(df_snr["snr_after"], df_snr["snr_before"])
    r    = rank_biserial(df_snr["snr_after"], df_snr["snr_before"])
    gain = np.median(df_snr["snr_after"] - df_snr["snr_before"])

    # Plot visualisasi SNR (Minggu 2)
    plots.t1_snr_berpasangan(df_snr)

    print(
        f"[Minggu 2 - SNR] Selesai: "
        f"median gain ΔSNR = {gain:.2f} dB (p = {p:.2e}, r = {r:.2f})."
    )
    print(f"                 Output tersimpan di results/minggu_2/ (figures & tables).")
