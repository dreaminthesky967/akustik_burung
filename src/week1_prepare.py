"""
Minggu 1 — Persiapan Dataset, Manifest, Stratified Folds, dan Representasi Sinyal.
Seluruh luaran Minggu 1 disimpan rapi di results/minggu_1/ (figures & tables).
"""

from __future__ import annotations

import pandas as pd
import soundfile as sf

from src.common import (
    INTERIM, RAW, SEEDS, TAB_W1, FIG_W1,
    buat_fold, catat_lingkungan, simpan_fold,
)
from src import plots
from src.preprocess import (
    plot_filter_response_comparison,
    plot_sample_preprocessing_effect,
)

TRAIN_DIR = RAW / "bird_sounds" / "training"


def bangun_manifest() -> pd.DataFrame:
    """Pindai file WAV di TRAIN_DIR dan buat DataFrame manifest."""
    baris = []
    for path in sorted(TRAIN_DIR.rglob("*.wav")):
        if path.name.startswith("._") or path.name.startswith("."):
            continue
        if "__MACOSX" in path.parts:
            continue

        label = (
            path.name.split(".")[0]
            .lower().strip()
            .replace(" ", "_")
            .replace("-", "_")
        )
        try:
            info = sf.info(str(path))
            baris.append({
                "path"    : str(path),
                "label"   : label,
                "sr"      : info.samplerate,
                "duration": round(info.duration, 4),
                "channels": info.channels,
            })
        except RuntimeError:
            pass

    df = pd.DataFrame(baris)
    assert len(df) > 0, f"Tidak ada file WAV ditemukan di {TRAIN_DIR}."
    return df


def validasi_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Filter klip rusak/terlalu pendek (< 0.5 detik)."""
    df = df[df["duration"] > 0.5].copy()
    assert df["label"].nunique() == 6, (
        f"Ditemukan {df['label'].nunique()} kelas (harus 6)."
    )
    return df


if __name__ == "__main__":
    catat_lingkungan("t1")

    # 1. Bangun dan validasi manifest
    manifest = validasi_manifest(bangun_manifest())
    manifest.to_csv(INTERIM / "t1_manifest.csv", index=False)
    manifest.to_csv(TAB_W1 / "t1_manifest.csv", index=False)

    # 2. Tabel ringkasan distribusi kelas
    ringkasan_kelas = manifest.groupby("label").size().rename("n_clips")
    ringkasan_kelas.to_csv(TAB_W1 / "t1_class_counts.csv")

    ringkasan_distribusi = (
        manifest.groupby(["label", "sr"])
        .agg(
            jumlah_klip=("path", "count"),
            durasi_rata2_dtk=("duration", "mean"),
            durasi_min_dtk=("duration", "min"),
            durasi_max_dtk=("duration", "max"),
        )
        .reset_index()
    )
    ringkasan_distribusi.to_csv(TAB_W1 / "ringkasan_distribusi_dataset.csv", index=False)

    # 3. Stratified 5-fold cross validation untuk 3 seed
    y = manifest["label"].to_numpy()
    folds_per_seed = {s: buat_fold(y, seed=s) for s in SEEDS}
    simpan_fold(folds_per_seed, INTERIM / "t1_folds.json")

    # 4. Visualisasi Representasi Sinyal (Minggu 1)
    plots.t1_distribusi_dataset(manifest)
    plots.t1_contoh_sinyal_waveform(manifest, n_kelas=4)

    # 5. Visualisasi Desain Filter Praproses (Butterworth vs FIR dan Spektrum Daya)
    sample_wav = manifest["path"].iloc[0]
    p_filter = plot_filter_response_comparison(FIG_W1 / "minggu1_respons_filter_iir_vs_fir.png")
    p_psd = plot_sample_preprocessing_effect(sample_wav, FIG_W1 / "minggu1_efek_praproses_psd.png")

    print(f"[Minggu 1 - Prepare] Selesai: {len(manifest)} klip terverifikasi (6 kelas).")
    print(f"                     Output tersimpan di results/minggu_1/ (figures & tables).")
