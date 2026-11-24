"""
Semua fungsi grafik Topik 1 — Sinyal dan Sistem Akustik Burung.

Prinsip visualisasi:
  - Warna bersih, konsisten, dan mudah dibedakan (palet tab10)
  - Label sumbu minimal dan informatif
  - Setiap plot memperlihatkan representasi sinyal/sistem yang relevan

Daftar plot (16 konfigurasi, tanpa duplikasi):
  Minggu 1: distribusi dataset, waveform sinyal, resolusi STFT
  Minggu 2: heatmap F1, kurva F1 vs N, boxplot F1,
            trade-off akurasi-waktu, SNR berpasangan,
            respons window, respons bandpass
  Minggu 3: spektrogram, matriks konfusi, F1 per kelas, MFCC
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import get_window, butter, sosfilt

from src.common import FIG_W1, FIG_W2, FIG_W3

# ------------------------------------------------------------------ gaya global
plt.rcParams.update({
    "figure.dpi"       : 150,
    "savefig.bbox"     : "tight",
    "axes.grid"        : True,
    "grid.alpha"       : 0.25,
    "grid.linestyle"   : "--",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.size"        : 10,
})

# Palet warna utama (mudah dibedakan, ramah buta warna)
PAL  = sns.color_palette("tab10")
PAL4 = PAL[:4]
sns.set_palette("tab10")

# Baseline: hann_N2048_ov50 (sudah termasuk dalam 16 konfigurasi grid)
NAMA_BASELINE = "hann_N2048_ov50"
URUTAN_WINDOW = ["rect", "hann", "hamming", "blackman"]
WARNA_WINDOW  = {w: PAL4[i] for i, w in enumerate(URUTAN_WINDOW)}


# ================================================================== bantu

def _simpan(fig: plt.Figure, folder, nama: str) -> None:
    """Simpan gambar ke folder dan tutup figure."""
    fig.savefig(folder / f"{nama}.png")
    plt.close(fig)


def _ci95(x: pd.Series) -> float:
    """Confidence interval 95%."""
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


# ================================================================== MINGGU 1

def t1_distribusi_dataset(manifest: pd.DataFrame) -> None:
    """Bar chart jumlah klip per kelas + distribusi durasi."""
    kelas_counts = manifest["label"].value_counts().sort_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Panel kiri: jumlah klip per kelas
    bars = ax1.bar(
        kelas_counts.index, kelas_counts.values,
        color=sns.color_palette("tab10", len(kelas_counts)),
    )
    ax1.set(xlabel="Kelas", ylabel="Jumlah klip", title="Jumlah Klip per Kelas")
    plt.setp(ax1.get_xticklabels(), rotation=25, ha="right")
    for bar in bars:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=8,
        )

    # Panel kanan: distribusi durasi
    for label, grp in manifest.groupby("label"):
        ax2.hist(grp["duration"], bins=20, alpha=0.55, label=label)
    ax2.set(xlabel="Durasi (detik)", ylabel="Frekuensi",
            title="Distribusi Durasi Klip")
    ax2.legend(fontsize=7, ncol=2)

    _simpan(fig, FIG_W1, "w1_distribusi_dataset")


def t1_contoh_sinyal_waveform(manifest: pd.DataFrame, n_kelas: int = 4) -> None:
    """Waveform (domain waktu) representatif satu klip per kelas."""
    import librosa
    from src.features import TARGET_SR

    kelas_unik = sorted(manifest["label"].unique())[:n_kelas]
    fig, axes = plt.subplots(n_kelas, 1, figsize=(10, 2.4 * n_kelas), sharex=False)
    if n_kelas == 1:
        axes = [axes]

    for ax, kelas in zip(axes, kelas_unik):
        path = manifest[manifest["label"] == kelas]["path"].iloc[0]
        y, sr = librosa.load(path, sr=TARGET_SR, mono=True, duration=3.0)
        t = np.linspace(0, len(y) / sr, len(y))
        ax.plot(t, y, lw=0.6, color=WARNA_WINDOW.get(kelas, PAL[0]))
        ax.set(ylabel="Amplitudo", title=kelas)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    axes[-1].set_xlabel("Waktu (detik)")
    fig.suptitle("Waveform Sinyal Suara Burung", fontsize=12, y=1.01)
    fig.tight_layout()
    _simpan(fig, FIG_W1, "w1_waveform_sinyal")


def t1_resolusi_stft(df_resolusi: pd.DataFrame) -> None:
    """Visualisasi trade-off Δf vs Δt untuk setiap ukuran window N."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        df_resolusi["delta_t_ms"],
        df_resolusi["delta_f_hz"],
        s=120, zorder=5,
        c=range(len(df_resolusi)), cmap="tab10",
    )
    for _, row in df_resolusi.iterrows():
        ax.annotate(
            f"N={int(row['N'])}",
            (row["delta_t_ms"], row["delta_f_hz"]),
            textcoords="offset points", xytext=(6, 4), fontsize=9,
        )
    ax.set(
        xlabel="Resolusi waktu Δt (ms)",
        ylabel="Resolusi frekuensi Δf (Hz)",
        title="Trade-off Resolusi STFT: Δt vs Δf",
    )
    _simpan(fig, FIG_W1, "w1_resolusi_stft")


# ================================================================== MINGGU 2

def t1_heatmap_f1(skor: pd.DataFrame) -> None:
    """Heatmap rerata macro-F1: jenis window × ukuran N (16 sel)."""
    pivot = (
        skor[skor.overlap == 0.5]
        .pivot_table(index="window", columns="n", values="macro_f1", aggfunc="mean")
        .loc[URUTAN_WINDOW]
    )
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax,
        cbar_kws={"label": "Macro-F1"},
    )
    ax.set(
        xlabel="Ukuran N (sampel)", ylabel="Jenis window",
        title="Macro-F1 rata-rata — 16 Konfigurasi (overlap 50%)",
    )
    ax.grid(False)
    _simpan(fig, FIG_W2, "w2_f1_heatmap")


def t1_kurva_f1_vs_n(skor: pd.DataFrame) -> None:
    """Kurva macro-F1 vs ukuran N per jenis window (± 95% CI)."""
    g = (
        skor[skor.overlap == 0.5]
        .groupby(["window", "n"])["macro_f1"]
        .agg(["mean", _ci95])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for window in URUTAN_WINDOW:
        d = g[g.window == window]
        ax.errorbar(
            d["n"], d["mean"], yerr=d["_ci95"],
            marker="o", capsize=4, lw=1.8,
            color=WARNA_WINDOW[window], label=window,
        )
    ax.axvspan(512, 1024, color="gray", alpha=0.12, label="H1: N optimal")
    ax.set(
        xscale="log",
        xlabel="Ukuran N (sampel)",
        ylabel="Macro-F1 ± 95% CI",
        title="Macro-F1 vs Ukuran Window N",
    )
    ax.set_xticks(sorted(g.n.unique()))
    ax.set_xticklabels(sorted(g.n.unique()))
    ax.legend(fontsize=9)
    _simpan(fig, FIG_W2, "w2_f1_vs_n")


def t1_boxplot_f1(skor: pd.DataFrame) -> None:
    """Boxplot macro-F1 per konfigurasi (baseline = merah)."""
    urutan = (
        skor.groupby("config")["macro_f1"]
        .median().sort_values(ascending=False).index
    )
    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.boxplot(
        data=skor, x="config", y="macro_f1", order=urutan,
        ax=ax, color="#4fc3f7", linewidth=0.9, fliersize=3,
    )
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == NAMA_BASELINE:
            lbl.set_color("#e53935")
            lbl.set_fontweight("bold")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=7.5)
    ax.set(
        xlabel="Konfigurasi  (merah = baseline hann_N2048)",
        ylabel="Macro-F1",
        title="Sebaran Macro-F1 — 16 Konfigurasi",
    )
    _simpan(fig, FIG_W2, "w2_f1_boxplot")


def t1_tradeoff_akurasi_waktu(
    skor: pd.DataFrame, waktu: pd.DataFrame, folder=FIG_W2, nama_file: str = "w2_tradeoff"
) -> None:
    """Scatter Macro-F1 vs waktu komputasi per konfigurasi."""
    g = (
        skor.groupby("config")["macro_f1"]
        .agg(["mean", _ci95])
        .reset_index()
        .merge(waktu, on="config")
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        g["ms_per_clip"], g["mean"], yerr=g["_ci95"],
        fmt="o", capsize=3, color=PAL[1], alpha=0.8,
    )
    for row in g.itertuples():
        ax.annotate(
            row.config, (row.ms_per_clip, getattr(row, "mean")),
            fontsize=5.5, xytext=(3, 3), textcoords="offset points",
        )
    ax.set(
        xlabel="Waktu ekstraksi (ms/klip)",
        ylabel="Macro-F1 ± 95% CI",
        title="Trade-off Akurasi vs Biaya Komputasi",
    )
    _simpan(fig, folder, nama_file)


def t1_snr_berpasangan(snr: pd.DataFrame) -> None:
    """Scatter SNR sebelum vs sesudah bandpass + histogram ΔSNR."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    bmin = snr[["snr_before", "snr_after"]].min().min()
    bmax = snr[["snr_before", "snr_after"]].max().max()

    ax1.scatter(snr.snr_before, snr.snr_after, s=7, alpha=0.4, color=PAL[0])
    ax1.plot([bmin, bmax], [bmin, bmax], "r--", lw=1.2, label="y = x")
    ax1.set(
        xlabel="SNR sebelum bandpass (dB)",
        ylabel="SNR sesudah bandpass (dB)",
        title="SNR Sebelum vs Sesudah Bandpass",
    )
    ax1.legend()

    delta = snr.snr_after - snr.snr_before
    sns.histplot(delta, kde=True, ax=ax2, color=PAL[2])
    ax2.axvline(0, color="r", ls="--", lw=1.2)
    ax2.axvline(
        delta.median(), color=PAL[3], ls=":", lw=1.5,
        label=f"Median = {delta.median():.2f} dB",
    )
    ax2.set(xlabel="Δ SNR (dB)", title="Distribusi Selisih SNR")
    ax2.legend(fontsize=8)

    _simpan(fig, FIG_W2, "w2_snr_paired")


def t1_respons_window(n: int = 256, pad: int = 2**14) -> None:
    """Respons frekuensi 4 jenis window (domain frekuensi, dB)."""
    nama_window = {
        "rect": "boxcar", "hann": "hann",
        "hamming": "hamming", "blackman": "blackman",
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, nama in nama_window.items():
        r  = np.abs(np.fft.rfft(get_window(nama, n), pad))
        db = 20 * np.log10(r / r.max() + 1e-12)
        ax.plot(
            np.fft.rfftfreq(pad) * n, db,
            color=WARNA_WINDOW[label], lw=1.6, label=label,
        )
    ax.set(
        xlim=(0, 12), ylim=(-100, 3),
        xlabel="Frekuensi (bin DFT)",
        ylabel="Magnitudo (dB)",
        title=f"Respons Frekuensi Window  N={n}",
    )
    ax.legend(fontsize=9)
    _simpan(fig, FIG_W2, "w2_window_response")


def t1_respons_bandpass(sr: int = 22_050) -> None:
    """Respons frekuensi filter Butterworth bandpass 1–10 kHz."""
    sos     = butter(4, [1_000, 10_000], btype="band", fs=sr, output="sos")
    impulse = np.zeros(2048)
    impulse[0] = 1.0
    h       = sosfilt(sos, impulse)
    freqs   = np.fft.rfftfreq(len(h), d=1 / sr)
    H       = np.abs(np.fft.rfft(h))
    H_db    = 20 * np.log10(H / (H.max() + 1e-12))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs / 1000, H_db, color=PAL[1], lw=2)
    ax.axvspan(1, 10, alpha=0.10, color=PAL[2], label="Passband 1–10 kHz")
    ax.axhline(-3, color="gray", ls="--", lw=1, label="-3 dB")
    ax.set(
        xlim=(0, sr / 2000), ylim=(-80, 5),
        xlabel="Frekuensi (kHz)",
        ylabel="Magnitudo (dB)",
        title="Respons Frekuensi Filter Bandpass Butterworth Orde-4",
    )
    ax.legend(fontsize=9)
    _simpan(fig, FIG_W2, "w2_bandpass_response")


# ================================================================== MINGGU 3

def t1_spektrogram_contoh(
    manifest: pd.DataFrame, n: int = 1024, window: str = "hann",
) -> None:
    """Spektrogram STFT sebelum vs sesudah praproses."""
    import librosa
    from src.features import TARGET_SR, WINDOWS, praproses

    path = manifest[
        manifest["label"] == sorted(manifest["label"].unique())[0]
    ]["path"].iloc[0]
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True, duration=3.0)
    y_proc = praproses(y, sr, bandpass=True)

    S    = np.abs(librosa.stft(y_proc, n_fft=n, window=WINDOWS[window]))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Sebelum praproses
    S_raw = np.abs(librosa.stft(y, n_fft=n, window=WINDOWS[window]))
    img0  = librosa.display.specshow(
        librosa.amplitude_to_db(S_raw, ref=np.max),
        sr=sr, hop_length=n // 2,
        x_axis="time", y_axis="hz",
        ax=axes[0], cmap="magma",
    )
    axes[0].set_title("Spektrogram — Sinyal Asli")
    axes[0].set_ylabel("Frekuensi (Hz)")
    fig.colorbar(img0, ax=axes[0], format="%+2.0f dB")

    # Sesudah praproses
    img1 = librosa.display.specshow(
        S_db, sr=sr, hop_length=n // 2,
        x_axis="time", y_axis="hz",
        ax=axes[1], cmap="magma",
    )
    axes[1].set_title("Spektrogram — Sesudah Praproses")
    axes[1].set_ylabel("")
    fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")

    fig.tight_layout()
    _simpan(fig, FIG_W3, "w3_spektrogram_contoh")


def matriks_konfusi(
    cm: np.ndarray, label_kelas: list[str], nama_file: str,
) -> None:
    """Heatmap matriks konfusi ter-normalisasi."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        vmin=0, vmax=1, ax=ax,
        xticklabels=label_kelas,
        yticklabels=label_kelas,
        cbar_kws={"label": "Proporsi"},
    )
    ax.set(
        xlabel="Prediksi", ylabel="Kelas asli",
        title="Matriks Konfusi (Normalisasi Baris)",
    )
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _simpan(fig, FIG_W3, nama_file)


def t1_f1_per_kelas(df: pd.DataFrame, config: str) -> None:
    """Bar chart horizontal F1 per kelas (urut dari terendah)."""
    d = df.sort_values("f1")
    paleta = sns.color_palette("RdYlGn", len(d))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.barh(d["class"], d["f1"], color=paleta)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.axvline(0.5, color="gray", ls="--", lw=1)
    ax.set(xlim=(0, 1.08), xlabel="F1-score",
           title=f"F1 per Kelas — {config}")
    _simpan(fig, FIG_W3, "w3_f1_per_class")


def t1_mfcc_contoh(
    manifest: pd.DataFrame, config_nama: str = "hann_N1024_ov50",
) -> None:
    """Visualisasi MFCC dari konfigurasi terbaik."""
    import librosa
    from src.features import TARGET_SR, CONFIGS, praproses, WINDOWS, N_MFCC, N_MELS

    cfg  = next(c for c in CONFIGS if c.nama == config_nama)
    path = manifest["path"].iloc[0]
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    y = praproses(y, sr, bandpass=True)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC,
        n_fft=cfg.n, hop_length=cfg.hop,
        window=WINDOWS[cfg.window], n_mels=N_MELS,
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    img = librosa.display.specshow(mfcc, x_axis="time", ax=ax, cmap="coolwarm")
    fig.colorbar(img, ax=ax, label="Amplitudo MFCC")
    ax.set(ylabel="Koefisien MFCC", title=f"MFCC — {config_nama}")
    _simpan(fig, FIG_W3, f"w3_mfcc_{config_nama}")
