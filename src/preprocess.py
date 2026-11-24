"""
Praproses Sinyal (Resampling, Pre-emphasis) dan Desain Bandpass Filter 1-10 kHz.
Perbandingan IIR Butterworth Orde 4 vs FIR Orde Setara.
Output disimpan terstruktur ke results/minggu_1/figures/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import librosa

from src.config import CONFIG
from src.common import FIG_W1


def load_and_resample(
    file_path: str, target_fs: int = CONFIG.target_fs,
) -> np.ndarray:
    """Memuat klip audio sebagai sinyal mono dan menyeragamkan sample rate."""
    waveform, _ = librosa.load(file_path, sr=target_fs, mono=True)
    return waveform


def apply_pre_emphasis(
    waveform: np.ndarray, coef: float = CONFIG.pre_emphasis_coef,
) -> np.ndarray:
    """Menerapkan pre-emphasis: y[n] = x[n] - 0,97·x[n-1]."""
    return np.append(waveform[0], waveform[1:] - coef * waveform[:-1])


def design_iir_butterworth(
    fs: int = CONFIG.target_fs,
    low_hz: float = CONFIG.lowcut_hz,
    high_hz: float = CONFIG.highcut_hz,
    order: int = CONFIG.iir_order,
) -> np.ndarray:
    """Merancang Bandpass Butterworth IIR (Second-Order Sections / SOS)."""
    nyquist = 0.5 * fs
    low  = low_hz / nyquist
    high = min(high_hz / nyquist, 0.99)
    return signal.butter(order, [low, high], btype="bandpass", output="sos")


def design_fir_bandpass(
    fs: int = CONFIG.target_fs,
    low_hz: float = CONFIG.lowcut_hz,
    high_hz: float = CONFIG.highcut_hz,
    numtaps: int = CONFIG.fir_numtaps,
) -> np.ndarray:
    """Merancang Bandpass FIR menggunakan metode windowing Hamming."""
    nyquist = 0.5 * fs
    low  = low_hz / nyquist
    high = min(high_hz / nyquist, 0.99)
    return signal.firwin(numtaps, [low, high], pass_zero=False, window="hamming")


def apply_iir_filter(waveform: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Menyaring sinyal dengan filter IIR Butterworth."""
    return signal.sosfilt(sos, waveform)


def apply_fir_filter(waveform: np.ndarray, fir_taps: np.ndarray) -> np.ndarray:
    """Menyaring sinyal dengan filter FIR."""
    return signal.lfilter(fir_taps, [1.0], waveform)


def plot_filter_response_comparison(out_path: Path | None = None) -> Path:
    """Membuat grafik perbandingan respons magnitudo dan fase IIR vs FIR."""
    if out_path is None:
        out_path = FIG_W1 / "minggu1_respons_filter_iir_vs_fir.png"

    sos_iir  = design_iir_butterworth()
    taps_fir = design_fir_bandpass()

    w_iir, h_iir = signal.sosfreqz(sos_iir, worN=4096, fs=CONFIG.target_fs)
    w_fir, h_fir = signal.freqz(taps_fir, [1.0], worN=4096, fs=CONFIG.target_fs)

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Respons magnitudo
    ax_mag.plot(
        w_iir, 20 * np.log10(np.maximum(np.abs(h_iir), 1e-8)),
        label=f"IIR Butterworth (Orde {CONFIG.iir_order})", lw=2,
    )
    ax_mag.plot(
        w_fir, 20 * np.log10(np.maximum(np.abs(h_fir), 1e-8)),
        label=f"FIR Hamming ({CONFIG.fir_numtaps} Taps)", lw=1.8, ls="--",
    )
    ax_mag.axvline(CONFIG.lowcut_hz, color="red", ls=":", alpha=0.7,
                   label="Cutoff 1 kHz & 10 kHz")
    ax_mag.axvline(CONFIG.highcut_hz, color="red", ls=":", alpha=0.7)
    ax_mag.set_title(
        "Perbandingan Respons Frekuensi & Fase: Bandpass IIR Butterworth vs FIR (1–10 kHz)"
    )
    ax_mag.set_ylabel("Magnitudo (dB)")
    ax_mag.set_ylim(-80, 5)
    ax_mag.grid(True, alpha=0.3)
    ax_mag.legend(loc="lower center")

    # Respons fase
    ax_phase.plot(
        w_iir, np.unwrap(np.angle(h_iir)),
        label="Fase IIR (Non-linear)", lw=2,
    )
    ax_phase.plot(
        w_fir, np.unwrap(np.angle(h_fir)),
        label="Fase FIR (Linear di Passband)", lw=1.8, ls="--",
    )
    ax_phase.set_xlabel("Frekuensi (Hz)")
    ax_phase.set_ylabel("Fase Unwrapped (Radian)")
    ax_phase.grid(True, alpha=0.3)
    ax_phase.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_sample_preprocessing_effect(
    sample_wav_path: str, out_path: Path | None = None
) -> Path:
    """Memvisualisasikan sinyal sampel asli vs setelah praproses."""
    if out_path is None:
        out_path = FIG_W1 / "minggu1_efek_praproses_psd.png"

    raw_sig  = load_and_resample(sample_wav_path)
    pre_sig  = apply_pre_emphasis(raw_sig)
    sos      = design_iir_butterworth()
    filt_sig = apply_iir_filter(pre_sig, sos)

    freqs_raw, psd_raw   = signal.welch(raw_sig, fs=CONFIG.target_fs, nperseg=1024)
    freqs_filt, psd_filt = signal.welch(filt_sig, fs=CONFIG.target_fs, nperseg=1024)

    plt.figure(figsize=(9, 4.5))
    plt.semilogy(freqs_raw, psd_raw,
                 label="Sinyal Asli (Sebelum Praproses)", alpha=0.7)
    plt.semilogy(freqs_filt, psd_filt,
                 label="Setelah Pre-emphasis + Bandpass 1–10 kHz", lw=1.8)
    plt.axvline(CONFIG.lowcut_hz, color="red", ls=":", label="Batas Pita 1–10 kHz")
    plt.axvline(CONFIG.highcut_hz, color="red", ls=":")
    plt.title("Perbandingan Spektrum Daya (PSD) Sebelum vs Sesudah Praproses")
    plt.xlabel("Frekuensi (Hz)")
    plt.ylabel("Power Spectral Density (V²/Hz)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path
