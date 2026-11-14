from src.config import CONFIG, RAW_DATA_DIR, TABLE_DIR
from src.dataset import run_data_verification
from src.preprocess import (
    load_and_resample,
    apply_pre_emphasis,
    design_iir_butterworth,
    apply_iir_filter,
    plot_filter_response_comparison,
    plot_sample_preprocessing_effect,
)
from src.features import save_stft_resolution_table, extract_features_single_clip
import pandas as pd


def main() -> None:
    
    # TAHAP 1: Verifikasi Dataset, Cek Sample Rate (fs), & Buat 5-Fold Split
    df_meta = run_data_verification()

    # TAHAP 2: Desain Filter Bandpass (IIR vs FIR) & Plot Respons Frekuensi/Fase
    plot_filter_response_comparison()

    # Ambil 1 sampel audio pertama untuk uji coba pipeline praproses & ekstraksi fitur
    sample_rel_path = df_meta.iloc[0]["relative_path"]
    sample_full_path = str(RAW_DATA_DIR / sample_rel_path)
    plot_sample_preprocessing_effect(sample_full_path)

    # TAHAP 3: Tabel Resolusi STFT & Verifikasi Skrip Ekstraksi 16 Konfigurasi Window
    save_stft_resolution_table()

    raw_wav = load_and_resample(sample_full_path)
    filt_wav = apply_iir_filter(apply_pre_emphasis(raw_wav), design_iir_butterworth())

    test_logs = []
    for n_fft in CONFIG.window_sizes:
        for win_type in CONFIG.window_types:
            feat_vec, mel_mat, dt_ms = extract_features_single_clip(
                filt_wav, n_fft=n_fft, hop_length=n_fft // 2, window_type=win_type
            )
            test_logs.append({
                "window_type": win_type,
                "window_size_N": n_fft,
                "hop_length": n_fft // 2,
                "mfcc_vector_dim": feat_vec.shape[0],
                "mel_spec_shape": f"{mel_mat.shape[0]}x{mel_mat.shape[1]}",
                "waktu_ekstraksi_ms": round(dt_ms, 3),
            })

    df_test_feat = pd.DataFrame(test_logs)
    feat_check_csv = TABLE_DIR / "minggu1_uji_ekstraksi_16konfigurasi.csv"
    df_test_feat.to_csv(feat_check_csv, index=False)

    print("\n=== BUKTI UJI JALAN EKSTRAKSI FITUR (16 KONFIGURASI WINDOW) ===")
    print(df_test_feat.to_string(index=False))
    print("\n[SELESAI] Seluruh tugas Minggu 1 berhasil dijalankan!")


if __name__ == "__main__":
    main()