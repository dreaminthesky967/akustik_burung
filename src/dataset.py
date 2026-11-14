"""
Modul Minggu 1 (Hari 1-2):
Pemindaian dataset Mendeley, pencatatan sample rate (fs) & durasi,
serta pembuatan skema 5-fold stratified cross-validation (3 seed).
"""

from pathlib import Path
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.config import CONFIG, RAW_DATA_DIR, META_DIR, TABLE_DIR


def is_valid_audio_file(file_path: Path) -> bool:
    """Mengabaikan folder __MACOSX dan file tersembunyi macOS (diawali '._' atau '.')."""
    if "__MACOSX" in file_path.parts:
        return False
    if file_path.name.startswith("._") or file_path.name.startswith("."):
        return False
    return True


def scan_dataset(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Memindai seluruh file audio valid dan mencatat metadata (kelas, fs, kanal, durasi)."""
    extensions = ("*.wav", "*.WAV", "*.mp3", "*.flac")
    candidates = []
    for ext in extensions:
        candidates.extend(raw_dir.rglob(ext))

    audio_files = sorted([f for f in candidates if is_valid_audio_file(f)])

    if not audio_files:
        raise FileNotFoundError(
            f"Tidak ditemukan file audio di {raw_dir}. "
            "Pastikan folder 'bird_sounds' sudah berisi file audio."
        )

    records = []
    for file_path in tqdm(audio_files, desc="Memindai file audio"):
        label = file_path.parent.name.lower().strip().replace(" ", "_").replace("-", "_")
        try:
            info = sf.info(str(file_path))
            records.append({
                "relative_path": str(file_path.relative_to(raw_dir)),
                "label": label,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "duration_sec": round(info.duration, 4),
            })
        except RuntimeError as err:
            print(f"[LEWATI] File rusak/tidak terbaca ({file_path.name}): {err}")

    return pd.DataFrame(records)


def create_stratified_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Menyusun pembagian 5-fold stratified untuk 3 random seed berbeda."""
    df_out = df.copy()
    for seed in CONFIG.random_seeds:
        col_name = f"fold_seed_{seed}"
        df_out[col_name] = -1
        skf = StratifiedKFold(n_splits=CONFIG.n_folds, shuffle=True, random_state=seed)
        for fold_idx, (_, val_idx) in enumerate(skf.split(df_out, df_out["label"])):
            df_out.loc[val_idx, col_name] = fold_idx

    return df_out


def run_data_verification() -> pd.DataFrame:
    # Menjalankan verifikasi data penuh dan menyimpan hasil
    df = scan_dataset()
    df = create_stratified_folds(df)

    # 1. Simpan metadata lengkap + 5-fold split
    meta_csv_path = META_DIR / "metadata_folds.csv"
    df.to_csv(meta_csv_path, index=False)

    # 2. Simpan tabel ringkasan distribusi kelas & sample rate
    class_summary = (
        df.groupby(["label", "sample_rate"])
        .agg(
            jumlah_klip=("relative_path", "count"),
            durasi_rata2_dtk=("duration_sec", "mean"),
            durasi_min_dtk=("duration_sec", "min"),
            durasi_max_dtk=("duration_sec", "max"),
        )
        .reset_index()
    )
    summary_csv_path = TABLE_DIR / "ringkasan_distribusi_dataset.csv"
    class_summary.to_csv(summary_csv_path, index=False)

    print("\nRINGKASAN DISTRIBUSI KELAS & SAMPLE RATE (fs)")
    print(class_summary.to_string(index=False))
    print(f"\n[SUKSES] Metadata & 5-fold tersimpan di : {meta_csv_path}")
    print(f"[SUKSES] Tabel ringkasan kelas di       : {summary_csv_path}")
    return df