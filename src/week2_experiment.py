"""
16 konfigurasi × 5-fold × 3 seed = 240 evaluasi fold.
kato kenzo fiks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.common import (
    INTERIM, N_CONFIGS, N_SPLITS, SEEDS, TAB_W2,
    friedman_dan_posthoc, muat_fold, uji_sanity,
)
from src.features import BASELINE, CONFIGS
from src import plots


def buat_model(seed: int) -> Pipeline:
    """Pipeline StandardScaler + SVM RBF."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel       = "rbf",
            C            = 10,
            gamma        = "scale",
            class_weight = "balanced",
            random_state = seed,
        )),
    ])


def evaluasi_satu_fold(
    nama_config: str,
    X          : np.ndarray,
    y          : np.ndarray,
    label_kelas: list[str],
    seed       : int,
    fold       : int,
    idx_train  : np.ndarray,
    idx_test   : np.ndarray,
) -> dict:
    """Latih dan evaluasi model pada satu pasang fold."""
    prediksi = (
        buat_model(seed)
        .fit(X[idx_train], y[idx_train])
        .predict(X[idx_test])
    )
    baris = {
        "config"  : nama_config,
        "seed"    : seed,
        "fold"    : fold,
        "macro_f1": f1_score(y[idx_test], prediksi, average="macro", zero_division=0),
        "accuracy": accuracy_score(y[idx_test], prediksi),
    }
    f1_tiap_kelas = f1_score(
        y[idx_test], prediksi,
        labels        = label_kelas,
        average       = None,
        zero_division = 0,
    )
    baris.update({f"f1_{kelas}": v for kelas, v in zip(label_kelas, f1_tiap_kelas)})
    return baris


if __name__ == "__main__":
    manifest    = pd.read_csv(INTERIM / "t1_manifest.csv")
    y           = manifest["label"].to_numpy()
    label_kelas = sorted(set(y))
    fitur       = np.load(INTERIM / "t1_features.npz")
    folds       = muat_fold(INTERIM / "t1_folds.json")

    target_csv  = TAB_W2 / "t1_fold_scores.csv"
    n_evaluasi  = N_CONFIGS * len(SEEDS) * N_SPLITS   # 16 × 3 × 5 = 240

    if target_csv.exists() and len(pd.read_csv(target_csv)) == n_evaluasi:
        df = pd.read_csv(target_csv)
    else:
        # Sanity check ringkas
        uji_sanity(fitur[BASELINE.nama], y, buat_model, folds[0])

        pekerjaan = [
            delayed(evaluasi_satu_fold)(
                cfg.nama, fitur[cfg.nama], y, label_kelas,
                seed, k, tr, te,
            )
            for cfg in CONFIGS
            for seed, daftar_fold in folds.items()
            for k, (tr, te) in enumerate(daftar_fold)
        ]
        df = pd.DataFrame(Parallel(n_jobs=-2, verbose=0)(pekerjaan))

        # Tambahkan metadata konfigurasi
        meta = pd.DataFrame([
            {"config": c.nama, "window": c.window, "n": c.n, "overlap": c.overlap}
            for c in CONFIGS
        ])
        df = df.merge(meta, on="config")
        df.to_csv(target_csv, index=False)

    # Uji Friedman & Nemenyi
    chi2, p, w, post = friedman_dan_posthoc(df, "config", "macro_f1")
    s_stat = pd.Series({"chi2": chi2, "p": p, "kendall_w": w})
    s_stat.to_csv(TAB_W2 / "t1_friedman.csv")
    post.to_csv(TAB_W2 / "t1_nemenyi.csv")

    # Plot domain sistem: respons window & respons filter bandpass (Minggu 2)
    plots.t1_respons_window(n=256)
    plots.t1_respons_bandpass(sr=22_050)

    print(
        f"[Minggu 2 - Experiment] Selesai: "
        f"{n_evaluasi} evaluasi fold (chi2={chi2:.2f}, p={p:.2e})."
    )
    print(f"                       Output tersimpan di results/minggu_2/ (figures & tables).")
