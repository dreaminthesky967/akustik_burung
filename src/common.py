"""
Utilitas bersama: seed, fold, statistik, dan pengecekan kualitas
"""

from __future__ import annotations

import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# - path root
ROOT    = Path(__file__).resolve().parents[1]
RAW     = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

# Struktur folder hasil 
RESULTS = ROOT / "results"

FIG_W1  = RESULTS / "minggu_1" / "figures"
TAB_W1  = RESULTS / "minggu_1" / "tables"

FIG_W2  = RESULTS / "minggu_2" / "figures"
TAB_W2  = RESULTS / "minggu_2" / "tables"

FIG_W3  = RESULTS / "minggu_3" / "figures"
TAB_W3  = RESULTS / "minggu_3" / "tables"

for _d in (INTERIM, FIG_W1, TAB_W1, FIG_W2, TAB_W2, FIG_W3, TAB_W3):
    _d.mkdir(parents=True, exist_ok=True)

# konstanta
SEEDS: tuple[int, ...] = (0, 1, 2)
N_SPLITS: int = 5
N_CONFIGS: int = 16   # 4 window × 4 ukuran N


# reproduksibilitas

def atur_seed(seed: int) -> None:
    """Menetapkan seed global untuk numpy dan random agar hasil bisa diulang."""
    random.seed(seed)
    np.random.seed(seed)


def catat_lingkungan(nama: str) -> None:
    """Simpan versi Python dan pustaka ke results/minggu_1/tables/env_<nama>.json."""
    import librosa
    import scipy
    import sklearn

    info = {
        "python"  : sys.version,
        "platform": platform.platform(),
        "numpy"   : np.__version__,
        "scipy"   : scipy.__version__,
        "librosa" : librosa.__version__,
        "sklearn" : sklearn.__version__,
    }
    (TAB_W1 / f"env_{nama}.json").write_text(json.dumps(info, indent=2))


# fold

def buat_fold(y, groups=None, seed: int = 0, n_splits: int = N_SPLITS):
    """
    Membagi data menjadi n_splits fold stratified.

    Jika ``groups`` diberikan, dipakai StratifiedGroupKFold sehingga
    klip dari rekaman yang sama tidak terpecah antar fold.

    Returns
    -------
    list[(train_idx, test_idx)]
    """
    y = np.asarray(y)
    if groups is None:
        pembagi = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        return list(pembagi.split(np.zeros(len(y)), y))

    pembagi = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    folds   = list(pembagi.split(np.zeros(len(y)), y, groups))
    periksa_tidak_bocor(folds, groups)
    return folds


def periksa_tidak_bocor(folds, groups) -> None:
    """Pastikan tidak ada grup yang muncul di train sekaligus di test."""
    groups = np.asarray(groups)
    for k, (tr, te) in enumerate(folds):
        overlap = set(groups[tr]) & set(groups[te])
        assert not overlap, (
            f"Fold {k}: {len(overlap)} grup bocor antara train dan test"
        )


def simpan_fold(folds_per_seed: dict, path: Path) -> None:
    """Serialisasi dict {seed: [(train, test), ...]} ke JSON."""
    payload = {
        str(s): [[tr.tolist(), te.tolist()] for tr, te in f]
        for s, f in folds_per_seed.items()
    }
    path.write_text(json.dumps(payload))


def muat_fold(path: Path) -> dict[int, list]:
    """Muat fold dari JSON dan kembalikan {seed(int): [(np.array, np.array), ...]}."""
    raw = json.loads(path.read_text())
    return {
        int(s): [(np.array(tr), np.array(te)) for tr, te in f]
        for s, f in raw.items()
    }


# ================================================================== pengecekan kualitas

def periksa_fitur(X: np.ndarray, n_sampel: int, n_dim: int | None = None) -> None:
    """
    Periksa dimensi dan nilai fitur.

    Raises
    ------
    AssertionError jika ada masalah dimensi, NaN, atau Inf.
    """
    assert X.shape[0] == n_sampel, (
        f"jumlah baris {X.shape[0]} != {n_sampel}"
    )
    if n_dim is not None:
        assert X.shape[1] == n_dim, (
            f"dimensi fitur {X.shape[1]} != {n_dim}"
        )
    assert np.isfinite(X).all(), (
        "Ada NaN/Inf pada fitur; cari klip rusak atau senyap total"
    )


def uji_sanity(X, y, buat_model, folds, seed: int = 0) -> dict:
    """
    Jalankan SEBELUM eksperimen besar untuk mendeteksi bug lebih awal.

    Tiga pengujian:
    1. DummyClassifier (prior kelas) → batas bawah.
    2. Label train diacak → macro-F1 harus mendekati peluang acak.
    3. Model asli → harus jelas di atas keduanya.

    Returns
    -------
    dict {"dummy": float, "shuffled": float, "real": float}
    """
    y   = np.asarray(y)
    tr, te = folds[0]
    rng = np.random.default_rng(seed)

    # 1. dummy
    dummy   = DummyClassifier(strategy="prior").fit(X[tr], y[tr])
    f_dummy = f1_score(y[te], dummy.predict(X[te]), average="macro", zero_division=0)

    # 2. label diacak
    y_acak  = rng.permutation(y[tr])
    f_acak  = f1_score(
        y[te],
        buat_model(seed).fit(X[tr], y_acak).predict(X[te]),
        average="macro",
        zero_division=0,
    )

    # 3. model asli
    f_asli  = f1_score(
        y[te],
        buat_model(seed).fit(X[tr], y[tr]).predict(X[te]),
        average="macro",
        zero_division=0,
    )

    peluang_acak = 1.0 / len(np.unique(y))
    assert f_asli > f_dummy + 0.05, (
        "Model tidak lebih baik dari dummy: cek fitur/label"
    )
    assert f_acak < peluang_acak + 0.15, (
        "Label teracak masih tinggi: ada kebocoran data"
    )
    return {"dummy": f_dummy, "shuffled": f_acak, "real": f_asli}


# ================================================================== statistik

def rank_biserial(x, y) -> float:
    """
    Ukuran efek Wilcoxon berpasangan.

    Returns
    -------
    float dalam rentang [-1, 1]; positif artinya x > y.
    """
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def kendall_w(chi2: float, n_blok: int, k: int) -> float:
    """Ukuran efek Friedman (Kendall W)."""
    return chi2 / (n_blok * (k - 1))


def friedman_dan_posthoc(
    df: pd.DataFrame,
    kolom_perlakuan: str,
    kolom_nilai: str,
    blok: tuple = ("seed", "fold"),
):
    """
    Uji Friedman + post-hoc Nemenyi pada data long-format.

    Returns
    -------
    (chi2, p, kendall_w, DataFrame p-value Nemenyi)
    """
    import scikit_posthocs as sp

    wide = df.pivot_table(
        index=list(blok), columns=kolom_perlakuan, values=kolom_nilai
    )
    assert not wide.isna().any().any(), (
        "Blok tidak lengkap: ada konfigurasi yang belum selesai"
    )
    chi2, p = friedmanchisquare(*[wide[c] for c in wide.columns])
    post = sp.posthoc_nemenyi_friedman(wide.to_numpy())
    post.index = post.columns = wide.columns
    w = kendall_w(chi2, len(wide), wide.shape[1])
    return chi2, p, w, post
