from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# path direktori
BASE_DIR     = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "bird_sounds" / "training"
META_DIR     = BASE_DIR / "data" / "interim"

# Subfolder 
RESULTS_DIR  = BASE_DIR / "results"
W1_DIR       = RESULTS_DIR / "minggu_1"
W2_DIR       = RESULTS_DIR / "minggu_2"
W3_DIR       = RESULTS_DIR / "minggu_3"

FIG_W1 = W1_DIR / "figures"
TAB_W1 = W1_DIR / "tables"
FIG_W2 = W2_DIR / "figures"
TAB_W2 = W2_DIR / "tables"
FIG_W3 = W3_DIR / "figures"
TAB_W3 = W3_DIR / "tables"

# Backward compatibility alias
FIG_DIR   = FIG_W1
TABLE_DIR = TAB_W1

for _folder in [META_DIR, FIG_W1, TAB_W1, FIG_W2, TAB_W2, FIG_W3, TAB_W3]:
    _folder.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Week1Config:
    """Parameter audio dan praproses sesuai standar metodologi Topik 1."""

    # Audio
    target_fs         : int   = 22_050
    pre_emphasis_coef : float = 0.97

    # Filter bandpass 1–10 kHz
    lowcut_hz  : float = 1_000.0
    highcut_hz : float = 10_000.0
    iir_order  : int   = 4
    fir_numtaps: int   = 101

    # STFT & fitur — 4 ukuran N × 4 jenis window = 16 konfigurasi (overlap 50%)
    #  disuruh kenzo konfigurasi kalo salah dikick
    window_sizes: Tuple[int, ...] = (256, 512, 1024, 2048)
    window_types: Tuple[str, ...] = ("boxcar", "hann", "hamming", "blackman")
    n_mels      : int             = 40
    n_mfcc      : int             = 13

    # Validasi silang
    n_folds     : int             = 5
    random_seeds: Tuple[int, ...] = (0, 1, 2)


CONFIG = Week1Config()
