# -*- coding: utf-8 -*-
"""
Cruza english_real con valence/arousal del dataset Echonest.

Autor: Uziel Luján (Uzi)
"""
from pathlib import Path
import pandas as pd

def load_valence_arousal(meta_dir):
    meta_dir = Path(meta_dir)
    echo_path = meta_dir / "echonest.csv"

    df = pd.read_csv(echo_path, index_col=0, header=[0,1,2])

    val_col = ("echonest", "audio_features", "valence")
    aro_col = ("echonest", "audio_features", "energy")  # <- arousal real

    df_val = df[[val_col, aro_col]].copy()
    df_val.columns = ["valence", "arousal"]
    df_val = df_val.dropna()

    va_ids = set(df_val.index)

    print(f"[✔] Tracks con valence/arousal (valence + energy): {len(va_ids)}")
    return df_val, va_ids


def cross_with_english_real(english_real_ids, va_ids):
    english_real_ids = set(english_real_ids)
    va_ids = set(va_ids)

    intersection = english_real_ids & va_ids

    print("\n=== Intersección ENGLISH_REAL ∩ VA ===")
    print(f"Inglés real: {len(english_real_ids)}")
    print(f"Valence/arousal: {len(va_ids)}")
    print(f"---------------------------------------")
    print(f"Total tracks con ambas: {len(intersection)}\n")

    return intersection
