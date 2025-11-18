# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 16:02:44 2025

@author: Brenda Tránsito
"""

# scripts/scripts02_filter_english_medium.py
import os
import pandas as pd

def filter_english_tracks(metadata_dir):
    tracks_path = os.path.join(metadata_dir, "tracks.csv")

    tracks = pd.read_csv(
        tracks_path,
        index_col=0,
        header=[0,1]  # MultiIndex
    )

    # Asegurar que la columna existe
    col_lang = ("track", "language_code")

    if col_lang not in tracks.columns:
        raise ValueError(f"No se encontró la columna {col_lang} en tracks.csv")

    english = tracks[tracks[col_lang] == "en"]

    print(f"Tracks inglés: {len(english)}")

    return english.index.tolist()
