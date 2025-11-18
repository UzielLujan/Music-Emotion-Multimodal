# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:47:39 2025

@author: Brenda Tránsito
"""

# scripts/scripts06_merge_lyrics.py
import os
import pandas as pd

def merge_lyrics(df_va, processed_dir):
    """
    Une características musicales + valence/arousal con las letras
    guardadas en un archivo .csv.
    """

    lyrics_path = os.path.join(processed_dir, "lyrics.csv")

    if not os.path.exists(lyrics_path):
        raise FileNotFoundError(f"No se encontró {lyrics_path}")

    # Leer las letras
    df_lyrics = pd.read_csv(lyrics_path)

    # Asegurar que track_id es numérico
    df_lyrics["track_id"] = pd.to_numeric(df_lyrics["track_id"], errors="coerce")

    merged = df_va.merge(df_lyrics, on="track_id", how="left")

    print(f"Letras unidas: {merged['lyrics'].notna().sum()} tracks con letra")
    print(f"Total final: {len(merged)} filas")

    return merged

