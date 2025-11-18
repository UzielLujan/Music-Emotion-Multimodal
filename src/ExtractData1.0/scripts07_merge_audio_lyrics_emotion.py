# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 09:37:04 2025

@author: Brenda Tránsito
"""

import os
import pandas as pd

def merge_all(df_audio_va, lyrics_path="C:/Users/Brenda Tránsito/Documents/Maestría/Tercer Semestre/MIR/Proyecto/Proyecto_MIR/data/processed/lyrics.csv"):
    """
    Une en un solo DataFrame:
    - Características de audio (df_audio_va)
    - Valence y arousal
    - Letras (lyrics.csv)
    """

    # Cargar letras
    df_lyrics = pd.read_csv(lyrics_path)

    # Asegurar consistencia en track_id
    df_lyrics["track_id"] = df_lyrics["track_id"].astype(int)
    df_audio_va = df_audio_va.copy()
    df_audio_va.index = df_audio_va.index.astype(int)

    # Unir por track_id
    df_final = df_audio_va.join(
        df_lyrics.set_index("track_id"),
        how="inner"
    )

    print("=== Dimensiones del dataset ===")
    print("Audio + VA:", df_audio_va.shape)
    print("Letras:", df_lyrics.shape)
    print("Final:", df_final.shape)
    
    return df_final
