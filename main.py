# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 16:04:48 2025

@author: Brenda Tránsito
"""

from scripts02_filter_english import filter_english_tracks
from scripts03_extract_audio_features import extract_audio_features
from scripts04_generate_spectrograms import generate_spectrograms
from scripts05_add_valence_arousal import add_valence_arousal
from scripts06_merge_lyrics import merge_lyrics
from scripts07_merge_lyrics_kaggle import merge_kaggle_lyrics



import os
import pandas as pd

# === RUTAS ===
RAW = r"C:/Users/Brenda Tránsito/Documents/Maestría/Tercer Semestre/MIR/Proyecto/Proyecto_MIR/data/raw"
LR=r"C:/Users/Brenda Tránsito/Documents/Maestría/Tercer Semestre/MIR/Proyecto/Proyecto_MIR/data/processed"
META = os.path.join(RAW, "fma_metadata")
AUDIO = os.path.join(RAW, "fma_medium")
PROCESSED = r"C:/Users/Brenda Tránsito/Documents/Maestría/Tercer Semestre/MIR/Proyecto/Proyecto_MIR/data/processed"
SPEC_DIR = os.path.join(RAW, "spectrograms_medium")

# === 1. Filtrar inglés ===
english_ids = filter_english_tracks(META)

# === 2. Audio features ===
df_feats = extract_audio_features(english_ids, AUDIO)

# === 3. Espectrogramas ===
generate_spectrograms(english_ids, AUDIO, SPEC_DIR)

# === 4. Valence & arousal ===
df_va = add_valence_arousal(df_feats, META)


# === 5. Unir letras ===
df_final = merge_lyrics(df_va, PROCESSED)

KAGGLE_LYRICS = r"C:/Users/Brenda Tránsito/Documents/Maestría/Tercer Semestre/MIR/Proyecto/Proyecto_MIR/data/external/kaggle_lyrics/lyrics.csv"

df_with_lyrics = merge_kaggle_lyrics(df_va, KAGGLE_LYRICS)

df_with_lyrics.to_csv("data/processed/fma_with_lyrics.csv", index=False)


df_final.to_csv(os.path.join(RAW, "df_fma_medium_final.csv"), index=False)

print("=== PIPELINE COMPLETO CON FMA_MEDIUM ===")
print(df_final.head())

