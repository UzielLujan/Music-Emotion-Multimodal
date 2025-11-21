# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 16:04:48 2025

@author: Brenda Tránsito
"""
from scripts01_download_fma_medium import extract_fma_medium
from scripts02_filter_english import filter_english_tracks
from scripts03_extract_audio_features import extract_audio_features
from scripts04_generate_spectrograms import generate_spectrograms
from scripts05_add_valence_arousal import add_valence_arousal
from scripts05_merge_valence import merge_valence_arousal
from scripts05_predict_emotions import add_emotion_labels
from script06_lyrics import fetch_lyrics
from scripts06_merge_lyrics import merge_lyrics
from scripts07_merge_lyrics_kaggle import merge_kaggle_lyrics


# Uso de rutas relativas y autodetección de carpetas
import os
import pandas as pd
from pathlib import Path

# Directorio base: carpeta donde está este script
BASE_DIR = Path(__file__).resolve().parent  # src/ExtractData1.0/.. = src/
BASE_OR_DIR = BASE_DIR.parent  # src/.. = Music-Emotion-Multimodal/
BASE_DATA_DIR = BASE_OR_DIR.parent
DATA_DIR = BASE_DATA_DIR / 'data'
RAW = DATA_DIR / 'raw'
PROCESSED = DATA_DIR / 'processed'
META = DATA_DIR / 'raw' /'fma_metadata'
AUDIO = RAW / 'fma_medium'
SPEC_DIR = DATA_DIR /'spectrograms_medium'

# Si alguna carpeta no existe, mostrar advertencia
for folder in [RAW, PROCESSED, META, AUDIO, SPEC_DIR]:
	if not folder.exists():
		print(f"[ADVERTENCIA] La carpeta no existe: {folder}")

META = str(META)
AUDIO = str(AUDIO)
SPEC_DIR = str(SPEC_DIR)

#=== PIPELINE COMPLETO CON FMA_MEDIUM ===#

#extract_fma_medium()

# === 1. Filtrar inglés ===
english_ids = filter_english_tracks(META)

# === 2. Audio features ===
df_feats = extract_audio_features(english_ids, AUDIO)

# === 3. Espectrogramas ===
generate_spectrograms(english_ids, AUDIO, SPEC_DIR)

# === 4. Valence & arousal ===
df_va = add_valence_arousal(df_feats, META)

# === 5. Merge to df_feats ===
df_merged = merge_valence_arousal(df_feats, META)

# === 6. PREDICT EMOTIONS ===

df_va = add_emotion_labels(df_merged)

# === 7. Descargar letras de Genius ===

#fetch_lyrics(english_ids, META)

# === 7. Unir letras ===
df_final = merge_lyrics(df_va, PROCESSED)
