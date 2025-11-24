"""
@author: Brenda Tránsito and Uziel Luján
"""

from scripts01_download_fma import download_fma_medium
from scripts01_extract_fma import extract_fma_medium_and_metadata
from explore_fma import explore_fma_metadata
'''
from scripts02_filter_english import filter_english_tracks
from scripts03_extract_audio_features import extract_audio_features
from scripts04_generate_spectrograms import generate_spectrograms
from scripts05_add_valence_arousal import add_valence_arousal
from scripts05_merge_valence import merge_valence_arousal
from scripts05_predict_emotions import add_emotion_labels
from scripts06_lyrics import fetch_lyrics
from scripts06_merge_lyrics import merge_lyrics
from scripts07_merge_lyrics_kaggle import merge_kaggle_lyrics
'''

# Uso de rutas relativas y autodetección de carpetas
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Detecta automáticamente la raíz del proyecto (buscando la carpeta 'data')
def find_project_root(starting_point: Path = None) -> Path:
    current = starting_point or Path(__file__).resolve()
    while current != current.parent:
        if (current / "data").exists():
            return current
        current = current.parent
    raise FileNotFoundError("No se encontró carpeta 'data' en la jerarquía superior.")

# === Definir rutas base ===
PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"
META = RAW / "fma_metadata"
AUDIO = RAW / "fma_medium"
SPEC_DIR = DATA_DIR / "spectrograms_medium"
FEATURES_DIR = DATA_DIR / "features"
LYRICS_PATH = DATA_DIR / "processed" / "lyrics.csv"

'''
# Crear directorios si no existen
for folder in [RAW, PROCESSED, META, AUDIO, SPEC_DIR, FEATURES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Si alguna carpeta no existe, mostrar advertencia
for folder in [RAW, PROCESSED, META, AUDIO, SPEC_DIR, FEATURES_DIR]:
	if not folder.exists():
		print(f"[ADVERTENCIA] La carpeta no existe: {folder}")
        
# Mostrar rutas para depuración
print(f"[RUTA] ROOT:         {PROJECT_ROOT}")
print(f"[RUTA] DATA:         {DATA_DIR}")
print(f"[RUTA] AUDIO:        {AUDIO}")
print(f"[RUTA] META:         {META}")
print(f"[RUTA] SPECTROGRAMS: {SPEC_DIR}")
print(f"[RUTA] FEATURES:     {FEATURES_DIR}")
'''


#=== PIPELINE COMPLETO CON FMA_MEDIUM ===#

# === 0. Descargar y extraer FMA Medium ===
#download_fma_medium(RAW)

#extract_fma_medium_and_metadata(RAW)

explore_fma_metadata(META, AUDIO)

# === 1. Filtrar inglés ===
#english_ids = filter_english_tracks(META)

# === 2. Audio features ===
#df_feats = extract_audio_features(english_ids, AUDIO)

# === 3. Espectrogramas ===
#generate_spectrograms(english_ids, AUDIO, SPEC_DIR)

# === 4. Valence & arousal ===
#df_va = add_valence_arousal(df_feats, META)

# === 5. Merge to df_feats ===
#df_merged = merge_valence_arousal(df_feats, META)

# === 6. PREDICT EMOTIONS ===

# df_va = add_emotion_labels(df_merged, FEATURES_DIR)

# === 7. Descargar letras de Genius ===

# df_lyrics = fetch_lyrics(english_ids, META, LYRICS_PATH)

# === 7. Unir letras ===
#df_final = merge_lyrics(df_va, PROCESSED)
