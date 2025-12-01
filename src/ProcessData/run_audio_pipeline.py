"""
Pipeline de Procesamiento de Audio (Fase 2)
-------------------------------------------
Orquesta la transformación: MP3 -> LLDs/HSFs + Espectrogramas.
Autor: Uzi (Audio Architect)
"""

import sys
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from pathlib import Path

# Ajuste de paths para importar módulos locales
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Importar tus módulos
from src.ProcessData.audio.trimming import smart_trim
from src.ProcessData.audio.features_1d import extract_hsfs
from src.ProcessData.audio.spectrograms import extract_melspectrogram

# Rutas
DATA_DIR = PROJECT_ROOT / "data"
INPUT_CSV = DATA_DIR / "interim" / "aligned_metadata.csv"
AUDIO_DIR = DATA_DIR / "raw_v2" / "audio"

# Outputs
OUT_1D_DIR = DATA_DIR / "processed" / "features_1d"
OUT_2D_DIR = DATA_DIR / "processed" / "features_2d" / "spectrograms"

OUT_CSV_HSFS = OUT_1D_DIR / "features_audio_1d.csv"

def run_pipeline():
    print("🎛️  INICIANDO PIPELINE DE AUDIO...")
    
    # 1. Preparar Directorios
    OUT_1D_DIR.mkdir(parents=True, exist_ok=True)
    OUT_2D_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Cargar Lista de Tareas
    if not INPUT_CSV.exists():
        print(f"❌ Error: No existe {INPUT_CSV}. Ejecuta alignment.py primero.")
        return
    
    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    print(f"   🎯 Objetivo: Procesar {total} canciones.")
    
    # Lista para acumular los HSFs (filas del futuro CSV)
    hsf_data = []
    
    # 3. Iterar (con barra de progreso)
    for i, row in tqdm(df.iterrows(), total=total, desc="Procesando Audio"):
        spotify_id = row['spotify_id']
        audio_path = AUDIO_DIR / f"{spotify_id}.mp3"
        
        # Output esperado del espectrograma
        spec_out_path = OUT_2D_DIR / f"{spotify_id}.npy"
        
        # Checkpoint: Si ya existe el npy, asumimos que este ID está listo
        if spec_out_path.exists():
            continue

        try:
            # A. Cargar Audio
            y, sr = librosa.load(audio_path, sr=22050)
            
            # B. Recorte Inteligente (15s con más energía)
            y_trimmed = smart_trim(y, sr, duration=15)
            
            # C. Rama 1D: HSFs
            features = extract_hsfs(y_trimmed, sr)
            features['spotify_id'] = spotify_id # Llave primaria
            hsf_data.append(features)
            
            # D. Rama 2D: Espectrograma
            melspec = extract_melspectrogram(y_trimmed, sr)
            np.save(spec_out_path, melspec) # Guardar binario .npy
            
        except Exception as e:
            print(f"\n❌ Error en {spotify_id}: {e}")
            continue

    # 4. Guardar CSV de Features 1D
    if hsf_data:
        print(f"\n💾 Guardando tabla de HSFs...")
        df_hsf = pd.DataFrame(hsf_data)
        
        # Reordenar: poner spotify_id primero
        cols = ['spotify_id'] + [c for c in df_hsf.columns if c != 'spotify_id']
        df_hsf = df_hsf[cols]
        
        # Guardado incremental (append) o overwrite si es nuevo
        if OUT_CSV_HSFS.exists():
             df_hsf.to_csv(OUT_CSV_HSFS, mode='a', header=False, index=False)
        else:
             df_hsf.to_csv(OUT_CSV_HSFS, index=False)
             
    print("\n✨ PIPELINE DE AUDIO FINALIZADO.")

if __name__ == "__main__":
    run_pipeline()