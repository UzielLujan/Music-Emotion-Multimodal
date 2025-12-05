"""
Paso 1: La Gran Alineación (Alignment)
--------------------------------------
Objetivo: Generar la "Lista de Tareas" (aligned_metadata.csv).
Lógica:
    1. Cargar metadata limpia (Lyrics Clean).
    2. Escanear carpeta de audios (físicos).
    3. Intersection: Mantener solo (CSV ∩ Audio).
    4. Guardar en data/interim/aligned_metadata.csv.
"""

import os
import pandas as pd
from pathlib import Path

# === CONFIGURACIÓN DE RUTAS ===
# Detectar raíz del proyecto
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"

# Inputs
RAW_AUDIO_DIR = DATA_DIR / "raw_v2" / "audio"
INPUT_CSV = DATA_DIR / "raw_v2" / "metadata_step2_lyrics_clean.csv"

# Output
INTERIM_DIR = DATA_DIR / "interim"
OUTPUT_CSV = INTERIM_DIR / "aligned_metadata.csv"

def align_dataset():
    print(f" Iniciando Alineación de Datos...")
    
    # 1. Verificar inputs
    if not INPUT_CSV.exists():
        print(f"❌ Error: No existe el CSV de entrada: {INPUT_CSV}")
        return
    
    if not RAW_AUDIO_DIR.exists():
        print(f"❌ Error: No existe la carpeta de audios: {RAW_AUDIO_DIR}")
        return

    # 2. Cargar CSV
    print(f"    Cargando CSV de letras...")
    df = pd.read_csv(INPUT_CSV)
    total_csv = len(df)
    print(f"      -> {total_csv} registros en CSV.")

    # 3. Escanear Audios Reales
    print(f"    Escaneando archivos MP3 en disco...")
    # Obtenemos todos los archivos .mp3 y quitamos la extensión para tener el ID puro
    # Usamos set() para búsqueda O(1) ultra rápida
    audio_files = {f.stem for f in RAW_AUDIO_DIR.glob("*.mp3") if f.stat().st_size > 0} 
    # (Nota: st_size > 0 filtra archivos corruptos vacíos de 0kb)
    
    total_audio = len(audio_files)
    print(f"      -> {total_audio} archivos MP3 válidos encontrados.")

    # 4. CRUCE (Intersection)
    print(f"    Realizando cruce (Match por spotify_id)...")
    
    # Filtramos el DataFrame: mantenemos solo si el spotify_id está en el set de audios
    df_aligned = df[df['spotify_id'].isin(audio_files)].copy()
    
    # Calcular pérdidas
    lost = total_csv - len(df_aligned)
    
    # 5. Guardar Resultado
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df_aligned.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "="*40)
    print(" ALINEACIÓN COMPLETADA")
    print("="*40)
    print(f"   Originales en CSV: {total_csv}")
    print(f"   Audios disponibles: {total_audio}")
    print(f"   --------------------------------")
    print(f"    Perdidos (Sin audio): {lost}")
    print(f"    DATASET FINAL ALINEADO: {len(df_aligned)}")
    print(f"    Guardado en: {OUTPUT_CSV}")
    print("="*40)
    print("   Este archivo 'aligned_metadata.csv' es la LISTA DE TAREAS para la Fase 2.")

if __name__ == "__main__":
    align_dataset()