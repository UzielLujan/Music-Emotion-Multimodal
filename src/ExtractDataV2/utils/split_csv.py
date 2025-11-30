"""
Utilidad: Split CSV
-------------------
Divide el dataset maestro de letras en dos partes iguales (Uzi y Brenda)
para paralelizar la descarga de audios sin conflictos de IP.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Configuración de rutas
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2] # Subir hasta la raíz del proyecto
DATA_DIR = PROJECT_ROOT / "data" / "raw_v2"
INPUT_CSV = DATA_DIR / "metadata_step2_lyrics_clean.csv"

def split_workload():
    print(f"🔪 Iniciando división de trabajo...")
    
    if not INPUT_CSV.exists():
        print(f"❌ Error: No existe el archivo {INPUT_CSV}")
        return

    # Cargar el dataset limpio
    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    print(f"   📊 Total de canciones: {total}")

    # Calcular el punto medio
    mid_point = total // 2
    
    # Dividir
    df_uzi = df.iloc[:mid_point]
    df_brenda = df.iloc[mid_point:]

    # Rutas de salida
    path_uzi = DATA_DIR / "metadata_part_uzi.csv"
    path_brenda = DATA_DIR / "metadata_part_brenda.csv"

    # Guardar
    df_uzi.to_csv(path_uzi, index=False)
    df_brenda.to_csv(path_brenda, index=False)

    print("\n✅ DIVISIÓN COMPLETADA")
    print(f"   👤 Parte Uzi:    {len(df_uzi)} canciones -> {path_uzi.name}")
    print(f"   👤 Parte Brenda: {len(df_brenda)} canciones -> {path_brenda.name}")
    print("\n   ⚠️ IMPORTANTE: Ahora cada uno debe actualizar 'CSV_STEP2' en su main.py")

if __name__ == "__main__":
    split_workload()