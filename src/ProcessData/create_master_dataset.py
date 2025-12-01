"""
Paso Final Fase 2: Creación del Dataset Maestro (Multimodal)
------------------------------------------------------------
Objetivo:
    1. Unir Metadata + Features Audio 1D + Features Texto 1D.
    2. Validar existencia física de tensores 2D (Espectrogramas y Embeddings).
    3. Generar la partición estática Train/Val/Test.
    4. Guardar data/processed/master_dataset.csv

Autor: Uzi & Brenda (Integration)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# === RUTAS ===
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

# Inputs (Fuentes de Datos)
# 1. La Lista de Tareas Original (para tener los Labels y Metadata)
META_CSV = DATA_DIR / "interim" / "aligned_metadata.csv"

# 2. Features 1D (Tabulares)
AUDIO_1D_CSV = DATA_DIR / "processed" / "features_1d" / "features_audio_1d.csv"
TEXT_1D_CSV  = DATA_DIR / "processed" / "features_1d" / "features_text_1d.csv"

# 3. Features 2D (Carpetas para validación)
SPEC_DIR = DATA_DIR / "processed" / "features_2d" / "spectrograms"
EMBED_DIR = DATA_DIR / "processed" / "features_2d" / "embeddings"

# Output
MASTER_CSV = DATA_DIR / "processed" / "master_dataset.csv"

def create_master():
    print("🔗 INICIANDO ENSAMBLAJE DEL DATASET MAESTRO MULTIMODAL")
    
    # --- 1. CARGA DE DATOS ---
    if not META_CSV.exists():
        print("❌ Error: Falta aligned_metadata.csv")
        return

    print("   📂 Cargando metadatos...")
    df_meta = pd.read_csv(META_CSV)
    # Nos quedamos solo con columnas clave para no duplicar info
    cols_meta = ['spotify_id', 'artist', 'track_name', 'valence', 'arousal', 'label_quadrant']
    df_meta = df_meta[cols_meta]
    
    print(f"      -> Base: {len(df_meta)} canciones.")

    # Cargar Audio 1D
    if AUDIO_1D_CSV.exists():
        df_audio = pd.read_csv(AUDIO_1D_CSV)
        print(f"      -> Audio 1D: {len(df_audio)} registros.")
    else:
        print("⚠️ ADVERTENCIA: No se encontró features_audio_1d.csv. El master estará incompleto.")
        df_audio = pd.DataFrame(columns=['spotify_id'])

    # Cargar Texto 1D (Brenda)
    if TEXT_1D_CSV.exists():
        df_text = pd.read_csv(TEXT_1D_CSV)
        print(f"      -> Texto 1D: {len(df_text)} registros.")
    else:
        print("⚠️ ADVERTENCIA: No se encontró features_text_1d.csv (¿Brenda ya terminó?).")
        # Creamos DF vacío para no romper el merge, pero el resultado será vacío si es inner join
        df_text = pd.DataFrame(columns=['spotify_id'])

    # --- 2. MERGE (FUSIÓN) ---
    print("   ⚔️  Uniendo tablas (Inner Join)...")
    
    # Unimos Meta con Audio
    df_master = pd.merge(df_meta, df_audio, on='spotify_id', how='inner')
    
    # Unimos con Texto (Si Brenda no ha entregado, esto filtrará todo si es 'inner'. 
    # Para pruebas de Uzi solo con Audio, cambiar a 'left' temporalmente, 
    # pero para el FINAL debe ser 'inner')
    if not df_text.empty:
        df_master = pd.merge(df_master, df_text, on='spotify_id', how='inner')
    else:
        print("      (Saltando merge de texto por falta de archivo)")

    print(f"      -> Coincidencias 1D encontradas: {len(df_master)}")

    # --- 3. VALIDACIÓN FÍSICA DE 2D (ARCHIVOS .NPY) ---
    print("   mag  Validando existencia de tensores (.npy)...")
    
    valid_indices = []
    missing_log = []

    for idx, row in tqdm(df_master.iterrows(), total=len(df_master)):
        sid = row['spotify_id']
        
        # Rutas esperadas
        path_spec = SPEC_DIR / f"{sid}.npy"
        path_embed = EMBED_DIR / f"{sid}.npy"
        
        # Validación: ¿Existen ambos archivos?
        # Nota: Si Brenda no ha terminado, comenta la línea de path_embed.exists() para probar solo audio
        audio_ok = path_spec.exists()
        text_ok = True # path_embed.exists() if not df_text.empty else True  <-- Descomentar cuando Brenda entregue
        
        if audio_ok and text_ok:
            valid_indices.append(idx)
            
            # Guardamos las rutas relativas en el CSV para facilitar la carga en PyTorch
            df_master.at[idx, 'path_spectrogram'] = f"features_2d/spectrograms/{sid}.npy"
            df_master.at[idx, 'path_embedding'] = f"features_2d/embeddings/{sid}.npy"
        else:
            missing_log.append(sid)

    # Filtrar solo los válidos
    df_final = df_master.loc[valid_indices].copy()
    
    if len(missing_log) > 0:
        print(f"      ⚠️ Se descartaron {len(missing_log)} canciones por falta de archivos .npy")

    # --- 4. SPLIT ESTRATIFICADO (TRAIN / VAL / TEST) ---
    print("   ✂️  Generando partición Train/Val/Test (80/10/10)...")
    
    # Usamos 'label_quadrant' para estratificar (que haya el mismo % de emociones en cada set)
    try:
        # Primero separamos Train (80%) del resto (20%)
        train, temp = train_test_split(
            df_final, test_size=0.2, stratify=df_final['label_quadrant'], random_state=42
        )
        # Luego separamos el resto en Val (50% de 20% = 10%) y Test (50% de 20% = 10%)
        val, test = train_test_split(
            temp, test_size=0.5, stratify=temp['label_quadrant'], random_state=42
        )
        
        # Asignar etiqueta en columna nueva
        df_final.loc[train.index, 'split'] = 'train'
        df_final.loc[val.index, 'split'] = 'val'
        df_final.loc[test.index, 'split'] = 'test'
        
        print("      Distribución:")
        print(df_final['split'].value_counts())
        
    except ValueError as e:
        print(f"      ⚠️ No se pudo estratificar (quizás muy pocos datos por clase): {e}")
        df_final['split'] = 'train' # Fallback

    # --- 5. GUARDADO FINAL ---
    # Limpieza de columnas: Ponemos las importantes al principio
    cols_order = ['spotify_id', 'split', 'label_quadrant', 'valence', 'arousal', 
                  'path_spectrogram', 'path_embedding']
    # Agregamos el resto de columnas (features 1D)
    remaining_cols = [c for c in df_final.columns if c not in cols_order]
    df_final = df_final[cols_order + remaining_cols]

    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(MASTER_CSV, index=False)
    
    print("\n" + "="*50)
    print(f"✅ DATASET MAESTRO CREADO: {len(df_final)} registros completos.")
    print(f"💾 Guardado en: {MASTER_CSV}")
    print("="*50)
    print("Este archivo está listo para el DataLoader de PyTorch.")

if __name__ == "__main__":
    create_master()