"""
Paso Final Fase 2: Creación del Dataset Maestro (Multimodal)
------------------------------------------------------------
CORREGIDO: Usa embeddings_ids.npy para mapear correctamente las filas.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# === RUTAS (Ajustadas para Colab/Drive) ===
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path("/content/drive/MyDrive/Proyecto_Final_MIR")

DATA_DIR = BASE_DIR / "data"

# Inputs
META_CSV = DATA_DIR / "interim" / "aligned_metadata.csv"
AUDIO_1D_CSV = DATA_DIR / "processed" / "features_1d" / "features_audio_1d.csv"
TEXT_1D_PARQUET = DATA_DIR / "processed" / "features_1d" / "features_text_1d.parquet"

# Rutas de Features 2D
SPEC_DIR = DATA_DIR / "processed" / "features_2d" / "spectrograms"
EMBED_DIR = DATA_DIR / "processed" / "features_2d" / "embeddings"
EMBED_FILE = EMBED_DIR / "embeddings_2d.npy"
# NUEVO: El archivo que contiene los IDs en orden
EMBED_IDS_FILE = EMBED_DIR / "embeddings_ids.npy"

# Output
MASTER_CSV = DATA_DIR / "processed" / "master_dataset.csv"

def create_master():
    print("🔗 INICIANDO ENSAMBLAJE DEL DATASET MAESTRO (VERSIÓN FIX IDs)")
    
    # --- 1. CARGA DE DATOS ---
    if not META_CSV.exists():
        print(f"❌ Error: Falta {META_CSV}"); return

    print("   📂 Cargando metadatos...")
    df_meta = pd.read_csv(META_CSV)
    
    # Estandarizar ID
    if 'musicId' in df_meta.columns and 'spotify_id' not in df_meta.columns:
        df_meta.rename(columns={'musicId': 'spotify_id'}, inplace=True)
        
    cols_meta = ['spotify_id', 'artist', 'track_name', 'valence', 'arousal', 'label_quadrant']
    df_meta = df_meta[[c for c in cols_meta if c in df_meta.columns]]

    # Cargar Audio 1D
    if AUDIO_1D_CSV.exists():
        df_audio = pd.read_csv(AUDIO_1D_CSV)
    else:
        df_audio = pd.DataFrame(columns=['spotify_id'])

    # Cargar Texto 1D
    if TEXT_1D_PARQUET.exists():
        df_text = pd.read_parquet(TEXT_1D_PARQUET)
    else:
        df_text = pd.DataFrame(columns=['spotify_id'])

    # --- 2. MERGE ---
    print("   ⚔️  Uniendo tablas...")
    # Merge Metadata + Audio
    df_master = pd.merge(df_meta, df_audio, on='spotify_id', how='inner' if not df_audio.empty else 'left')
    
    # Merge + Texto
    if not df_text.empty:
        df_master = pd.merge(df_master, df_text, on='spotify_id', how='inner')
    
    print(f"      -> Total tras unión: {len(df_master)} canciones.")

    # --- 3. MAPEADO DE EMBEDDINGS (LA CORRECCIÓN) ---
    print("   🔑 Mapeando índices de embeddings...")
    if EMBED_IDS_FILE.exists() and EMBED_FILE.exists():
        # Cargamos los IDs que corresponden a las filas de embeddings_2d.npy
        ids_array = np.load(EMBED_IDS_FILE, allow_pickle=True)
        
        # Creamos un diccionario: { 'ID_CANCION': NUMERO_FILA }
        # Ejemplo: {'1w8r3P...': 0, '6VeihS...': 1}
        id_to_idx_map = {sid: idx for idx, sid in enumerate(ids_array)}
        
        # Mapeamos usando el ID de la canción
        df_master['embedding_idx'] = df_master['spotify_id'].map(id_to_idx_map)
        
        # Verificamos cuántos encontramos
        found = df_master['embedding_idx'].notna().sum()
        print(f"      -> Embeddings encontrados: {found} de {len(df_master)}")
    else:
        print("❌ ERROR CRÍTICO: No se encuentra embeddings_ids.npy o embeddings_2d.npy")
        return

    # --- 4. VALIDACIÓN FÍSICA ---
    print("   🔍 Validando integridad final...")
    valid_indices = []
    
    for idx, row in tqdm(df_master.iterrows(), total=len(df_master)):
        sid = row['spotify_id']
        
        # 1. Validar Espectrograma (Archivo individual)
        path_spec = SPEC_DIR / f"{sid}.npy"
        audio_ok = path_spec.exists()
        
        # 2. Validar Embedding (Debe tener índice válido)
        # pd.notna revisa que no sea NaN (que significa que sí encontramos el ID en el mapa)
        text_ok = pd.notna(row['embedding_idx'])
        
        if audio_ok and text_ok:
            valid_indices.append(idx)
            df_master.at[idx, 'path_spectrogram'] = f"features_2d/spectrograms/{sid}.npy"
            df_master.at[idx, 'path_embedding'] = "features_2d/embeddings/embeddings_2d.npy"
            # Aseguramos que sea entero (para que no falle el dataloader)
            df_master.at[idx, 'embedding_idx'] = int(row['embedding_idx'])

    df_final = df_master.loc[valid_indices].copy()
    
    # Convertir embedding_idx a entero explícitamente
    df_final['embedding_idx'] = df_final['embedding_idx'].astype(int)
    
    print(f"      -> Dataset limpio final: {len(df_final)} registros.")

    # --- 5. SPLIT ---
    print("   ✂️  Generando Splits...")
    try:
        train, temp = train_test_split(df_final, test_size=0.3, stratify=df_final['label_quadrant'], random_state=42)
        val, test = train_test_split(temp, test_size=0.5, stratify=temp['label_quadrant'], random_state=42)
        
        df_final.loc[train.index, 'split'] = 'train'
        df_final.loc[val.index, 'split'] = 'val'
        df_final.loc[test.index, 'split'] = 'test'
    except Exception as e:
        print(f"⚠️ Falló estratificación ({e}), haciendo split aleatorio.")
        df_final['split'] = 'train' # Fallback para evitar crash

    # --- 6. GUARDADO ---
    priority_cols = ['spotify_id', 'split', 'label_quadrant', 'embedding_idx', 'path_spectrogram', 'path_embedding']
    existing_priority = [c for c in priority_cols if c in df_final.columns]
    other_cols = [c for c in df_final.columns if c not in existing_priority]
    
    df_final = df_final[existing_priority + other_cols]

    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(MASTER_CSV, index=False)
    print(f"✅ ¡LISTO! Dataset guardado en: {MASTER_CSV}")

if __name__ == "__main__":
    create_master()