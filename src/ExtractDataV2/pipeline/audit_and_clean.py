import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Configuración
BASE_DIR = Path(__file__).resolve().parents[3] # Raíz del proyecto
INPUT_CSV = BASE_DIR / "data" / "raw_v2" / "metadata_step2_lyrics.csv"
OUTPUT_CSV = BASE_DIR / "data" / "raw_v2" / "metadata_step2_lyrics_clean.csv"

def analyze_and_clean():
    if not INPUT_CSV.exists():
        print("❌ No se encontró el archivo de letras.")
        return

    print(f"📂 Cargando: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"📊 Total inicial: {len(df)} filas")
    print("\n" + "="*40)
    print("🔍 ANÁLISIS Y LIMPIEZA DE DATASET DE LETRAS")
    print("="*40)
    print("\nResumen general:")
    print(df.info())
    print("\nEstadísticas:")
    print(df.describe(include='all'))
    print("\nConteo de valores nulos por columna:")
    print(df.isnull().sum())
    
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    if empty_cols:
        print(f"\n⚠️ Columnas completamente vacías: {empty_cols}")
    
    print("\nEjemplo de filas con valores nulos:")
    print(df[df.isnull().any(axis=1)].head())
    
    print("\nDuplicados por 'spotify_id':", df.duplicated(subset=['spotify_id']).sum())

    # --- 1. LIMPIEZA DE BASURA ---
    print("\n🧹 --- LIMPIEZA ---")
    
    # Eliminar filas con IDs o Letras vacías
    initial_len = len(df)
    df = df.dropna(subset=['spotify_id', 'lyrics', 'valence', 'arousal'])
    nan_dropped = initial_len - len(df)
    print(f"   - Filas con nulos eliminadas: {nan_dropped}")

    # Eliminar letras demasiado cortas (menos de 50 caracteres suele ser error/instrumental)
    initial_len = len(df)
    df = df[df['lyrics'].str.len() > 100]
    short_dropped = initial_len - len(df)
    print(f"   - Letras basura (<100 chars) eliminadas: {short_dropped}")

    print(f"✅ Total tras limpieza básica: {len(df)}")

    # --- 2. ANÁLISIS DE CLASES ---
    print("\n⚖️ --- DISTRIBUCIÓN DE CLASES ---")
    # Asumiendo que tenemos la columna 'label_quadrant' del Paso 1
    # Si se perdió en algún merge, la recalculamos rápido:
    if 'label_quadrant' not in df.columns:
        conditions = [
            (df['valence'] >= 0.6) & (df['arousal'] >= 0.6), # Q1
            (df['valence'] <= 0.4) & (df['arousal'] >= 0.6), # Q2
            (df['valence'] <= 0.4) & (df['arousal'] <= 0.4), # Q3
            (df['valence'] >= 0.6) & (df['arousal'] <= 0.4)  # Q4
        ]
        choices = ['Q1_Happy', 'Q2_Angry', 'Q3_Sad', 'Q4_Relaxed']
        df['label_quadrant'] = np.select(conditions, choices, default='Neutral/Borderline')

    counts = df['label_quadrant'].value_counts()
    print(counts)
    
    # --- 3. BALANCEO INTELIGENTE ---
    print("\n✂️ --- RECORTANDO DATASET ---")
    
    # Definir objetivo por clase (para llegar a aprox 6000 totales)
    # Confirmamos que Q4 tiene ~500, intentemos que las otras no pasen de 2000 para no desbalancear tanto.
    # Total aprox: 2000 * 3 + 500 = 6500.
    TARGET_PER_CLASS = 2000 
    
    dfs_balanced = []
    for label in counts.index:
        df_class = df[df['label_quadrant'] == label]
        count = len(df_class)
        
        if count > TARGET_PER_CLASS:
            # Si sobran, tomamos una muestra aleatoria
            print(f"   📉 Recortando {label}: De {count} a {TARGET_PER_CLASS}")
            df_sampled = df_class.sample(n=TARGET_PER_CLASS, random_state=42)
            dfs_balanced.append(df_sampled)
        else:
            # Si faltan (ej. Q4), tomamos todo lo que hay
            print(f"   ✅ Manteniendo {label}: {count} (Son minoría, se quedan todos)")
            dfs_balanced.append(df_class)
            
    df_final = pd.concat(dfs_balanced)
    
    # --- 4. GUARDADO ---
    # Seleccionar solo columnas útiles para limpiar el CSV final
    cols_to_keep = ['spotify_id', 'artist', 'track_name', 'valence', 'arousal', 'label_quadrant', 'lyrics']
    # Filtrar solo si existen esas columnas
    cols_final = [c for c in cols_to_keep if c in df_final.columns]
    df_final = df_final[cols_final]

    df_final.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*40)
    print(f"🚀 DATASET OPTIMIZADO LISTO: {len(df_final)} canciones")
    print(f"💾 Guardado en: {OUTPUT_CSV}")
    print("="*40)
    print("AHORA: Actualiza tu main.py para leer este archivo en el Paso 3.")

if __name__ == "__main__":
    analyze_and_clean()