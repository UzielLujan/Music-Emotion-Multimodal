"""
Paso 1: Selección de Semillas desde Dataset Kaggle (Local)
----------------------------------------------------------
Estrategia: Sampling Estratificado por Cuadrantes de Russell.
Input: data/raw_v2/archive/dataset.csv
Output: data/raw_v2/metadata_step1_spotify.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

def clean_artist_name(artist_str):
    """
    El dataset usa ';' para separar artistas (ej: 'Shakira;Rihanna').
    Nos quedamos solo con el primero para facilitar la búsqueda.
    """
    if pd.isna(artist_str):
        return "Unknown"
    return artist_str.split(';')[0]

def extract_spotify_metadata(kaggle_csv_path, output_csv, samples_per_quadrant=500):
    print(f" Cargando dataset maestro desde: {kaggle_csv_path}")
    
    if not kaggle_csv_path.exists():
        print(f"ERROR: No se encuentra el archivo {kaggle_csv_path}")
        return

    # Cargar solo columnas necesarias para ahorrar memoria
    cols = ['track_id', 'artists', 'track_name', 'valence', 'energy', 'instrumentalness', 'popularity']
    df = pd.read_csv(kaggle_csv_path, usecols=cols)
    
    print(f"   Total tracks crudos: {len(df)}")

    # --- FILTROS DE CALIDAD ---
    # 1. Eliminar instrumentales (ahorra tiempo en Genius)
    df = df[df['instrumentalness'] < 0.5]
    
    # 2. Filtrar por popularidad (eliminar ruido/canciones desconocidas)
    # Un umbral de 20 elimina rarezas que quizás no tengan letra en Genius
    df = df[df['popularity'] >= 20]
    
    # 3. Eliminar duplicados de IDs
    df = df.drop_duplicates(subset=['track_id'])
    
    print(f"     Tracks tras limpieza (Vocal + Popular): {len(df)}")

    # --- DEFINICIÓN DE CUADRANTES (RUSSELL) ---
    # Q1: Happy (High Val, High Eng)
    q1 = df[(df['valence'] >= 0.6) & (df['energy'] >= 0.6)].copy()
    q1['label_quadrant'] = 'Q1_Happy'

    # Q2: Angry/Intense (Low Val, High Eng)
    q2 = df[(df['valence'] <= 0.4) & (df['energy'] >= 0.6)].copy()
    q2['label_quadrant'] = 'Q2_Angry'

    # Q3: Sad (Low Val, Low Eng)
    q3 = df[(df['valence'] <= 0.4) & (df['energy'] <= 0.4)].copy()
    q3['label_quadrant'] = 'Q3_Sad'

    # Q4: Relaxed (High Val, Low Eng)
    q4 = df[(df['valence'] >= 0.6) & (df['energy'] <= 0.4)].copy()
    q4['label_quadrant'] = 'Q4_Relaxed'

    print(f"   Potencial Distribución :")
    print(f"      Q1 (Happy): {len(q1)}")
    print(f"      Q2 (Angry): {len(q2)}")
    print(f"      Q3 (Sad):   {len(q3)}")
    print(f"      Q4 (Relaxed): {len(q4)}")

    # --- SAMPLING (BALANCEO) ---
    semillas = []
    for q_df in [q1, q2, q3, q4]:
        # Si hay menos canciones que el sample pedido, tomamos todas
        n = min(len(q_df), samples_per_quadrant)
        sample = q_df.sample(n=n, random_state=42) # random_state para reproducibilidad
        semillas.append(sample)
    
    df_final = pd.concat(semillas)

    # --- FORMATEO FINAL ---
    # Renombrar columnas para que coincidan con el pipeline v2 original
    # Kaggle: 'track_id', 'energy' -> Pipeline: 'spotify_id', 'arousal'
    df_final.rename(columns={
        'track_id': 'spotify_id',
        'energy': 'arousal' # Usamos energy como proxy de arousal
    }, inplace=True)

    # Limpiar nombre del artista
    df_final['artist'] = df_final['artists'].apply(clean_artist_name)
    
    # Seleccionar columnas finales
    cols_final = ['spotify_id', 'artist', 'track_name', 'valence', 'arousal', 'label_quadrant', 'instrumentalness']
    df_final = df_final[cols_final]

    # Guardar
    df_final.to_csv(output_csv, index=False)
    print(f"   Paso 1 completado. Dataset semilla generado: {len(df_final)} canciones.")
    print(f"   Guardado en: {output_csv}")