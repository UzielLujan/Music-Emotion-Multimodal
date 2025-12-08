# src/ProcessData/text/features_1d.py

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def generate_text_1d_features(
        clean_file_path: Path,     
        output_path: Path,         
        max_vocab_size=3000,       
        format='parquet'):
    """
    Genera la representación vectorial base (TF-IDF) preparando el terreno
    para la selección Chi2 que se hará en el entrenamiento.
    """

    print(f"📚 Cargando texto limpio desde: {clean_file_path}")
    
    # Soporte para Parquet (Fase 2)
    if clean_file_path.suffix == '.parquet':
        df = pd.read_parquet(clean_file_path)
    elif clean_file_path.suffix == '.json':
        df = pd.read_json(clean_file_path, lines=True)
    else:
        df = pd.read_csv(clean_file_path)

    # Validar nombre de columna de texto
    # ESTANDARIZACIÓN: Siempre busca esta columna específica
    col_text = "clean_lyrics_tfidf"
    
    if col_text not in df.columns:
        # Fallback de seguridad por si acaso
        if 'lyrics' in df.columns:
            print(f"⚠️ Advertencia: No se halló '{col_text}', usando 'lyrics' como fallback.")
            col_text = 'lyrics'
        else:
            raise ValueError(f"Falta la columna '{col_text}' en el archivo de entrada.")

    # Asegurar que sean strings y llenar nulos
    corpus = df[col_text].fillna("").astype(str)

    # ---------------------------------------------------------
    # 1. Vectorización TF-IDF (Pre-requisito para Chi2)
    # ---------------------------------------------------------
    print(f"🧮 Generando Vectores TF-IDF (Vocabulario base: {max_vocab_size})...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_vocab_size,
        min_df=5,
        max_df=0.95
    )
    
    X_vec = vectorizer.fit_transform(corpus)
    
    print(f"   ℹ️ Vocabulario generado: {len(vectorizer.get_feature_names_out())} palabras")

    # ------------------------
    # 2. Construcción del DataFrame
    # ------------------------
    feature_names = vectorizer.get_feature_names_out()
    
    # Usamos float32 para ahorrar memoria
    df_out = pd.DataFrame(X_vec.toarray().astype(np.float32), 
                          columns=[f"tfidf_{name}" for name in feature_names])

    # Insertar ID para cruces futuros
    if 'spotify_id' in df.columns:
        df_out.insert(0, "spotify_id", df["spotify_id"])
    elif 'musicId' in df.columns:
        df_out.insert(0, "musicId", df["musicId"])

    # ------------------------
    # 3. Guardar
    # ------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        df_out.to_parquet(output_path, index=False)
    else:
        df_out.to_csv(output_path, index=False)

    print(f"✅ Features base generadas ({format}): {output_path}")
    print(f"   Shape: {df_out.shape}")
    print("   ⚠️ RECUERDA: El filtro Chi-cuadrado se aplicará en el entrenamiento.")

if __name__ == "__main__":
    pass