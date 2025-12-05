# src/ProcessData/text/features_1d.py

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD


def generate_text_1d_features(
        clean_tfidf_csv_path: Path,
        output_csv_path: Path,
        tfidf_max_features=5000,
        chi2_dims=500,
        lsa_dims=100):
    """
    Genera la representación 1D del texto con los pasos:
    1. TF-IDF (hasta tfidf_max_features)
    2. Chi² supervisado → chi2_dims
    3. LSA (TruncatedSVD) → lsa_dims
    """

    # ------------------------
    # 1. Cargar CSV limpio
    # ------------------------
    df = pd.read_csv(clean_tfidf_csv_path)

    if "clean_lyrics_tfidf" not in df.columns:
        raise ValueError("Se esperaba la columna 'clean_lyrics_tfidf'.")

    corpus = df["clean_lyrics_tfidf"].astype(str)

    # ------------------------
    # 2. TF-IDF base
    # ------------------------
    tfidf = TfidfVectorizer(max_features=tfidf_max_features)
    X_tfidf = tfidf.fit_transform(corpus)

    # ------------------------
    # 3. Selección Chi²
    # ------------------------
    if "label_quadrant" in df.columns:
        target_col = "label_quadrant"
    elif "quadrant" in df.columns:
        target_col = "quadrant"
    else:
        raise ValueError("No existe columna de etiqueta ('label_quadrant' o 'quadrant').")

    selector = SelectKBest(chi2, k=chi2_dims)
    X_chi2 = selector.fit_transform(X_tfidf, df[target_col])

    # ------------------------
    # 4. LSA / TruncatedSVD a lsa_dims
    # ------------------------
    lsa = TruncatedSVD(n_components=lsa_dims, random_state=42)
    X_lsa = lsa.fit_transform(X_chi2)

    var_total = lsa.explained_variance_ratio_.sum()
    print(f"Varianza explicada por {lsa_dims} componentes LSA: {var_total:.4f}")

    # ------------------------
    # 5. Construcción del DataFrame final
    # ------------------------
    feature_cols = [f"f_{i}" for i in range(lsa_dims)]
    df_out = pd.DataFrame(X_lsa, columns=feature_cols)

    df_out.insert(0, "spotify_id", df["spotify_id"])

    # ------------------------
    # 6. Guardar archivo
    # ------------------------
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv_path, index=False)

    print(f"Archivo 1D generado en: {output_csv_path}")
    print(f"Dimensiones finales: {df_out.shape}")
