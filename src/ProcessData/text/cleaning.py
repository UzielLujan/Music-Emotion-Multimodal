# src/ProcessData/text/cleaning.py

import re
from pathlib import Path
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Descargar stopwords una sola vez
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))


# ----------------------------------------------------
# Funciones internas de limpieza
# ----------------------------------------------------

def remove_section_tags(text: str):
    """Elimina etiquetas tipo [Chorus], [Verse 1], etc."""
    return re.sub(r"\[.*?\]", "", text)


def clean_text_basic(text: str, remove_stopwords=False):
    """
    Limpieza ligera para letras.
    Mantiene saltos de línea para modelado semántico.
    """
    if not isinstance(text, str):
        text = str(text)

    text = remove_section_tags(text)

    # Arreglo básico de codificación
    text = text.encode("utf-8", "ignore").decode()

    # Minúsculas
    text = text.lower()

    # Mantener saltos de línea, quitar caracteres raros
    text = re.sub(r"[^a-z0-9\n\s]", " ", text)

    # Reducir espacios
    text = re.sub(r"\s+", " ", text).strip()

    # Stopwords (solo TF-IDF)
    if remove_stopwords:
        tokens = [t for t in text.split() if t not in STOPWORDS]
        text = " ".join(tokens)

    return text


def process_lyrics(text: str, for_embeddings=True):
    """
    - Para TF-IDF: for_embeddings=False → quitar stopwords
    - Para BERT:    for_embeddings=True  → limpieza suave
    """
    if for_embeddings:
        return clean_text_basic(text, remove_stopwords=False)
    else:
        return clean_text_basic(text, remove_stopwords=True)


# ----------------------------------------------------
# Función principal: genera un solo archivo limpio
# ----------------------------------------------------

def save_clean_lyrics_single(aligned_csv_path: Path, output_csv_path: Path):
    """
    Genera un único archivo limpio que conserva todas las columnas originales
    y agrega:
        - clean_lyrics_tfidf
        - clean_lyrics_bert
    """
    df = pd.read_csv(aligned_csv_path)

    # Limpieza para TF-IDF
    df["clean_lyrics_tfidf"] = df["lyrics"].apply(
        lambda x: process_lyrics(x, for_embeddings=False)
    )

    # Limpieza para BERT
    df["clean_lyrics_bert"] = df["lyrics"].apply(
        lambda x: process_lyrics(x, for_embeddings=True)
    )

    # Guardar archivo único
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print(f"Archivo limpio generado:\n{output_csv_path}")


# ----------------------------------------------------
# Ejecución independiente
# ----------------------------------------------------

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[3]
    ALIGNED = ROOT / "data" / "interim" / "aligned_metadata.csv"
    OUTCSV = ROOT / "data" / "interim" / "aligned_metadata_cleanlyrics.csv"

    print("Generando archivo único de letras limpias...")
    save_clean_lyrics_single(ALIGNED, OUTCSV)
