# src/ProcessData/text/cleaning.py

import re
from pathlib import Path
import pandas as pd
import nltk
from nltk.corpus import stopwords
import ftfy  # pip install ftfy

# Descargar stopwords una sola vez
nltk.download("stopwords", quiet=True)

# 1. Definir la lista personalizada de ruido musical (Interjecciones / Coros)
MUSIC_NOISE = {
    'la', 'ooh', 'ah', 'doo', 'hey', 'eh', 'woah', 'whoa', 'ha', 'mmm',
    'dum', 'pam', 'da', 'uh', 'ba', 'ya', 'pa', 'yo', 'bam', 'boom', 'um', 
    'tit', 'sir', 'shake', 'bum', 'hmm', 'huh', 'cha', 'lala', 'mm', 'nah', 'wow',
    # Agregamos variantes comunes por si acaso
    'oh', 'yeah', 'na', 'baby', 'oops','que', 'tu', 'lo','mi'
}

# 2. Crear el set maestro de Stopwords (Inglés estándar + Ruido Musical)
STOPWORDS = set(stopwords.words("english")).union(MUSIC_NOISE)


# ----------------------------------------------------
# Funciones internas de limpieza
# ----------------------------------------------------

def fix_encoding(text: str) -> str:
    """Arregla errores comunes de codificación (mojibake)."""
    return ftfy.fix_text(text)

def remove_section_tags(text: str) -> str:
    """Elimina etiquetas tipo [Chorus], [Verse 1], etc."""
    return re.sub(r"\[.*?\]", "", text)

def clean_text_bert(text: str) -> str:
    """
    Limpieza para BERT:
    - Mantiene mayúsculas.
    - Mantiene puntuación básica.
    - Mantiene saltos de línea (\n).
    - NO elimina stopwords (BERT necesita contexto).
    """
    if not isinstance(text, str): return ""
    
    text = fix_encoding(text)
    text = remove_section_tags(text)
    
    # 1. Normalizar espacios horizontales pero protegiendo los \n
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 2. Reducir saltos de línea múltiples a máximo 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def clean_text_tfidf(text: str) -> str:
    """
    Limpieza para TF-IDF:
    - Minúsculas.
    - Sin puntuación.
    - Sin saltos de línea (texto plano).
    - SIN stopwords (incluyendo ruido musical).
    """
    if not isinstance(text, str): return ""

    text = fix_encoding(text)
    text = remove_section_tags(text)
    
    # 1. Minúsculas
    text = text.lower()
    
    # 2. Reemplazar saltos de línea por espacio (aplanar)
    text = text.replace('\n', ' ')
    
    # 3. Eliminar todo lo que no sea letra o número
    text = re.sub(r"[^a-z0-9\s]", "", text)
    
    # 4. Eliminar espacios extra resultantes
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Filtrar Stopwords (Aquí se van 'the', 'is' y también 'ooh', 'yeah', 'la')
    tokens = [t for t in text.split() if t not in STOPWORDS]
    
    return " ".join(tokens)


def process_lyrics(text: str, method: str):
    """Factory function para seleccionar limpieza."""
    if method == "bert":
        return clean_text_bert(text)
    elif method == "tfidf":
        return clean_text_tfidf(text)
    return text


# ----------------------------------------------------
# Función principal
# ----------------------------------------------------

def save_clean_lyrics(aligned_csv_path: Path, output_path: Path, format='parquet'):
    """
    Genera archivo con IDs y letras limpias.
    Args:
        format: 'parquet' (recomendado) o 'json'
    """
    print(f"Leyendo datos de: {aligned_csv_path}")
    df = pd.read_csv(aligned_csv_path)

    # Validar que exista el ID
    if 'spotify_id' not in df.columns:
        id_col = 'musicId' if 'musicId' in df.columns else df.columns[0]
    else:
        id_col = 'spotify_id'

    print("Procesando limpieza TF-IDF (Quitando ruido musical)...")
    df["clean_lyrics_tfidf"] = df["lyrics"].apply(lambda x: process_lyrics(x, "tfidf"))

    print("Procesando limpieza BERT (Preservando estructura)...")
    df["clean_lyrics_bert"] = df["lyrics"].apply(lambda x: process_lyrics(x, "bert"))

    # Seleccionar SOLO columnas necesarias
    cols_to_keep = [id_col, "clean_lyrics_tfidf", "clean_lyrics_bert"]
    df_clean = df[cols_to_keep]

    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        df_clean.to_parquet(output_path, index=False)
    elif format == 'json':
        df_clean.to_json(output_path, orient='records', lines=True)
    else:
        df_clean.to_csv(output_path, index=False)

    print(f"Archivo limpio generado ({format}):\n{output_path}")


# ----------------------------------------------------
# Ejecución independiente
# ----------------------------------------------------

if __name__ == "__main__":
    # Ajuste de rutas para prueba
    ROOT = Path(__file__).resolve().parents[3]
    ALIGNED = ROOT / "data" / "interim" / "aligned_metadata.csv"
    
    # Cambiamos extensión a .parquet
    OUT_FILE = ROOT / "data" / "interim" / "lyrics_cleaned.parquet"

    print("Iniciando limpieza de texto...")
    save_clean_lyrics(ALIGNED, OUT_FILE, format='parquet')