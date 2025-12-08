import re
from pathlib import Path
import pandas as pd
import nltk
from nltk.corpus import stopwords
import ftfy 

nltk.download("stopwords", quiet=True)

# --- CONFIGURACIÓN ---
MUSIC_NOISE = {
    'la', 'ooh', 'ah', 'doo', 'hey', 'eh', 'woah', 'whoa', 'ha', 'mmm',
    'dum', 'pam', 'da', 'uh', 'ba', 'ya', 'pa', 'yo', 'bam', 'boom', 'um', 
    'tit', 'sir', 'shake', 'bum', 'hmm', 'huh', 'cha', 'lala', 'mm', 'nah', 'wow',
    'oh', 'yeah', 'na', 'baby', 'oops','que', 'tu', 'lo','mi'
}
STOPWORDS = set(stopwords.words("english")).union(MUSIC_NOISE)

# --- FUNCIONES DE TEXTO ---
def fix_encoding(text: str) -> str:
    return ftfy.fix_text(text)

def remove_section_tags(text: str) -> str:
    return re.sub(r"\[.*?\]", "", text)

def clean_text_bert(text: str) -> str:
    # Mantiene estructura, puntuación y mayúsculas.
    # ELIMINA: Asiáticos, Rusos, Emojis (Lista Blanca Estricta)
    if not isinstance(text, str): return ""
    text = fix_encoding(text)
    text = remove_section_tags(text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\"\n-]", "", text) # Solo caracteres ingleses y puntuación
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def clean_text_tfidf(text: str) -> str:
    # Aplanado, minúsculas, sin puntuación.
    if not isinstance(text, str): return ""
    text = fix_encoding(text)
    text = remove_section_tags(text)
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r"[^a-z0-9\s]", "", text) # Solo letras y números
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)

def process_lyrics(text: str, method: str):
    if method == "bert": return clean_text_bert(text)
    elif method == "tfidf": return clean_text_tfidf(text)
    return text

# --- ORQUESTADOR ---
def save_clean_lyrics(aligned_csv_path: Path, output_path: Path, format='parquet'):
    print(f"   Leyendo: {aligned_csv_path}")
    df = pd.read_csv(aligned_csv_path)

    # 1. ESTANDARIZACIÓN DE ID (CRÍTICO PARA EQUIPO)
    # Buscamos el ID y lo renombramos a 'spotify_id' para que siempre sea igual
    if 'spotify_id' in df.columns:
        id_col = 'spotify_id'
    elif 'musicId' in df.columns:
        df.rename(columns={'musicId': 'spotify_id'}, inplace=True)
        id_col = 'spotify_id'
    else:
        # Fallback: usar la primera columna y renombrarla
        first_col = df.columns[0]
        df.rename(columns={first_col: 'spotify_id'}, inplace=True)
        id_col = 'spotify_id'

    print(f"   Usando columna ID estandarizada: {id_col}")

    # 2. GENERAR COLUMNAS LIMPIAS
    print("   Generando clean_lyrics_tfidf...")
    df["clean_lyrics_tfidf"] = df["lyrics"].apply(lambda x: process_lyrics(x, "tfidf"))

    print("   Generando clean_lyrics_bert...")
    df["clean_lyrics_bert"] = df["lyrics"].apply(lambda x: process_lyrics(x, "bert"))

    # 3. SELECCIONAR SOLO LAS 3 COLUMNAS ACORDADAS
    cols_to_keep = ["spotify_id", "clean_lyrics_tfidf", "clean_lyrics_bert"]
    df_clean = df[cols_to_keep]

    # 4. GUARDAR
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if format == 'parquet':
        df_clean.to_parquet(output_path, index=False)
    else:
        df_clean.to_csv(output_path, index=False)

    print(f"   ✅ Archivo guardado con estructura correcta: {output_path}")
    print(f"   Columnas finales: {df_clean.columns.tolist()}")

if __name__ == "__main__":
    pass