import pandas as pd
from pathlib import Path

def explore_fma_metadata(meta_dir, audio_dir):
    meta_dir = Path(meta_dir)
    audio_dir = Path(audio_dir)

    print("=== Explorando metadata FMA ===")

    # === 1. Cargar tracks.csv ===
    tracks_path = meta_dir / "tracks.csv"
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0,1])

    print(f"\nTotal de pistas en tracks.csv: {len(tracks)}")

    # === 2. Análisis de idiomas ===
    lang_col = ("track", "language_code")

    if lang_col in tracks.columns:
        lang_counts = tracks[lang_col].value_counts(dropna=False)
        print("\nFrecuencia de idiomas (language_code):")
        print(lang_counts.head(20))
    else:
        print("\nWARNING: No existe columna language_code en tracks.csv")

    # === 3. Revisar cuántos archivos MP3 existen realmente ===
    print("\nEscaneando presencia de archivos .mp3...")
    mp3_exists = []
    for track_id in tracks.index[:5000]:  # limitado para no tardar años
        folder = f"{track_id:06d}"[:3]
        mp3_path = audio_dir / folder / f"{track_id:06d}.mp3"
        mp3_exists.append(mp3_path.exists())

    print(f"MP3 encontrados: {sum(mp3_exists)} / {len(mp3_exists)} (primeros 5000)")

    # === 4. Cargar echonest.csv (valence/arousal) ===
    echo_path = meta_dir / "echonest.csv"
    echo = pd.read_csv(echo_path, index_col=0, header=[0,1,2])

    col_val = ("echonest", "audio_features", "valence")
    available_va = echo[col_val].notna().sum()

    print(f"\nTracks con valence/arousal disponibles: {available_va} / {len(echo)}")

    print("\n=== Exploración inicial completa ===")
