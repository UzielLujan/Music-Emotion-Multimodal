# -*- coding: utf-8 -*-
"""
Detecta qué pistas de FMA tienen archivo MP3 real dentro del dataset FMA Medium.
Autor: Uziel Luján (Uzi) & Brenda Tránsito
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm

def detect_mp3_tracks(meta_dir, audio_dir, limit=None):
    """
    Detecta qué track_ids de tracks.csv tienen archivo MP3 real en fma_medium/.

    Parámetros
    ----------
    meta_dir : Path or str
        Directorio donde está fma_metadata (contiene tracks.csv).
    audio_dir : Path or str
        Directorio raíz de fma_medium (contiene carpetas 000/, 001/, ...).
    limit : int or None
        Si se asigna, revisa solo los primeros N track_ids (útil para pruebas).

    Retorna
    -------
    set
        Conjunto de track_ids que tienen .mp3 real.
    """

    meta_dir = Path(meta_dir)
    audio_dir = Path(audio_dir)

    tracks_path = meta_dir / "tracks.csv"
    if not tracks_path.exists():
        raise FileNotFoundError(f"No se encontró tracks.csv en {tracks_path}")

    print("=== Detectando MP3 reales en FMA Medium ===")
    print(f"Metadata: {tracks_path}")
    print(f"Audio dir: {audio_dir}")

    # Cargar metadata
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0,1])
    all_ids = tracks.index.tolist()

    if limit is not None:
        print(f"[INFO] Limitando búsqueda a los primeros {limit} tracks.")
        all_ids = all_ids[:limit]

    mp3_ids = set()

    for tid in tqdm(all_ids, desc="Buscando archivos .mp3"):
        # Los nombres de carpeta siguen formato basado en los primeros 3 dígitos
        folder = f"{tid:06d}"[:3]
        mp3_path = audio_dir / folder / f"{tid:06d}.mp3"

        if mp3_path.exists():
            mp3_ids.add(tid)

    print(f"\n[✔] MP3 reales encontrados: {len(mp3_ids)} / {len(all_ids)}")

    return mp3_ids


# Ejemplo de uso directo:
if __name__ == "__main__":
    META = Path("../../data/raw/fma_metadata")
    AUDIO = Path("../../data/raw/fma_medium")
    ids = detect_mp3_tracks(META, AUDIO, limit=5000)
    print("Ejemplo completado.")
