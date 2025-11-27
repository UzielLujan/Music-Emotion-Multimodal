# -*- coding: utf-8 -*-
"""
fetch_lyrics_v2 — versión optimizada, compatible con lyricsgenius 2.x
"""

import os
import time
import json
from pathlib import Path
import pandas as pd
import lyricsgenius
from tqdm import tqdm


def fetch_lyrics_v2(track_ids, metadata_dir, out_csv, cache_json=None, batch_size=20):
    """
    Descarga letras de Genius de forma eficiente y segura.
    
    Parámetros:
        track_ids: lista de track IDs
        metadata_dir: directorio de fma_metadata (tracks.csv)
        out_csv: ruta donde guardar el CSV final
        cache_json: archivo JSON cacheado (para no repetir búsquedas)
        batch_size: cuántas letras guardar antes de hacer flush a disco
    """

    metadata_dir = Path(metadata_dir)
    tracks_path = metadata_dir / "tracks.csv"

    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])

    # --- INICIAR GENIUS API (SINV reintentos para lyricsgenius 2.x) ---
    genius_token = os.getenv("GENIUS_API_KEY")
    if not genius_token:
        raise ValueError("GENIUS_API_KEY no encontrada")

    genius = lyricsgenius.Genius(
        genius_token,
        timeout=15,
        sleep_time=1,
        remove_section_headers=True,
        verbose=False
    )

    # --- CARGAR CACHE PARA NO REPETIR CONSULTAS ---
    if cache_json and Path(cache_json).exists():
        with open(cache_json, "r", encoding="utf8") as f:
            cache = json.load(f)
    else:
        cache = {}

    results = []
    os.makedirs(Path(out_csv).parent, exist_ok=True)

    print("\n=== Descargando letras (versión optimizada) ===")
    for tid in tqdm(track_ids, desc="Lyrics"):
        tid_str = str(tid)

        # Si ya está en cache NO consultamos nada
        if tid_str in cache:
            results.append(cache[tid_str])
            continue

        try:
            title = tracks.loc[tid, ("track", "title")]
            artist = tracks.loc[tid, ("artist", "name")]

            # NO consultamos si falta título o artista
            if pd.isna(title) or pd.isna(artist):
                cache[tid_str] = {
                    "track_id": tid,
                    "artist": artist,
                    "title": title,
                    "lyrics": None
                }
                results.append(cache[tid_str])
                continue

            # Probabilidad muy alta de instrumental: título corto
            if len(str(title).split()) <= 1:
                cache[tid_str] = {
                    "track_id": tid,
                    "artist": artist,
                    "title": title,
                    "lyrics": None
                }
                results.append(cache[tid_str])
                continue

            # --- CONSULTA REAL ---
            song = genius.search_song(title, artist)
            lyrics = song.lyrics if song else None

            cache[tid_str] = {
                "track_id": tid,
                "artist": artist,
                "title": title,
                "lyrics": lyrics
            }
            results.append(cache[tid_str])

        except Exception as e:
            cache[tid_str] = {
                "track_id": tid,
                "artist": None,
                "title": None,
                "lyrics": None
            }
            results.append(cache[tid_str])

        # --- GUARDADO INCREMENTAL ---
        if len(results) % batch_size == 0:
            pd.DataFrame(results).to_csv(out_csv, index=False)
            with open(cache_json, "w", encoding="utf8") as f:
                json.dump(cache, f, indent=2)

            # dormir ligeramente para respetar rate-limit
            time.sleep(1.0)

    # --- GUARDADO FINAL ---
    pd.DataFrame(results).to_csv(out_csv, index=False)
    if cache_json:
        with open(cache_json, "w", encoding="utf8") as f:
            json.dump(cache, f, indent=2)

    print(f"\n=== Letras guardadas en: {out_csv} ===")
    return pd.DataFrame(results)
