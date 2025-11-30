# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 22:58:11 2025

@author: Brenda Tránsito
"""

import os
import pandas as pd
import lyricsgenius # type: ignore 

def fetch_lyrics(english_ids, metadata_dir, out_path):
    # Cargar tracks.csv con MultiIndex
    tracks = pd.read_csv(
        os.path.join(metadata_dir, "tracks.csv"),
        index_col=0,
        header=[0, 1]
    )

    # Inicializar Genius
    genius_token = os.getenv("GENIUS_API_KEY")
    if genius_token is None:
        raise ValueError("No se encontró la variable de entorno GENIUS_API_KEY")

    # Inicializar Genius API
    genius = lyricsgenius.Genius(
        genius_token,
        timeout=15,
        sleep_time=1,
        retries=3
    )
    
    # Crear carpeta de salida
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = []

    for tid in english_ids:
        try:
            title = tracks.loc[tid, ("track", "title")]
            artist = tracks.loc[tid, ("artist", "name")]

            print(f"Buscando letra: {artist} - {title}")

            song = genius.search_song(title, artist)

            if song:
                lyrics = song.lyrics
            else:
                lyrics = None

            results.append({
                "track_id": tid,
                "artist": artist,
                "title": title,
                "lyrics": lyrics
            })

        except Exception as e:
            print(f"Error con track {tid}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    print(f"\n=== Letras guardadas en: {out_path} ===\n")

    return df