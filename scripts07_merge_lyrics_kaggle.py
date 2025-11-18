# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:04:20 2025

@author: Brenda Tránsito
"""

# scripts/scripts07_merge_lyrics_kaggle.py
import os
import pandas as pd
import re
from unidecode import unidecode
from rapidfuzz import process, fuzz

def clean_text(s):
    if pd.isna(s):
        return ""
    s = s.lower()
    s = unidecode(s)  # remove accents
    s = re.sub(r"\(.*?\)", "", s)  # remove (feat...), (remix)
    s = re.sub(r"[^a-z0-9\s]", "", s)  # keep alphanumeric
    s = re.sub(r"\s+", " ", s).strip()
    return s

def generate_key(artist, title):
    return clean_text(artist) + " - " + clean_text(title)

def merge_kaggle_lyrics(df_fma, kaggle_lyrics_path, fuzzy_threshold=90):
    """
    Une letras del dataset de Kaggle con canciones FMA usando:
    - match exacto
    - fuzzy matching (RapidFuzz)
    """

    # === cargar letras de Kaggle ===
    df_lyrics = pd.read_csv(kaggle_lyrics_path)
    df_lyrics["lyrics"] = df_lyrics["lyrics"].astype(str)

    # Normalizar texto
    df_lyrics["artist_clean"] = df_lyrics["artist"].apply(clean_text)
    df_lyrics["title_clean"] = df_lyrics["song"].apply(clean_text)
    df_lyrics["key"] = df_lyrics.apply(lambda x: generate_key(x["artist"], x["song"]), axis=1)

    # Normalizar FMA
    df_fma["artist_clean"] = df_fma["artist"].apply(clean_text)
    df_fma["title_clean"] = df_fma["title"].apply(clean_text)
    df_fma["key"] = df_fma.apply(lambda x: generate_key(x["artist"], x["title"]), axis=1)

    # === Merge exacto ===
    df_exact = df_fma.merge(df_lyrics[["key", "lyrics"]], on="key", how="left")
    print(f"Coincidencias EXACTAS: {df_exact['lyrics'].notna().sum()}")

    # === Fuzzy matching para los que no tuvieron letra ===
    missing = df_exact[df_exact["lyrics"].isna()].copy()
    print(f"Canciones sin letra para fuzzy matching: {len(missing)}")

    kaggle_keys = df_lyrics["key"].tolist()

    fuzzy_matches = []
    for key in missing["key"]:
        match = process.extractOne(key, kaggle_keys, scorer=fuzz.token_set_ratio)
        if match and match[1] >= fuzzy_threshold:
            fuzzy_matches.append((key, match[0]))

    fuzzy_df = pd.DataFrame(fuzzy_matches, columns=["key", "matched_key"])

    df_fuzzy_merged = fuzzy_df.merge(
        df_lyrics[["key", "lyrics"]].rename(columns={"key": "matched_key"}),
        on="matched_key",
        how="left"
    )

    print(f"Canciones añadidas por FUZZY matching: {df_fuzzy_merged['lyrics'].notna().sum()}")

    # Unir fuzzy con el exacto
    df_final = df_exact.merge(df_fuzzy_merged[["key", "lyrics"]], on="key", how="left", suffixes=("", "_fuzzy"))
    df_final["lyrics"] = df_final["lyrics"].fillna(df_final["lyrics_fuzzy"])
    df_final = df_final.drop(columns=["lyrics_fuzzy"])

    print(f"TOTAL canciones con letra después del merge: {df_final['lyrics'].notna().sum()}")

    return df_final
