# -*- coding: utf-8 -*-
"""
Detecta títulos probablemente en inglés usando heurísticas simples.

Autor: Uziel Luján (Uzi) 
Fecha: 2025
"""

from pathlib import Path
import pandas as pd
import re

# Palabras funcionales comunes del inglés
COMMON_EN_WORDS = {
    "the", "and", "you", "me", "my", "your", "love",
    "of", "in", "on", "at", "for", "is", "are",
    "to", "with", "from", "by", "be", "this", "that",
    "a", "an", "it", "we", "they", "she", "he"
}

def clean_title(title):
    if pd.isna(title):
        return ""
    title = title.lower()
    title = re.sub(r"[^a-z\s]", " ", title)  # solo letras
    title = re.sub(r"\s+", " ", title)
    return title.strip()

def is_probably_english(title, threshold=0.4):
    """
    Determina si el título es probablemente inglés.
    threshold: proporción de palabras del título que deben aparecer en COMMON_EN_WORDS.
    """
    words = title.split()
    if not words:
        return False
    
    matches = sum(1 for w in words if w in COMMON_EN_WORDS)
    ratio = matches / len(words)
    
    return ratio >= threshold

def detect_english_titles(meta_dir, mp3_ids):
    meta_dir = Path(meta_dir)
    tracks_path = meta_dir / "tracks.csv"

    tracks = pd.read_csv(tracks_path, index_col=0, header=[0,1])

    candidate_ids = []

    for tid in mp3_ids:
        try:
            title = tracks.loc[tid, ("track", "title")]
            clean = clean_title(title)
            if is_probably_english(clean):
                candidate_ids.append(tid)
        except Exception:
            pass

    print(f"[✔] Títulos en inglés detectados por heurística: {len(candidate_ids)}")

    return set(candidate_ids)
