# -*- coding: utf-8 -*-
"""
Construye el conjunto real de tracks en inglés basados en:
- idioma oficial (language_code == "en")
- heurística de título (probablemente inglés)
- presencia real de .mp3 en FMA Medium

Autor: Uziel Luján (Uzi)
"""

def build_english_dataset(mp3_ids, english_official_ids, english_heuristic_ids):
    """
    Construye el conjunto real de tracks en inglés.

    Parámetros
    ----------
    mp3_ids : set[int]
        Tracks con archivo MP3 real.
    english_official_ids : set[int]
        Tracks oficialmente etiquetados como inglés en tracks.csv.
    english_heuristic_ids : set[int]
        Tracks detectados por heurística de título.

    Retorna
    -------
    dict
        Diccionario con:
        - official_with_mp3
        - rescued
        - english_real
    """

    # Tracks oficialmente en inglés Y con mp3 real
    official_with_mp3 = english_official_ids & mp3_ids

    # Tracks con título en inglés PERO no oficiales en inglés
    rescued = english_heuristic_ids - english_official_ids

    # Unión final
    english_real = official_with_mp3 | rescued

    return {
        "official_with_mp3": official_with_mp3,
        "rescued": rescued,
        "english_real": english_real
    }


def print_english_stats(stats):
    """ Imprime estadísticas del dataset inglés detectado. """

    off = stats["official_with_mp3"]
    res = stats["rescued"]
    total = stats["english_real"]

    print("\n=== Estadísticas del dataset en inglés ===")
    print(f"Oficiales con MP3 real:        {len(off)}")
    print(f"Rescatados desde NaN:          {len(res)}")
    print(f"----------------------------------------")
    print(f"TOTAL inglés real disponible:   {len(total)}")
    print(f"\nProporción rescatada: {len(res) / max(len(total),1) * 100:.2f}%\n")
