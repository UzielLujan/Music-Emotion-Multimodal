"""
Paso 2: Filtro y Extracción de Letras (Genius) - Versión FINAL BLINDADA
-----------------------------------------------------------------------
Mejoras:
1. Detecta el error 429 incluso si viene en formato texto.
2. Se DETIENE TOTALMENTE (break) al primer signo de ban.
3. Mantiene el guardado incremental.
"""

import os
import re
import time
import random
import pandas as pd
import lyricsgenius
from langdetect import detect, LangDetectException
from dotenv import load_dotenv

load_dotenv()

GENIUS_TOKEN = os.getenv("GENIUS_API_KEY")

# Timeout alto para evitar falsos positivos
genius = lyricsgenius.Genius(GENIUS_TOKEN, verbose=False, remove_section_headers=True, sleep_time=0.5, retries=3, timeout=20)

def clean_title_for_search(title):
    title = re.sub(r"\(.*?\)|\[.*?\]", "", title)
    title = title.split("-")[0]
    return title.strip()

def is_english(text):
    """
    Detector de inglés híbrido: Probabilístico + Vocabulario Básico.
    Recupera falsos negativos en canciones repetitivas o cortas.
    """
    # 1. Intento estándar con langdetect
    try:
        if detect(text) == 'en':
            return True
    except LangDetectException:
        pass # Si falla, pasamos al plan B

    # 2. Red de Seguridad: Vocabulario Básico (Stop Words)
    
    # Lista de 50 palabras más comunes en letras de canciones en inglés
    common_english_words = {
        'the', 'and', 'is', 'to', 'in', 'you', 'me', 'it', 'of', 'for', 
        'on', 'my', 'that', 'your', 'we', 'are', 'be', 'do', 'can', 'all',
        'love', 'baby', 'yeah', 'oh', 'know', 'got', 'like', 'just', 'go',
        'with', 'but', 'so', 'what', 'now', 'get', 'up', 'out', 'if', 
        'this', 'one', 'time', 'see', 'will', 'don', 'no', 'way', 'make'
    }
    
    # Tokenización simple: pasar a minúsculas y separar por espacios/signos
    # (re.findall extrae solo palabras alfanuméricas)
    words_in_lyrics = set(re.findall(r'\b[a-z]+\b', text.lower()))
    
    # Contamos cuántas palabras comunes aparecen en la letra
    matches = words_in_lyrics.intersection(common_english_words)
    
    # UMBRAL: Si encontramos al menos 5 palabras únicas de la lista, es inglés.
    # (Ej. "Baby Shark" tiene: baby, do, the, end, it, at, safe... -> Pasa el filtro)
    if len(matches) >= 5:
        # print(f"   (Rescatada por vocabulario: {len(matches)} matches)") # Descomentar para debug
        return True
        
    return False

def filter_and_fetch_lyrics(input_csv, output_csv):
    print(f"   📂 Cargando metadata desde: {input_csv}")
    df = pd.read_csv(input_csv)
    total = len(df)
    
    processed_ids = set()
    if os.path.exists(output_csv):
        try:
            df_existing = pd.read_csv(output_csv)
            processed_ids = set(df_existing['spotify_id'].tolist())
            print(f"   🔄 Checkpoint detectado: {len(processed_ids)} canciones ya procesadas.")
        except Exception:
            pass

    print(f"   🎯 Objetivo: Procesar {total} canciones.")
    
    success_count = 0
    skipped_count = 0
    
    for i, row in df.iterrows():
        spotify_id = row['spotify_id']
        if spotify_id in processed_ids:
            continue

        artist = row['artist']
        track = row['track_name']
        
        print(f"   [{i+1}/{total}] {artist} - {track}...", end=" ", flush=True)

        if row.get('instrumentalness', 0) > 0.5:
            print("⏭️  Instr. (Meta)")
            skipped_count += 1
            continue

        search_title = clean_title_for_search(track)
        
        try:
            # RETRASO OBLIGATORIO: 3 a 7 segundos para calmar al servidor
            time.sleep(random.uniform(3.0, 7.0))
            
            song = genius.search_song(search_title, artist)
            
            if song and song.lyrics:
                lyrics = song.lyrics
                if "instrumental" in lyrics.lower() and len(lyrics) < 200:
                    print("⏭️  Instr. (Lyrics)")
                    skipped_count += 1
                elif not is_english(lyrics):
                    print("⏭️  No Inglés")
                    skipped_count += 1
                else:
                    entry = row.to_dict()
                    entry['lyrics'] = lyrics
                    entry['genius_url'] = song.url
                    
                    df_row = pd.DataFrame([entry])
                    header = not os.path.isfile(output_csv)
                    df_row.to_csv(output_csv, mode='a', header=header, index=False)
                    print("✅ Guardado")
                    success_count += 1
            else:
                print("❌ No hallada")
                skipped_count += 1

        except Exception as e:
            # ANÁLISIS DE ERROR INTELIGENTE
            error_str = str(e)
            
            # Si el error contiene "429" o "Response body: error code: 1015" (Cloudflare)
            if "429" in error_str or "1015" in error_str:
                print("\n\n🚨🚨 ¡ALERTA DE BAN! DETENIENDO INMEDIATAMENTE 🚨🚨")
                print(f"   El servidor nos ha bloqueado. Mensaje: {error_str[:100]}...")
                print("   🚫 NO REINICIES EL SCRIPT POR AL MENOS 1 HORA.")
                break # <--- ESTO ES VITAL: ROMPE EL CICLO
            
            elif "Timeout" in error_str:
                print("⏳ Timeout")
            else:
                print(f"⚠️ Error: {error_str[:50]}...")

    print("\n" + "="*30)
    print(f"   ✅ Guardados hoy: {success_count}")
    print("="*30)