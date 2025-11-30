"""
Paso 3: Descarga de Audio (YouTube -> MP3) - Versión Anti-Bot "Level 2" (Cookies + Random Queries)
------------------------------------------
Recibe: CSV Maestro del Paso 2 (con letras y metadata validada).
Hace:   1. Lee el CSV para obtener 'Artist - Track Name' y 'spotify_id'.
        2. Busca el mejor resultado en YouTube.
        3. Descarga y convierte a MP3 (192kbps).
        4. Renombra el archivo a {spotify_id}.mp3.
Devuelve: Carpeta data/raw_v2/audio poblada.
-------------------------------------------------------------
Mejoras:
1. Retrasos aleatorios (Sleep) entre descargas.
2. Detección de Bloqueo de Bot (Detiene el script si Google nos bloquea).
3. Manejo de errores de restricción de edad.
4. Uso de Cookies del navegador (Bypass "Not a bot" y "Age restricted") opcional.
5. Queries de búsqueda aleatorios (Humanización).
"""

import os
import time
import random
import pandas as pd
import yt_dlp
from pathlib import Path

# === CONFIGURACIÓN ===
# Cambia esto por tu navegador: 'chrome', 'firefox', 'edge', 'opera'
BROWSER_FOR_COOKIES = 'chrome' 
USE_COOKIES = False  # Pon False si no quieres arriesgar una cuenta o si falla la extracción

def get_random_query(artist, track_name):
    """Genera una búsqueda diferente cada vez para despistar."""
    templates = [
        f"{artist} - {track_name} audio",
        f"{artist} {track_name} lyrics",
        f"{track_name} by {artist}",
        f"{artist} {track_name} official audio",
        f"{track_name} {artist} hq audio"
    ]
    return random.choice(templates)

def download_one_track(artist, track_name, output_path):
    
    query = get_random_query(artist, track_name)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_path) + '.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'postprocessor_args': [
            '-ss', '30', 
            '-t', '30' 
        ],
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'default_search': 'ytsearch1',
        
        # === NIVEL 2: COOKIES ===
        # Esto extrae las cookies de tu navegador local.
        # ¡Cierra el navegador antes de ejecutar!
        'cookiesfrombrowser': (BROWSER_FOR_COOKIES,) if USE_COOKIES else None,
    }

    # Eliminar clave si es None para evitar errores de yt-dlp
    if not ydl_opts['cookiesfrombrowser']:
        del ydl_opts['cookiesfrombrowser']

    try:
        # Nota: Usamos la query dinámica aquí
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([query])
        return True
    
    except Exception as e:
        error_str = str(e)
        
        # DETECCIÓN DE BLOQUEO DE BOT
        # Con cookies, este error debería desaparecer, pero si la cuenta es flaggeada, aparecerá.
        if "Sign in to confirm you’re not a bot" in error_str:
            print(f"\n🚨 BLOQUEO CRÍTICO: YouTube rechazó incluso con cookies.")
            return 'BOT_BLOCK'
        
        if "Sign in to confirm your age" in error_str:
            print(f"   ⚠️ Restricción de edad (Cookies fallaron o no usadas).")
            return False # No detenemos el script, solo saltamos
            
        if "Video unavailable" in error_str:
            print(f"   ⚠️ Video no disponible.")
            return False

        print(f"   ❌ Error: {error_str[:100]}...")    
        return False

def download_audio_batch(input_csv, audio_output_dir):
    print(f"   📂 Cargando lista desde: {input_csv}")
    df = pd.read_csv(input_csv)
    total = len(df)
    
    print(f"   📂 Destino: {audio_output_dir}")
    print(f"   🍪 Cookies: {'ACTIVADAS (' + BROWSER_FOR_COOKIES + ')' if USE_COOKIES else 'DESACTIVADAS'}")
    
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    skipped_count = 0
    
    for i, row in df.iterrows():
        spotify_id = row['spotify_id']
        artist = row['artist']
        track_name = row['track_name']
        
        target_file = audio_output_dir / f"{spotify_id}.mp3"
        
        if target_file.exists():
            skipped_count += 1
            continue
            
        print(f"   [{i+1}/{total}] 🔎 {artist} - {track_name}...", end=" ", flush=True)
        
        # --- PAUSA ---
        # Con cookies podemos ser un POCO más rápidos, pero mantengamos la calma
        time.sleep(random.uniform(6.0, 15.0))
        
        base_path = target_file.parent / target_file.stem 
        result = download_one_track(artist, track_name, base_path)
        
        if result == True:
            print("✅")
            success_count += 1
        elif result == 'BOT_BLOCK':
            print("\n\n🛑 DETENIENDO SCRIPT (Bloqueo detectado).")
            break 
        else:
            print("❌")

    print("\n" + "="*50)
    print(f"   ✅ Descargados sesión: {success_count}")
    print(f"   ⏭️  Ya existían: {skipped_count}")
    print("="*50)