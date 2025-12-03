import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === CONFIGURACIÓN DE RUTAS ===
BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_CSV = BASE_DIR / "data" / "interim" / "aligned_metadata.csv"

def check_balance():
    print(" ANALIZANDO DISTRIBUCIÓN FINAL DEL DATASET")
    print(f"    Leyendo: {INPUT_CSV}")
    
    if not INPUT_CSV.exists():
        print("❌ Error: No se encontró el archivo aligned_metadata.csv")
        print("   Asegúrate de ejecutar 'alignment.py' primero.")
        return

    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    
    print(f"\n   ✅ Total de canciones alineadas (Audio + Texto): {total}")
    
    # --- ANÁLISIS POR CUADRANTE ---
    if 'label_quadrant' in df.columns:
        counts = df['label_quadrant'].value_counts()
        percentages = df['label_quadrant'].value_counts(normalize=True) * 100
        
        print("\n     DISTRIBUCIÓN DE CLASES:")
        print(f"   {'Cuadrante':<15} | {'Cantidad':<8} | {'%':<6}")
        print("   " + "-"*35)
        
        for label in counts.index:
            count = counts[label]
            perc = percentages[label]
            print(f"   {label:<15} | {count:<8} | {perc:.1f}%")
            
        # Alerta de Desbalance
        min_class = counts.min()
        max_class = counts.max()
        ratio = max_class / min_class
        
        print("\n   🔍 DIAGNÓSTICO:")
        if ratio > 1.5:
            print(f"      ⚠️ Desbalance moderado detectado (Ratio 1:{ratio:.1f}).")
            print("      Considera usar 'class_weights' al entrenar el modelo.")
        else:
            print("      ✅ Dataset bien balanceado.")
            
    else:
        print("⚠️ No se encontró la columna 'label_quadrant'.")

if __name__ == "__main__":
    check_balance()