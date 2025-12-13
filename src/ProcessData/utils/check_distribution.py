import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# === CONFIGURACIÓN DE RUTAS ===
try:
    # Intenta resolver la ruta relativa si se ejecuta como script
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:
    # Fallback por si se ejecuta interactivo
    BASE_DIR = Path(".").resolve()

INPUT_CSV = BASE_DIR / "data" / "interim" / "aligned_metadata.csv"
# Guardamos la imagen en la misma carpeta donde corres el script (o ajusta según prefieras)
OUTPUT_IMG = BASE_DIR / "reports" / "figures" / "class_distribution.png"

def check_balance():
    print("📊 ANALIZANDO DISTRIBUCIÓN Y GENERANDO GRÁFICA...")
    
    if not INPUT_CSV.exists():
        print(f"❌ Error: No se encontró {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    
    if 'label_quadrant' not in df.columns:
        print("⚠️ Error: Columna 'label_quadrant' no encontrada.")
        return

    # Calcular conteos y porcentajes
    counts = df['label_quadrant'].value_counts()
    percentages = df['label_quadrant'].value_counts(normalize=True) * 100
    
    # Ordenar para consistencia visual (Q1, Q2, Q3, Q4 si es posible, o por cantidad)
    # Vamos a ordenar por el índice por defecto (Q1, Q2...) si tienen ese formato, o descendente
    counts = counts.sort_index() 
    
    # --- 1. IMPRIMIR EN TERMINAL (Tu lógica original) ---
    print(f"\n   ✅ Total de canciones: {total}")
    print("\n     DISTRIBUCIÓN DE CLASES:")
    print(f"   {'Cuadrante':<15} | {'Cantidad':<8} | {'%':<6}")
    print("   " + "-"*35)
    
    for label in counts.index:
        print(f"   {label:<15} | {counts[label]:<8} | {percentages[label]:.1f}%")

    # Diagnóstico
    ratio = counts.max() / counts.min()
    print(f"\n   🔍 Ratio de Desbalance: 1:{ratio:.1f}")

    # --- 2. GENERAR GRÁFICA (Nuevo) ---
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Paleta de colores personalizada para emociones (opcional)
    # Q1:Happy(Amarillo/Naranja), Q2:Angry(Rojo), Q3:Sad(Azul), Q4:Relaxed(Verde/Cyan)
    # Ajusta según tus etiquetas exactas. Aquí uso una genérica.
    colors = sns.color_palette("Reds", len(counts))
    
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend= False, edgecolor="black")
    
    plt.title(f"Distribución de Clases Emocionales (Total: {total})", fontsize=14, pad=20)
    plt.ylabel("Número de Canciones", fontsize=12)
    plt.xlabel("Cuadrante Emocional", fontsize=12)
    plt.ylim(0, counts.max() * 1.15) # Espacio para etiquetas

    # Anotar barras
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        perc = percentages[counts.index[i]]
        ax.text(p.get_x() + p.get_width() / 2., height + 20,
                f'{int(height)}\n({perc:.1f}%)',
                ha="center", fontsize=11, fontweight='bold', color='black')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"\n   📸 Gráfica guardada exitosamente: {OUTPUT_IMG.absolute()}")

if __name__ == "__main__":
    check_balance()