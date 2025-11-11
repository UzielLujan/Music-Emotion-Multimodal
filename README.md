# Análisis de emoción y sentimiento en música desde una perspectiva multimodal

Sistema de reconocimiento de emociones musicales utilizando características acústicas y textuales, con modelos de aprendizaje profundo y fusión basada en **stacking**. 

El proyecto incluye pipelines modulares para preprocesamiento, modelado unimodal y fusión multimodal, enfocado en la reproducibilidad y la extensibilidad dentro de la investigación en MIR (Music Information Retrieval).


## Descripción general

Este repositorio forma parte del proyecto final del curso  
**Temas Selectos de Ciencia de Datos: Recuperación de Información Musical (MIR) — 2025. Centro de Investigación en Matemáticas (CIMAT) Unidad Monterrey.**

El objetivo principal es analizar y clasificar **emociones en música** desde una **perspectiva multimodal**, integrando datos que provienen tanto de **audio** como de **letras**.  
El diseño se basa en enfoques recientes de MIR que utilizan arquitecturas de redes profundas y estrategias de fusión por aprendizaje supervisado.



## Objetivos

- Construir un pipeline reproducible para clasificación emocional multimodal.
- Combinar representaciones acústicas y textuales para enriquecer el análisis emocional.
- Implementar y comparar modelos unimodales (audio / letras).
- Diseñar un modelo de fusión por *stacking* para integrar decisiones multimodales.
- Documentar el proceso en formato técnico y académico.

---

## Estructura del proyecto

```bash
music-emotion-multimodal/
│
├── data/ # Conjuntos de datos (crudos / procesados)
│ ├── raw/
│ └── processed/
│
├── src/ # Código fuente del modelo
│ ├── audio/
│ ├── text/
│ ├── fusion/
│ └── utils/
│
├── notebooks/ # Experimentos exploratorios y prototipos
│
├── results/ # Resultados, métricas, gráficas
│
├── reports/ # Reportes en LaTeX
│
├── docs/ # Documentación técnica y decisiones de diseño
│
├── requirements.txt
├── README.md
└── .gitignore
```


---

## Equipo

| Nombre | Responsabilidades adoptadas en el proyecto |
|--------|------------------|
| **Brenda Transito** | - Curación y construcción del conjunto de datos multimodal. |
| **Uziel Luján** | - Estructura del repositorio, diseño del pipeline, documentación técnica. |

---

## Stack tecnológico propuesto

- **Python 3.10+**
- **PyTorch / TensorFlow** (por definir)
- **Librosa** – extracción de características acústicas  
- **Transformers (Hugging Face)** – embeddings textuales  
- **Scikit-learn** – modelos dde machine learning tradicionales y evaluación  
- **Pandas / NumPy** – manipulación y análisis de datos 
- **Matplotlib / Seaborn / Plotly** – visualizaciones

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/UzielLujan/Music-Emotion-Multimodal.git
cd Music-Emotion-Multimodal

# Crear entorno Conda desde el archivo environment.yaml
conda env create -f environment.yaml

# Activar entorno
conda activate mem-env
```

