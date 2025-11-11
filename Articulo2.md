# Bitácora técnica: Clasificación multimodal de emociones en música 
**Artículo analizado:** *Music Emotion Classification Method Based on Deep Learning and Explicit Sparse Attention Network*  
**Proyecto MIR:** *Análisis de emoción y sentimiento en música desde una perspectiva multimodal*

--- 
## 1. Propósito del artículo
El artículo propone un método para mejorar la clasificación de emociones en música mediante una arquitectura de red profunda que combina técnicas de procesamiento acústico y atención selectiva. Su objetivo principal es abordar los desafíos de ruido, subjetividad y alta dimensionalidad presentes en datos musicales reales, proponiendo un modelo que:

- Integra espectrogramas y descriptores acústicos clásicos (LLDs/HSFs).
- Utiliza una arquitectura híbrida CNN-LSTM para capturar patrones espaciales y temporales.
- Introduce un mecanismo de atención explícitamente dispersa (Explicit Sparse Attention Network) que filtra activamente las características irrelevantes.
- Clasifica canciones en cuatro emociones básicas: **happy**, **sad**, **relax** y **anger**.

El enfoque se centra exclusivamente en señales acústicas, sin incorporar información lírica o textual, pero con un pipeline robusto y replicable.


## 2. Arquitectura propuesta

El modelo propuesto combina tres componentes principales:

### 🔹 1. Modelo híbrido CNN-LSTM + DNN
- **CNN**: extrae características espaciales de espectrogramas mediante tres capas de convolución + max pooling.
- **LSTM bidireccional**: captura dependencias temporales en la secuencia de espectrogramas procesados.
- **DNN**: procesa los descriptores acústicos clásicos (LLDs), convertidos en HSFs mediante estadísticas (media, varianza, máximo).
- Las salidas de CNN-LSTM y DNN se concatenan y se envían a una capa Softmax para clasificación.

### 🔸 2. Fusión de espectrogramas + LLDs/HSFs
- **Espectrogramas**: tratados como imágenes RGB de tamaño 512×512×4.
- **LLDs**: incluyen MFCC-13, ZCR, centroides espectrales, bandwidth, flux, roll-off, cromaticidad.
- **HSFs**: se obtienen aplicando estadísticas sobre los LLDs.
- Esta fusión permite representar tanto la estructura espectral como los patrones acústicos globales.

### 🔺 3. Mecanismo de atención explícitamente dispersa
Este componente reemplaza la atención estándar por una variante que **filtra activamente las características irrelevantes** antes del softmax.

#### 🧩 Funcionamiento paso a paso:
1. Se calcula la matriz de atención $K$ como en el mecanismo estándar (producto escalar entre claves y consultas).
2. Para cada fila $u$ de $K$, se identifica el umbral $t_u$ correspondiente al *top-d* valores más altos.
3. Se aplica una **máscara** que conserva solo los elementos $k_{um} \geq t_u$, y asigna $-\infty$ al resto:
$$ M(K, D ) = \begin{cases} k_{um}, & \text{si } k_{um} \geq t_u \\ -\infty, & \text{si } k_{um} < t_u \end{cases} $$
- Esto fuerza a que el softmax posterior asigne peso cero a los elementos descartados.
4. Se normaliza la matriz enmascarada con softmax, obteniendo una distribución de atención **concentrada en los elementos más relevantes**.
5. Durante backpropagation, se calcula el gradiente solo sobre los elementos seleccionados, lo que reduce la dispersión y mejora la eficiencia.

#### ⚙️ Implementación práctica:
- Puede implementarse como una capa personalizada en PyTorch o TensorFlow.
- Requiere definir:
  - Umbral dinámico por fila (top-d).
  - Operación de máscara antes del softmax.
  - Propagación de gradientes solo sobre elementos seleccionados.
- Ideal para tareas con ruido o alta dimensionalidad, como música real con mezcla vocal/instrumental.

### 🎯 Clasificación final
- La salida fusionada se clasifica en una de cuatro emociones: `happy`, `sad`, `relax`, `anger`.
- Se utiliza **votación por bloques segmentados** para decidir la emoción dominante de cada canción, es decir, se divide la canción en segmentos (10s, 15s, 25s), se clasifica cada segmento y se elige la emoción más frecuente.


## 3. Resultados clave

El artículo presenta una serie de experimentos comparativos que demuestran la efectividad del enfoque propuesto. Los resultados más relevantes son:

### 📊 Precisión de clasificación
- **Accuracy promedio del modelo completo**: `0.712`
- Emociones individuales:
  - Happy: `0.737`
  - Sad: `0.723`
  - Relax: `0.698`
  - Anger: `0.688`

### 🔍 Comparación de mecanismos de atención
| Mecanismo de atención         | Accuracy | Cross-entropy |
|------------------------------|----------|----------------|
| Tradicional (soft attention) | 0.682    | 0.654          |
| Explícitamente dispersa      | **0.712**| **0.631**      |

La atención dispersa mejora tanto la precisión como la entropía cruzada, indicando una distribución más enfocada y eficiente.

### 🧪 Comparación de preprocesamiento
| Método de preprocesamiento             | Accuracy promedio |
|----------------------------------------|-------------------|
| Segmentación fina                      | 0.650             |
| Separación vocal                       | 0.653             |
| Segmentación + separación              | 0.679             |
| Método completo (con atención dispersa)| **0.712**         |

Separar la voz del fondo musical y aplicar segmentación mejora significativamente el rendimiento del modelo, validando la utilidad de estas técnicas en entornos acústicos complejos.



## 4. Dataset utilizado

El artículo no utiliza un dataset público ni proporciona enlaces a recursos externos. En su lugar, propone construir un corpus propio a partir de plataformas musicales chinas, siguiendo estos criterios:

- Selección de canciones con más de 3 millones de reproducciones, para asegurar relevancia y popularidad.
- Uso de listas de reproducción etiquetadas emocionalmente (*happy*, *sad*, *relax*, *anger*).
- Filtrado por calidad de audio, duración y lenguaje.
- Resultado final: 2147 canciones divididas en conjunto de entrenamiento (80%) y prueba (20%).

### 🔍 Observaciones
- El procedimiento es **general y flexible**, lo que permite replicarlo con otras plataformas (Spotify, YouTube Music, etc.).
- No se incluyen metadatos, letras ni anotaciones manuales.
- Las etiquetas emocionales provienen de las playlists, lo que implica una **etiquetación débil pero escalable**.

Este enfoque es útil como guía para construir datasets personalizados, aunque no garantiza comparabilidad directa con otros estudios.


## 5. Cómo recrear el dataset
## 🛠️ Cómo recrear el dataset

Pasos sugeridos para construir un corpus emocional acústico replicable:

1. **Selección de playlists con etiquetas emocionales**
   - Buscar listas de reproducción públicas con etiquetas como `happy`, `sad`, `relax`, `anger`.
   - Priorizar aquellas con alto número de reproducciones para asegurar coherencia emocional.

2. **Filtrado por calidad, duración y lenguaje**
   - Eliminar canciones con baja calidad de audio, duración atípica (<1 min o >6 min), o idiomas no deseados.
   - Opcional: filtrar por género musical si se desea controlar la variabilidad.

3. **Descarga de audio (MP3)**
   - Usar herramientas de extracción desde plataformas musicales (respetando términos de uso).
   - Guardar los archivos con metadatos básicos: título, artista, etiqueta emocional.

4. **Segmentación en bloques (10s, 15s, 25s)**
   - Dividir cada canción en fragmentos temporales fijos.
   - Cada bloque se trata como una unidad de análisis independiente.
   - Esta estrategia simula la percepción humana: juzgamos la emoción de una canción por cómo se desarrolla, no por un instante aislado.
   - Permite aplicar clasificación por fragmento y luego realizar **votación por mayoría** para determinar la emoción dominante.

5. **Separación de voz y fondo (Spleeter/Demucs)**
   - Aplicar separación de fuentes para obtener:
     - Pista vocal pura
     - Pista instrumental pura
   - Esto permite analizar por separado las emociones transmitidas por letra y música.

6. **Extracción de características acústicas**
   - Por bloque y por pista (voz/fondo), extraer:
     - MFCC
     - ZCR
     - Spectral Centroid, Bandwidth, Flux, Roll-off
     - Chroma features
   - Usar herramientas como Librosa o Essentia.

7. **Etiquetado por playlist o curación manual**
   - Asignar la etiqueta emocional de la playlist a cada canción.
   - Opcional: realizar curación manual para verificar coherencia emocional o ajustar clases.

## 6. Flujo de trabajo replicable
- Preprocesamiento: segmentación + separación de fuentes
- Extracción de características: espectrogramas + LLDs
- Modelado: CNN-LSTM + DNN + atención dispersa
- Inferencia: clasificación por bloque + votación por mayoría

## 7. Ideas aprovechables para proyecto MIR
Este artículo ofrece varias estrategias técnicas que pueden integrarse o adaptarse al proyecto de análisis emocional multimodal en música:

### 🔺 Mecanismo de atención explícitamente dispersa
- Reemplaza la atención estándar por una variante que filtra activamente las características irrelevantes.
- Mejora la precisión y la eficiencia del modelo en entornos ruidosos o con alta dimensionalidad.
- Ideal para música real con mezcla vocal/instrumental.

### 🔄 Votación por bloques segmentados
- Divide cada canción en fragmentos temporales (10s, 15s, 25s).
- Clasifica cada bloque por separado y aplica votación por mayoría para determinar la emoción dominante.
- Simula la percepción humana acumulativa de la emoción musical.

### 🎼 Separación de fuentes (voz/fondo)
- Permite analizar por separado las emociones transmitidas por la letra y por la instrumentación.
- Mejora la concentración de las características acústicas y la precisión del modelo.

### 🔀 Fusión de representaciones acústicas
- Combina espectrogramas (procesados por CNN-LSTM) con descriptores acústicos clásicos (LLDs/HSFs procesados por DNN).
- Aporta una visión multimodal dentro del dominio acústico, sin necesidad de texto.

### 🧪 Flujo replicable con herramientas abiertas
- Todo el pipeline puede implementarse con herramientas como Librosa, Spleeter/Demucs, TensorFlow/Keras o PyTorch.
- La arquitectura es modular y adaptable a datasets personalizados.



## 8. Limitaciones
- No incluye análisis lírico ni embeddings textuales
- Dataset limitado a 4 emociones, no reportan recursos públicos disponibles ni un procedimiento detallado de recolección más allá de la plataforma china y etiquetas de playlists, esto dificulta la replicabilidad exacta, más allá de la guía general proporcionada.
- No se reporta validación cruzada ni análisis de overfitting

