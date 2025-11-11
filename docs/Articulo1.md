# Bitácora técnica: Clasificación multimodal de emociones en música  
**Artículo analizado:** *A Multimodal Music Emotion Classification Method Based on Multifeature Combined Network Classifier*  
**Proyecto MIR:** *Análisis de emoción y sentimiento en música desde una perspectiva multimodal*

---

## 0. 🧾 Resumen sintetizado del artículo

El artículo propone un sistema de clasificación de emociones musicales basado en una arquitectura multimodal que combina características acústicas y textuales. Utiliza una red CNN-LSTM para procesar entradas 2D (espectrogramas y Word2Vec) y una red DNN para entradas 1D (LLDs y vectores CHI). La fusión se realiza mediante *stacking ensemble learning*, evitando pérdida de información por reducción dimensional. Se aplican técnicas de preprocesamiento como separación de voz humana y segmentación fina de audio. El modelo alcanza una precisión promedio del 78% en clasificación multimodal, superando métodos tradicionales de fusión por características o decisiones.

---

## 1. 🎯 Motivación y objetivos del estudio

- **¿Qué problema aborda el artículo?**  
  La clasificación de emociones musicales ha sido abordada tradicionalmente desde modelos unimodales, centrados únicamente en características acústicas o textuales. Estos enfoques presentan limitaciones al tratar con la heterogeneidad de las fuentes de información (audio vs letras), lo que afecta la precisión y riqueza de la clasificación emocional.

- **¿Por qué es relevante en MIR?**  
  La recuperación de información musical (MIR) se beneficia directamente de una clasificación emocional precisa, ya que permite organizar, recomendar y analizar música en función de estados afectivos. Incorporar múltiples modalidades mejora la comprensión del contenido musical y su impacto emocional, alineándose con tendencias actuales en inteligencia artificial aplicada a música.

- **¿Qué propone como solución?**  
  El artículo propone un sistema de clasificación multimodal que combina:
  - Una arquitectura CNN-LSTM para procesar características 2D (espectrogramas y Word2Vec).
  - Una red DNN para características 1D (LLDs y vectores CHI).
  - Un método de fusión por *stacking ensemble learning*, que evita la pérdida de información por reducción dimensional y mejora la integración de modalidades heterogéneas.
  Además, se aplica un preprocesamiento especializado del audio (segmentación fina y separación de voz) para optimizar la calidad de las características extraídas.

---

## 2. 🧠 Arquitectura del modelo propuesto (Multifeature Combined Network Classifier)


### 2.1 Modelo CNN-LSTM previo

El artículo parte de modelos anteriores que combinan redes convolucionales (CNN) y redes de memoria a largo plazo (LSTM) para tareas de clasificación de audio o texto por separado. Estos modelos han demostrado buen desempeño en:

- **Clasificación de emociones en voz**: usando espectrogramas como entrada para CNN-LSTM.
- **Clasificación de texto largo**: aplicando CNN para compresión de características y LSTM para secuencias.

Sin embargo, presentan limitaciones cuando se aplican directamente a música:
- El audio musical es más largo y complejo que el habla.
- Las letras tienen alta dimensionalidad y dispersión semántica.
- Solo utilizan características 2D (espectrogramas o Word2Vec), ignorando otras representaciones útiles como LLDs o vectores de frecuencia.

Esto motiva la necesidad de un modelo más flexible que combine múltiples tipos de características y maneje su heterogeneidad.

---

### 2.2 Clasificador combinado propuesto basado en CNN-LSTM

El modelo propuesto mejora la arquitectura CNN-LSTM tradicional al incorporar múltiples tipos de características por modalidad:

- **Audio**:
  - 2D: espectrogramas → procesados con CNN + BiLSTM
  - 1D: LLDs resumidos (HSFs) → procesados con DNN

- **Letras**:
  - 2D: Word2Vec → procesado con CNN + BiLSTM + atención
  - 1D: vectores de frecuencia por prueba chi-cuadrado → procesados con DNN

La arquitectura se divide en dos ramas:
- **2D + CNN-LSTM**: para características con estructura matricial y secuencial.
- **1D + DNN**: para vectores estadísticos o de frecuencia.

Ambas salidas se concatenan para producir una predicción unimodal (audio o letras). Posteriormente, estas predicciones se integran en una etapa de fusión multimodal (ver sección 3).

**Comentario técnico:**
Este diseño evita la necesidad de normalizar o reducir dimensionalmente las características para fusionarlas desde el inicio. En su lugar, cada tipo de dato se procesa con la arquitectura más adecuada, y la fusión se realiza a nivel de etiquetas.

**Comentario técnico**:
Aunque los embeddings de palabras son vectores 1D, al representar una canción completa como una secuencia de embeddings (uno por palabra), se forma una matriz de tamaño [n_palabras × dimensión_embedding]. Esta estructura 2D permite aplicar convoluciones espaciales, tratándola como una imagen o mapa de características.

---

### 2.3 Descripción específica de los clasificadores de audio y letras

#### 2.3.1 Capa de entrada para clasificación de audio

El modelo utiliza dos tipos de características acústicas:

- **Espectrogramas (2D)**:  
  Se generan mediante transformada de Fourier de corto tiempo (STFT), representando la energía de frecuencias a lo largo del tiempo.  
  Para agregar **secuencia temporal**, se concatenan múltiples espectrogramas de segmentos consecutivos, formando una estructura tipo "video" o mapa temporal que puede ser procesado por LSTM.  
  Esto permite capturar la evolución emocional en el tiempo, no solo patrones estáticos.

- **LLDs (Low-Level Descriptors, 1D)**:  
  Son características extraídas directamente de la señal de audio en cada frame o ventana temporal.  
  Incluyen:
  - MFCC (coeficientes cepstrales en escala Mel)
  - ZCR (tasa de cruce por cero)
  - Centroides espectrales
  - Rolloff espectral
  - Flujo espectral
  - Chroma features (relación con notas musicales)

  Estas características se resumen mediante funciones estadísticas (máximo, media, varianza) para formar los **HSFs (High-Level Statistical Features)**, que se procesan con una red DNN.

**Comentario técnico:**
Los LLDs se extraen de la naturaleza unidimensional de la señal de audio, frame por frame. Aunque cada descriptor es escalar o vectorial, su secuencia en el tiempo puede formar una serie temporal, pero aquí se resumen para formar vectores fijos.

---

#### 2.3.2 Capa de entrada para clasificación de letras

El modelo utiliza dos tipos de representaciones textuales:

- **Word Embedding (2D)**:  
  Se extraen con Word2Vec, generando vectores de dimensión fija para cada palabra.  
  Al representar toda la letra como una secuencia de embeddings, se forma una matriz $[n_{palabras} \times dimensión_{embedding}]$, que se procesa con CNN para extraer patrones semánticos locales.

- **Vector de frecuencia por prueba chi-cuadrado (1D)**:  
  Se utiliza la prueba **CHI** para seleccionar palabras discriminativas entre clases emocionales.  
  A diferencia de TF-IDF, que pondera frecuencia y rareza, CHI mide la **correlación estadística** entre la presencia de una palabra y una clase emocional.  
  Esto permite seleccionar términos más relevantes para clasificación supervisada.

**Comentario técnico:**
La prueba CHI se basa en la dependencia entre variables: si una palabra aparece significativamente más en canciones de una emoción específica, su valor CHI será alto. Esto mejora la discriminación frente a TF-IDF, que no considera etiquetas.


#### 2.3.3 Capa CNN

Para procesar las características bidimensionales (espectrogramas y matrices de Word2Vec), se utiliza una capa CNN con arquitectura multiescala. Esta capa permite extraer patrones espaciales locales relevantes para la clasificación emocional.

- **Estructura de la capa CNN**:
  - 2 capas de convolución + 2 capas de max pooling.
  - Primera convolución: 64 filtros de tamaño $2 \times 2$, stride = 1.
  - Segunda convolución: filtros de tamaño $3 \times 3$.
  - Ambas capas aplican activación ReLU.
  - Las salidas se conectan en secuencia para formar una representación serializada que se envía a la capa LSTM.

- **Proceso de convolución**:
  1. Cada filtro convoluciona localmente sobre la entrada (espectrograma o matriz de embeddings).
  2. Se calcula la activación lineal:  
     $h_{1F}(i) = W_{F} \cdot X(i:i+F-1) + b$
  3. Se concatenan las activaciones:  
     $h_{1F} = [h_{1F}(1), h_{1F}(2), \dots, h_{1F}(H)]$
  4. Se aplica ReLU:  
     $h_{r1F} = \text{ReLU}(h_{1F})$

- **Proceso de pooling**:
  - Se aplica max pooling para reducir dimensionalidad y conservar los valores más representativos.
  - La operación se adapta al tamaño de la muestra.
  - Resultado:  
    $h_{rP1F} = \max(h_{r1F})$

- **Salida serializada**:
  - Se concatenan los resultados de todos los filtros:  
    $h_1 = \text{Concatenate}(h_{rP1F_1}, h_{rP1F_2}, \dots)$
  - Esta salida se conecta directamente a la capa LSTM para capturar dependencias temporales.

**Comentario técnico:**
La CNN actúa como extractor de patrones locales en espectrogramas y matrices de embeddings. Al aplicar múltiples filtros y pooling, se obtiene una representación compacta pero rica. La serialización posterior permite que LSTM procese la secuencia como si fuera una serie temporal, incluso si la entrada original no lo era explícitamente.

#### 2.3.4 Capa BiLSTM y mecanismo de atención

Una vez que la CNN ha extraído y serializado las características 2D, estas se procesan por una capa BiLSTM (LSTM bidireccional) para capturar dependencias temporales en ambas direcciones (pasado y futuro).

- **BiLSTM**:
  - Contiene 128 unidades en cada dirección.
  - La salida es una secuencia de vectores:  
    $$ [r_{(1)}, r_{(2)}, r_{(3)}, \dots, r_{(N)}] $$
  - Cada vector $r_{(i)}$ representa la activación en el paso temporal $i$, considerando contexto anterior y posterior.

- **Atención**:
  - Se aplica un mecanismo de atención para ponderar la importancia de cada paso temporal.
  - Se calcula un peso $a_i$ para cada vector $r_{(i)}$ usando softmax sobre una función de puntuación:  
    $$ a_i = \frac{\exp(f(r_{(i)}))}{\sum_j \exp(f(r_{(j)}))} $$
  - La salida final es una combinación ponderada de todos los vectores:  
    $$ \text{att}_n = \sum_i a_i \cdot r_{(i)} $$

Este mecanismo permite que el modelo se enfoque en los momentos más relevantes emocionalmente dentro de la secuencia, mejorando la capacidad de clasificación.

**Comentario técnico:**
La BiLSTM permite capturar patrones secuenciales en ambas direcciones, lo cual es útil en música y texto donde el contexto emocional puede depender de lo que viene antes y después. El mecanismo de atención actúa como un selector dinámico de momentos clave, similar a cómo un humano se enfoca en ciertas frases o pasajes musicales.

#### 2.3.5 Capa DNN

La red neuronal profunda (DNN) se utiliza para procesar las características unidimensionales (1D) tanto de audio como de letras:

- **Audio**:  
  Se ingresan los HSFs (estadísticos derivados de LLDs), que resumen la señal en vectores compactos.

- **Letras**:  
  Se ingresan los vectores de frecuencia obtenidos por la prueba chi-cuadrado, que representan la relevancia de palabras discriminativas.

La DNN está compuesta por:
- 3 capas ocultas con:
  - 256 nodos
  - 128 nodos
  - 64 nodos
- Cada capa aplica funciones de activación (presumiblemente ReLU) para sintetizar la información.

**Comentario técnico:**  
La DNN actúa como un sintetizador de patrones en vectores fijos, sin necesidad de convolución ni secuencias. Es ideal para procesar características estadísticas o de frecuencia que no tienen estructura espacial ni temporal.

#### 2.3.6 Capa de salida

La capa de salida se encarga de producir la predicción emocional para cada modalidad (audio o letras), combinando las representaciones extraídas por las ramas 2D y 1D.

- **Componentes**:
  - Capa completamente conectada (FC)
  - Activación softmax para clasificación multiclase

- **Proceso**:
  1. Se concatenan las salidas de:
     - CNN + BiLSTM + atención (características 2D)
     - DNN (características 1D)
  2. Esta representación fusionada se pasa por una capa FC.
  3. Se aplica softmax para obtener probabilidades sobre las clases emocionales.

**Comentario técnico:**  
La arquitectura permite que cada tipo de característica contribuya a la decisión final sin necesidad de normalización previa. La fusión se realiza a nivel de representación interna, antes de la clasificación, lo que preserva la riqueza semántica y acústica de cada modalidad.


---

## 3. 🔗 Fusión multimodal

El artículo destaca que las técnicas tradicionales de fusión multimodal —como la fusión temprana (feature fusion) y la fusión tardía (decision fusion)— presentan limitaciones importantes:
- La fusión de características heterogéneas requiere normalización o reducción dimensional, lo que puede causar pérdida de información emocional.
- La fusión de decisiones (por ejemplo, promediar probabilidades) ignora la correlación entre modalidades y no permite aprendizaje conjunto.

Para superar estas limitaciones, se propone un método de *stacking ensemble learning* que integra las salidas de clasificadores unimodales (audio y letras) mediante un subclasificador entrenado sobre sus predicciones.

### 3.1 Construcción del modelo de stacking
La idea central es utilizar las etiquetas predichas por los clasificadores de audio y letras como nuevas características de entrada para un subclasificador que aprende a combinarlas de manera óptima.

**Estructura del modelo de stacking:**

- **Clasificadores base**:
  - Clasificador de audio (CNN-LSTM + DNN)
  - Clasificador de letras (CNN-LSTM + DNN)
  - Cada uno produce una predicción emocional independiente (softmax)

- **Subclasificador**:
  - Recibe como entrada las etiquetas predichas por los clasificadores base.
  - Aprende a combinar estas salidas para producir una predicción final más robusta.
  - Se entrena sobre un nuevo conjunto de datos generado a partir de las salidas de los clasificadores base.

**Ventajas del enfoque:**
- Evita la fusión directa de vectores heterogéneos.
- Preserva la independencia de los modelos unimodales.
- Permite capturar correlaciones entre predicciones de distintas modalidades.
- Mejora la precisión sin modificar los clasificadores originales.

**Comentario técnico:**  
El stacking actúa como una capa de decisión aprendida, que reemplaza la simple combinación lineal de probabilidades. Es especialmente útil cuando las modalidades tienen estructuras y escalas distintas, como ocurre con audio y texto en MIR. En lugar de fusionar directamente las características heterogéneas, se combinan las salidas softmax de los clasificadores unimodales en un nuevo conjunto de datos. Este conjunto representa cada muestra como un vector de predicciones, que luego se utiliza para entrenar un subclasificador supervisado. Así, el modelo aprende a corregir errores, detectar patrones complementarios entre modalidades y mejorar la precisión global sin alterar los modelos base. Esta estrategia convierte la fusión multimodal en un problema de aprendizaje sobre decisiones, manteniendo la especialización de cada rama y evitando la pérdida de información por normalización o reducción dimensional.

### 3.2 Entrenamiento del modelo de stacking

Para evitar el sobreajuste en el subclasificador, se utiliza un esquema de **validación cruzada de 5 pliegues** durante el entrenamiento. Este procedimiento permite generar predicciones confiables de los clasificadores base sin reutilizar directamente los datos de entrenamiento, lo que garantiza que el subclasificador aprenda sobre salidas no sobreajustadas.

---

#### 3.2.1 Procesamiento del conjunto de datos

- El conjunto original contiene **2000 muestras** etiquetadas con emociones (enojado, feliz, relajado, triste).
- Se divide en:
  - **80% entrenamiento** (1600 muestras)
  - **20% prueba** (400 muestras)

- Sobre el conjunto de entrenamiento (1600 muestras), se aplica **validación cruzada de 5 pliegues**:
  - En cada iteración, se entrena el clasificador base (audio o letras) sobre 4 pliegues (1280 muestras) y se predice sobre el pliegue restante (320 muestras).
  - Esto se repite 5 veces, generando predicciones para todo el conjunto de entrenamiento sin reutilizar datos.

- Resultado:
  - Se obtiene un nuevo conjunto de entrenamiento para el subclasificador, compuesto por las **predicciones unimodales** (softmax) de cada muestra.
  - El conjunto de prueba se predice directamente con los clasificadores base entrenados sobre todo el conjunto de entrenamiento.

#### 3.2.2 Entrenamiento de clasificadores base

Cada clasificador (audio y letras) se entrena usando validación cruzada de 5 pliegues. En cada iteración, se generan predicciones sobre el pliegue de validación, que luego se usan como entrada para el subclasificador.

#### 3.2.3 Entrenamiento del subclasificador

Se construye un nuevo conjunto de datos a partir de las predicciones unimodales. Este conjunto se usa para entrenar un subclasificador (capa FC + softmax), que aprende a combinar las decisiones y produce la predicción final multimodal.

---
## 4. Experimentos

### 4.1 Conjunto de datos

El conjunto utilizado en los experimentos proviene del subset de etiquetas de Last.fm dentro del Million Song Dataset. Siguiendo el modelo emocional de Thayer, se extrajeron listas de canciones etiquetadas con cuatro emociones: **angry**, **happy**, **relaxed** y **sad**.

- Se seleccionaron **500 canciones por emoción**, totalizando **2000 muestras**.
- Los archivos de audio y letras fueron descargados manualmente mediante scripts, en función de las listas de etiquetas obtenidas.

**Comentario técnico:**  
Este proceso implica una curación manual del dataset, donde las etiquetas emocionales se derivan de tags públicos en Last.fm. La construcción cuidadosa del conjunto garantiza que cada clase esté balanceada y que las muestras reflejen las emociones objetivo de forma explícita.

### 4.2 Preprocesamiento de audio

Para mejorar la calidad de las características extraídas del audio, se aplicó un método de preprocesamiento en cuatro niveles:

- **Segmentación fina**:  
  Se dividieron los audios en clips de 15 segundos para amplificar la información emocional relevante.

- **Separación de voz humana**:  
  Se extrajeron dos tipos de fragmentos:
  - Clips de voz humana
  - Clips de fondo musical (sin voz)

- **Construcción de datasets**:  
  Se generaron cuatro variantes:
  - Audio original de 30s
  - Audio original de 15s
  - Fondo musical de 15s
  - Voz humana de 15s

- **Evaluación experimental**:  
  Se comparó el rendimiento de clasificación usando LLDs + SVM. Los clips de fondo musical de 15s mostraron la mejor precisión promedio.

**Comentario técnico:**  
La segmentación y separación permiten aislar componentes acústicos más estables y representativos. El fondo musical, al estar libre de variaciones vocales, ofrece una señal más uniforme para la extracción de características estadísticas como LLDs.

### 4.3 Experimentos con audio

Se evaluaron distintos modelos de clasificación sobre las características acústicas extraídas del audio:

- **Modelos comparados**:
  - CNN sobre espectrogramas
  - LSTM sobre espectrogramas
  - CNN-LSTM combinado
  - Modelo propuesto (CNN-LSTM + DNN con espectrogramas + LLDs)

- **Resultados**:
  - El modelo propuesto obtuvo la mejor precisión promedio (**68%**).
  - La combinación de espectrogramas y LLDs permitió mejorar especialmente la clasificación de la emoción “relajado”, que era la más difícil para los modelos simples.

**Comentario técnico:**  
El uso conjunto de características 2D (espectrogramas) y 1D (LLDs) permite capturar tanto la estructura temporal como las propiedades estadísticas del audio. La arquitectura CNN-LSTM extrae patrones locales y secuenciales, mientras que la DNN sintetiza información global, logrando una representación más robusta.


### 4.4 Experimento de clasificación de letras

Se evaluaron distintos modelos de clasificación sobre las características extraídas del texto de las letras:

- **Modelos comparados**:
  - CNN sobre Word2vec
  - LSTM sobre Word2vec
  - CNN-LSTM combinado
  - Modelo propuesto (CNN-LSTM + DNN con Word2vec + chi-cuadrado)

- **Resultados**:
  - El modelo propuesto obtuvo la mejor precisión promedio (**74%**).
  - La combinación de embeddings y vectores estadísticos permitió mejorar la clasificación de emociones como “triste” y “feliz”.

**Comentario técnico:**  
El uso conjunto de representaciones distribucionales (Word2vec) y estadísticas (chi-cuadrado) permite capturar tanto el contexto semántico como la frecuencia discriminativa de palabras clave. La arquitectura CNN-LSTM extrae patrones secuenciales, mientras que la DNN sintetiza correlaciones globales entre términos.

### 4.5 Experimento de fusión multimodal

Se compararon tres enfoques para integrar las modalidades de audio y letras:

- **Fusión de características**:  
  Se concatenaron los vectores de características de ambas modalidades, con normalización previa.  
  → Precisión promedio: **72.4%**

- **Fusión de decisiones**:  
  Se combinaron las probabilidades de salida de los clasificadores base mediante votación lineal.  
  → Precisión promedio: **74.8%**

- **Fusión por stacking (propuesta)**:  
  Se entrenó un subclasificador sobre las salidas softmax de los modelos unimodales.  
  → Precisión promedio: **78.2%**

**Comentario técnico:**  
La fusión por stacking supera a los métodos tradicionales al evitar la pérdida de información por normalización o reducción dimensional. Al aprender sobre las decisiones de cada modalidad, el subclasificador puede detectar patrones complementarios y corregir errores, logrando una integración más efectiva.

### 4.6 Experimento comparativo

Se comparó el rendimiento del modelo propuesto con otros enfoques publicados en años recientes, tanto unimodales como multimodales:

- **Modelos unimodales**:
  - LLDs + SVM (audio): 57.1%
  - MIDI + RNN (audio): 56.8%
  - TF-IDF (letras): 62.2%
  - Word2vec + LSTM (letras): 69.3%

- **Modelos multimodales**:
  - Random Forest con fusión de características: 73.8%
  - Fusión de decisiones con LFSM: 75.8%
  - Fusión por voto a nivel de oración: 80.6%
  - **Modelo propuesto (stacking + multifeatures)**: **78.2%**

**Comentario técnico:**  
El modelo propuesto logra una precisión competitiva frente a métodos multimodales más complejos. Su ventaja radica en la integración eficiente de características heterogéneas mediante stacking, sin necesidad de normalización cruzada ni estructuras de fusión manuales.


---
## 5. Conclusiones

El estudio propone un sistema de clasificación emocional musical multimodal que combina audio y letras mediante un modelo de red híbrida y una estrategia de ensamblado por stacking. Los principales aportes son:

- Un clasificador combinado que integra características 2D (espectrogramas, Word2vec) y 1D (LLDs, chi-cuadrado) mediante una arquitectura CNN-LSTM + DNN.
- Un método de fusión multimodal basado en stacking, que evita la pérdida de información por normalización cruzada y mejora la precisión frente a métodos tradicionales.
- Un esquema de preprocesamiento de audio que optimiza la segmentación y separación de voz para mejorar la calidad de las características.

El modelo alcanzó una precisión promedio del **78.2%** en clasificación emocional multimodal, superando enfoques unimodales y otros métodos de fusión. Se plantea como una solución eficaz y escalable para tareas de MIR con datos heterogéneos.

**Comentario técnico:**  
La arquitectura modular y el enfoque por stacking permiten adaptar el sistema a distintas combinaciones de características y modalidades. Esto lo convierte en una base sólida para proyectos que exploren clasificación emocional, recomendación musical o análisis afectivo en entornos reales.


## 6. Puntos clave para incorporar en el proyecto

### 6.1 Construcción del dataset

El artículo no proporciona el dataset final, pero describe un flujo replicable:

- **Fuente base**: Million Song Dataset + etiquetas de Last.fm.
- **Etiquetas emocionales**: angry, happy, relaxed, sad.
- **Curación**: 500 canciones por emoción (2000 en total).
- **Audio**: descargado manualmente; se segmenta en clips de 15s.
- **Letras**: extraídas por canción; no se especifica sincronización con el audio.
- **Separación de voz/fondo**: no se detalla el método, pero puede replicarse con herramientas como Spleeter.

**Recursos útiles para replicación**:
- [Subset del Million Song Dataset en Hugging Face](https://huggingface.co/datasets/trojblue/million-song-subset)  
- [Repositorio extendido con etiquetas y audio en GitHub](https://github.com/slettner/lastfm-spotify-tags-sim-userdata)  
- [Página de etiquetas de canciones en Last.fm](https://www.last.fm/tag/)  
- [Herramienta de separación de fuentes: Spleeter (Deezer)](https://github.com/deezer/spleeter)

**Flujo sugerido para replicación**:
1. Extraer listas de canciones etiquetadas desde Last.fm (API o scraping).
2. Cruzar con MSD para obtener metadatos y audio.
3. Descargar letras desde LyricWiki, Genius o Musixmatch.
4. Separar voz/fondo si se desea replicar esa parte.
5. Balancear clases y construir el dataset final.

---

### 6.2 Aspectos técnicos a adaptar del artículo

- **Fusión por stacking**: entrenar clasificadores unimodales y combinar sus salidas softmax en un subclasificador.
- **Arquitectura híbrida**:
  - Audio: CNN-LSTM + DNN sobre espectrogramas y LLDs.
  - Letras: CNN-LSTM + DNN sobre Word2vec y chi-cuadrado.
- **Preprocesamiento de audio**: segmentación en 15s, separación de voz si se desea.
- **Embeddings modernos**: reemplazar Word2vec por BERT, DistilBERT o similares.
- **Reducción de combinaciones**: limitar a 2 variantes de audio para simplificar el pipeline.

**Comentario técnico:**  
Este enfoque permite mantener la esencia multimodal del artículo, adaptando su complejidad a los recursos disponibles. La modularidad del sistema facilita la experimentación con distintas arquitecturas, embeddings y estrategias de fusión, sin comprometer la validez metodológica.
