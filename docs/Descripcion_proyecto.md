# Descripción Compacta del Proyecto — Sistema Multimodal para Clasificación de Emociones en Música

**Título : Análisis de emoción y sentimiento en música desde una perspectiva multimodal**


## 1. Enfoque general del proyecto
El proyecto desarrolla un sistema multimodal para clasificar emociones musicales combinando **audio** y **letras**. La metodología sigue un pipeline moderno que integra representaciones 2D y 1D por modalidad, modelos unimodales independientes y una fusión final mediante **stacking**.

El diseño está inspirado principalmente en el artículo:
- *“A Multimodal Music Emotion Classification Method Based on Multifeature Combined Network Classifier”*,  
y complementado por ideas del *explicit sparse attention* y *votación por bloques* de otro trabajo relevante, así como etiqutado de emociones con el modelo Arousal–Valence.

## 2. Estructura conceptual del pipeline

El sistema sigue un flujo multimodal compuesto por cinco etapas esenciales:

1. **Preprocesamiento**
   - Segmentación del audio en fragmentos de 15 s.
   - Separación opcional en pistas: original, instrumental, vocal.
   - Limpieza y normalización de letras.
   - Emparejamiento audio–texto vía metadatos.
   - Etiquetado emocional usando el modelo Arousal–Valence.

2. **Extracción de características**
   - **Audio 2D:** espectrogramas (STFT).  
   - **Audio 1D:** LLDs y estadísticas (HSFs).  
   - **Texto 2D:** secuencia de embeddings (Word2Vec o BERT).  
   - **Texto 1D:** TF–IDF y Chi-cuadrado.

3. **Modelado unimodal**
   - **Audio:** CNN–BiLSTM para 2D + DNN para 1D → softmax acústico.  
   - **Texto:** CNN–BiLSTM/BERT para 2D + DNN para 1D → softmax textual.

4. **Fusión multimodal (stacking)**
   - Concatenación de las salidas softmax unimodales.  
   - Entrenamiento de un subclasificador supervisado.  
   - Fusión clasificada como **late fusion mejorada**.

5. **Predicción final**
   - Softmax del subclasificador = emoción estimada.  
   - Opcional: votación por bloques si hay múltiples segmentos por canción.

## 3. Innovaciones clave del proyecto

El proyecto incorpora mejoras respecto al enfoque original del Artículo 1, basadas en
ideas complementarias de otros trabajos recientes en MIR y aprendizaje profundo.

### 3.1 Uso de modelos modernos de lenguaje (BERT)
Además de Word2Vec, se integra **BERT** para generar embeddings textuales
contextualizados, lo que permite capturar matices semánticos más finos y mejorar
la discriminación emocional proveniente de las letras.

### 3.2 Evaluación de variantes acústicas
Siguiendo el Artículo 2, se consideran tres configuraciones del audio:
- pista original,
- pista instrumental,
- pista vocal.

Esta comparación busca determinar qué componente aporta más información emocional
al clasificador acústico.

### 3.3 Atención dispersa (opcional)
Se planea evaluar la utilidad de la **explicit sparse attention**, un mecanismo que
prioriza características relevantes y reduce ruido en la representación del audio.

### 3.4 Votación por bloques (opcional)
Si se trabaja con varios segmentos por canción, puede aplicarse un esquema de votación
mayoritaria para obtener la emoción global de una obra. Esta técnica ayuda a estabilizar
la predicción a nivel de canción completa.

### 3.5 Fusión supervisada mediante *stacking*
La técnica central del sistema es la **late fusion mejorada** implementada con
*stacking*, que supera fusiones tempranas y votaciones simples al aprender relaciones
complementarias entre las modalidades acústica y textual.

## 4. Guía para iniciar la fase de modelado

La fase de modelado se construirá siguiendo un orden lógico que garantiza coherencia,
modularidad y facilidad para depurar resultados. Los pasos esenciales para comenzar
son los siguientes:

### 4.1 Definir las entradas por modalidad
Antes de entrenar cualquier modelo, se deben tener claros los formatos de entrada:

- **Audio 2D:** espectrogramas normalizados con forma consistente (H × W).  
- **Audio 1D:** vectores de HSFs con dimensión fija.  
- **Texto 2D:** matriz de embeddings (Word2Vec o BERT).  
- **Texto 1D:** vectores TF–IDF o Chi-cuadrado reducidos a una dimensión manejable.

Cada entrada debe tener un *shape* estable antes de construir las arquitecturas.

---

### 4.2 Construir y probar los modelos unimodales por separado
Iniciar siempre con modelos independientes:

1. **Modelo de audio (unimodal):**
   - CNN–BiLSTM → espectrogramas  
   - DNN → HSFs  
   - Fusión interna → softmax final  

2. **Modelo de texto (unimodal):**
   - CNN–BiLSTM o BERT → embeddings  
   - DNN → TF–IDF/CHI  
   - Fusión interna → softmax final  

Probar cada modelo por separado permite:
- validar dimensiones,
- verificar aprendizaje básico,
- comparar aportes acústicos vs. semánticos.

---

### 4.3 Generar y almacenar las predicciones unimodales
Cada modelo debe generar un vector **softmax** por segmento.

Estos vectores se guardan como:
```bash
pred_audio[segment_id] = [...]
pred_text[segment_id] = [...]
```


Estos serán la entrada del **stacking**, así que deben guardarse de forma ordenada y reproducible.

---

### 4.4 Entrenar el subclasificador para la fusión multimodal
Una vez obtenidas las salidas unimodales:

- concatenar softmax_audio + softmax_text  
- entrenar un submodelo (p. ej. DNN ligera o Logistic Regression)  
- validar mediante cross-validation  
- evaluar contra los modelos unimodales

Este paso produce la **predicción multimodal final**.

---

### 4.5 Implementar evaluación coherente
El sistema debe evaluarse de manera consistente:

- accuracy, F1 o métricas acordadas  
- matriz de confusión por modalidad  
- comparación: audio vs. texto vs. multimodal  
- análisis por clase (happy/sad/angry/relaxed)

Esto cierra la primera fase del modelado.

---

### 4.6 Opcionales si hay tiempo
Estos componentes pueden integrarse después:

- **Sparse attention** (Artículo 2)  
- **Votación por bloques** (si hay varios segmentos por canción)  
- **Ablation study**:  
  - espectrogramas vs. HSFs  
  - Word2Vec vs. BERT  
  - instrumental vs. original vs. vocal  

---

## Conclusión operativa

Para arrancar el modelado solo necesitas:

1. Inputs estables (audio/texto, 2D/1D).  
2. Implementar primero los modelos unimodales.  
3. Generar sus softmax.  
4. Entrenar el stacking.  
5. Evaluar.  

Con esto, el sistema multimodal ya estará funcional y listo para enriquecer.
