import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def generate_bert_embeddings(
        input_path: Path, 
        output_dir: Path, 
        model_name: str = "distilbert-base-uncased",
        max_length: int = 256,  # 256 es el balance ideal Memoria/Cobertura
        batch_size: int = 32,
        device_str: str = None
    ):
    """
    Genera embeddings 2D usando un modelo Transformer (BERT/DistilBERT).
    Diseñado para correr en GPU de forma eficiente y generar input para CNN.
    
    Salida: Tensor 3D (N_samples, Max_Length, Hidden_Dim) -> (N, 256, 768)
    """
    
    # 1. Configuración de Dispositivo
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 [Embeddings] Iniciando generación en: {device}")
    
    # 2. Cargar Datos
    if not input_path.exists():
        raise FileNotFoundError(f"❌ No se encuentra el archivo: {input_path}")

    print(f"[Embeddings] Leyendo: {input_path.name}")
    try:
        if input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)
    except Exception as e:
        raise ImportError(f"Error leyendo el archivo. Asegúrate de tener pyarrow/fastparquet instalado.\n{e}")

    # Validar columna de texto (Versión BERT conserva estructura)
    # ESTANDARIZACIÓN: Buscamos específicamente la columna limpia para BERT
    col_text = "clean_lyrics_bert"
    
    if col_text not in df.columns:
        # Fallback inteligente
        if 'lyrics' in df.columns:
            print(f"⚠️ Advertencia: No se halló '{col_text}', usando 'lyrics'.")
            col_text = 'lyrics'
        else:
            raise ValueError(f"Falta la columna '{col_text}'. Ejecuta cleaning.py primero.")

    print(f"[Embeddings] Usando columna de texto: '{col_text}'")

    # Convertir a lista (más rápido que iterar pandas)
    texts = df[col_text].fillna("").astype(str).tolist()
    
    # Manejo de IDs
    if "spotify_id" in df.columns:
        ids = df["spotify_id"].values
    elif "musicId" in df.columns:
        ids = df["musicId"].values
    else:
        print("Advertencia: No se encontró columna de ID. Usando índice numérico.")
        ids = df.index.values

    # 3. Cargar Modelo y Tokenizer
    print(f"[Embeddings] Cargando modelo: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    except OSError:
        raise ConnectionError("No se pudo descargar el modelo. Verifica tu internet.")

    model.eval() # Modo evaluación (apaga dropout, ahorra memoria)

    all_embeddings = []
    
    # 4. Procesamiento por Lotes
    print(f"[Embeddings] Procesando {len(texts)} canciones (Max Len: {max_length}, Batch: {batch_size})...")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generando Tensores"):
        batch_texts = texts[i : i + batch_size]
        
        # A. Tokenización
        encoded_input = tokenizer(
            batch_texts,
            padding="max_length", 
            truncation=True,      
            max_length=max_length,
            return_tensors="pt"
        )
        
        # B. Mover inputs a GPU
        input_ids = encoded_input["input_ids"].to(device)
        attention_mask = encoded_input["attention_mask"].to(device)
        
        # C. Inferencia
        with torch.no_grad(): # Desactivar gradientes (CRÍTICO para memoria)
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # last_hidden_state shape: (Batch, Seq_Len, Hidden_Dim)
            # Esto guarda TODA la secuencia (ideal para CNNs)
            sequence_output = outputs.last_hidden_state
            
            # D. Mover a CPU inmediatamente y convertir a numpy float32
            all_embeddings.append(sequence_output.cpu().numpy().astype(np.float32))
            
            # Limpieza explícita de caché de GPU
            del input_ids, attention_mask, outputs, sequence_output

    # 5. Concatenar y Guardar
    if all_embeddings:
        final_tensor = np.concatenate(all_embeddings, axis=0)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Rutas finales
        path_npy = output_dir / "embeddings_2d.npy"
        path_ids = output_dir / "embeddings_ids.npy"
        
        np.save(path_npy, final_tensor)
        np.save(path_ids, ids)
        
        print(f"[Embeddings] Guardado exitoso:")
        print(f"   Tensor: {path_npy} | Shape: {final_tensor.shape}")
        print(f"   IDs:    {path_ids}")
    else:
        print(" Error: No se generaron embeddings.")

if __name__ == "__main__":
    pass