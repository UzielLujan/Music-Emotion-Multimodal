import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path

class MultimodalDataset(Dataset):
    """
    Dataset Maestro para el Proyecto MIR (Music Emotion Recognition).
    Maneja 4 modalidades de entrada simultáneas:
    1. Audio 2D (Espectrogramas -> CNN)
    2. Texto 2D (Embeddings BERT -> CNN)
    3. Audio 1D (MFCCs -> MLP)
    4. Texto 1D (TF-IDF -> MLP)
    """
    
    def __init__(self, master_csv_path, split='train', root_dir=None):
        """
        Args:
            master_csv_path (str): Ruta al master_dataset.csv
            split (str): 'train', 'val', o 'test'.
            root_dir (str): Ruta base del proyecto (carpeta 'data').
        """
        self.split = split
        
        # 1. Configurar Ruta Raíz (data/)
        if root_dir is None:
            # Asume estructura: src/Loaders/dataset.py -> sube 2 niveles -> data/
            self.root_dir = Path(__file__).resolve().parents[2] / "data"
        else:
            self.root_dir = Path(root_dir)

        # 2. Cargar CSV y filtrar por split
        if not Path(master_csv_path).exists():
             raise FileNotFoundError(f"No se encuentra el CSV maestro: {master_csv_path}")

        full_df = pd.read_csv(master_csv_path)
        self.df = full_df[full_df['split'] == split].reset_index(drop=True)
        
        print(f"[{split.upper()}] Cargando {len(self.df)} muestras desde: {Path(master_csv_path).name}")

        # 3. Detectar columnas 1D automáticas
        self.audio_1d_cols = [c for c in self.df.columns if any(x in c for x in ['mfcc_', 'chroma_', 'contrast_', 'zcr_', 'centroid_'])]
        self.text_1d_cols = [c for c in self.df.columns if c.startswith('tfidf_')]
        
        # -----------------------------------------------------------
        # 4. CARGA DE EMBEDDINGS 2D (Corrección de Rutas)
        # -----------------------------------------------------------
        self.bert_matrix = None
        self.id_to_idx = {}
        
        # Estrategia de búsqueda de archivos:
        # El CSV tiene rutas tipo "features_2d/embeddings/...", pero el archivo físico
        # suele estar en "data/processed/features_2d/...". Probamos ambas.
        
        # Ruta base tentativa
        path_in_processed = self.root_dir / "processed" / "features_2d" / "embeddings"
        
        # Nombres exactos que confirmaste
        file_npy = "embeddings_2d.npy"
        file_ids = "embeddings_ids.npy"
        
        path_npy = path_in_processed / file_npy
        path_ids = path_in_processed / file_ids
        
        # DEBUG: Si falla, descomenta esto para ver dónde busca
        # print(f"DEBUG: Buscando embeddings en: {path_npy}")

        if path_npy.exists() and path_ids.exists():
            print(f"   ⚡ Cargando matriz BERT: {path_npy.name}")
            
            # Cargar matriz con memoria mapeada (no explota la RAM)
            self.bert_matrix = np.load(path_npy, mmap_mode='r') 
            
            # Cargar IDs y crear mapa
            self.bert_ids = np.load(path_ids, allow_pickle=True)
            self.id_to_idx = {str(uid): i for i, uid in enumerate(self.bert_ids)}
        else:
            print(f"   ⚠️ ALERTA: No se encontraron los archivos de embeddings.")
            print(f"      Ruta buscada: {path_npy}")
            print("      -> Se usarán ceros para la entrada de Texto 2D.")

        # 5. Mapeo de Etiquetas (Label Encoding)
        # Asumiendo cuadrantes Q1, Q2, Q3, Q4. Ajusta si tus etiquetas son diferentes.
        self.label_map = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        sid = str(row['spotify_id'])

        # --- A. FEATURES 1D (Tabular) ---
        audio_1d = torch.tensor(row[self.audio_1d_cols].values.astype(np.float32))
        text_1d  = torch.tensor(row[self.text_1d_cols].values.astype(np.float32))

        # --- B. FEATURES 2D (Tensores) ---
        
        # 1. Audio 2D (Espectrograma)
        # El CSV tiene rutas relativas, a veces falta "processed/"
        rel_spec = str(row['path_spectrogram'])
        spec_path = self.root_dir / "processed" / rel_spec if not rel_spec.startswith("processed") else self.root_dir / rel_spec
        
        # Fallback si la ruta en CSV ya incluía 'data/' o era absoluta (raro pero posible)
        if not spec_path.exists():
             spec_path = self.root_dir / rel_spec # Intento directo
        
        if spec_path.exists():
            try:
                spec = np.load(spec_path).astype(np.float32)
                if spec.ndim == 2: spec = spec[np.newaxis, ...] 
                spec_tensor = torch.from_numpy(spec)
            except:
                spec_tensor = torch.zeros((1, 128, 646)) # Fallback lectura corrupta
        else:
            spec_tensor = torch.zeros((1, 128, 646)) # Fallback no existe

        # 2. Texto 2D (Embedding BERT)
        if self.bert_matrix is not None and sid in self.id_to_idx:
            matrix_idx = self.id_to_idx[sid]
            # Extraer fila de la matriz gigante (256, 768)
            bert_data = self.bert_matrix[matrix_idx].astype(np.float32)
            # Agregar dimensión de canal -> (1, 256, 768)
            bert_tensor = torch.from_numpy(bert_data).unsqueeze(0) 
        else:
            bert_tensor = torch.zeros((1, 256, 768))

        # --- C. LABEL ---
        label_str = row['label_quadrant']
        label = self.label_map.get(label_str, 0) 
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            'id': sid,
            'audio_1d': audio_1d,
            'text_1d': text_1d,
            'audio_2d': spec_tensor,
            'text_2d': bert_tensor,
            'label': label_tensor
        }