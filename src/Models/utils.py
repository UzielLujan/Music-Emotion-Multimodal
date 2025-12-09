
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.feature_selection import SelectKBest, chi2  # <--- NUEVO IMPORT

class MultimodalDataset(Dataset):
    def __init__(self, master_csv_path, embeddings_path, mode='train', specific_text_cols=None):
        """
        Dataset que maneja Audio (2D+1D) y Texto (2D+1D) simultáneamente.
        Args:
            master_csv_path: Ruta al archivo master_dataset.csv
            embeddings_path: Ruta al archivo embeddings_2d.npy
            mode: 'train', 'val', o 'test'
            specific_text_cols: (Lista opcional) Lista de nombres de columnas TF-IDF seleccionadas
                                por Chi^2. Si es None, usa la lógica original (todo).
        """
        # 1. Cargar CSV Maestro
        # Ajusta la ruta base si tus datos están en otro lado, por defecto data/processed
        df_full = pd.read_csv(master_csv_path)
        
        # Filtrar por split (train / val / test)
        self.df = df_full[df_full['split'] == mode].reset_index(drop=True)
        
        # 2. Cargar la Matriz Gigante de Embeddings (Texto 2D) en memoria
        # Si tienes problemas de RAM, avísame para cambiarlo a modo lectura en disco (mmap)
        # Se agrega validación por si el archivo no existe (para pruebas rápidas)
        if Path(embeddings_path).exists():
            self.all_embeddings = np.load(embeddings_path, mmap_mode='r') # mmap para ahorrar RAM
        else:
            print(f"⚠️ {mode.upper()}: No se encontró {embeddings_path}. Se usarán ceros para BERT.")
            self.all_embeddings = None
        
        # 3. COLUMNAS DE AUDIO 1D (EXPLICITAMENTE DEFINIDAS)
        # Lista exacta proporcionada por el usuario
        self.audio_cols = [
            'mfcc_1_mean', 'mfcc_1_std', 'mfcc_2_mean', 'mfcc_2_std',
            'mfcc_3_mean', 'mfcc_3_std', 'mfcc_4_mean', 'mfcc_4_std',
            'mfcc_5_mean', 'mfcc_5_std', 'mfcc_6_mean', 'mfcc_6_std',
            'mfcc_7_mean', 'mfcc_7_std', 'mfcc_8_mean', 'mfcc_8_std',
            'mfcc_9_mean', 'mfcc_9_std', 'mfcc_10_mean', 'mfcc_10_std',
            'mfcc_11_mean', 'mfcc_11_std', 'mfcc_12_mean', 'mfcc_12_std',
            'mfcc_13_mean', 'mfcc_13_std',
            'chroma_mean', 'chroma_std',
            'contrast_mean', 'contrast_std',
            'zcr_mean', 'zcr_std',
            'centroid_mean', 'centroid_std'
        ]
        
        # Validación de seguridad: Rellenar con 0.0 si falta alguna columna en el CSV
        missing_audio = [c for c in self.audio_cols if c not in self.df.columns]
        if missing_audio:
            if mode == 'train': 
                print(f"⚠️ Aviso: Faltan columnas de audio en el CSV, rellenando con 0: {missing_audio}")
            for c in missing_audio:
                self.df[c] = 0.0

        # 4. COLUMNAS DE TEXTO 1D (LOGICA CHI^2 O AUTO-DETECTAR)
        
        if specific_text_cols is not None:
            # CASO A: Usamos las columnas filtradas por Chi^2
            # Verificamos intersección para que no falle si falta alguna
            self.text_cols = [c for c in specific_text_cols if c in self.df.columns]
            if mode == 'train':
                print(f"   -> Usando {len(self.text_cols)} features de texto seleccionadas (Chi^2).")
        else:
            # CASO B: Lógica original (Todo lo que no es meta ni audio)
            cols_metadata = [
                'spotify_id', 'split', 'label_quadrant', 'embedding_idx', 
                'path_spectrogram', 'path_embedding', 'track_name', 'artist', 
                'valence', 'arousal'
            ]
            # Unimos metadata + audio para saber qué EXCLUIR
            cols_excluded = set(cols_metadata) | set(self.audio_cols)
            # Las columnas restantes son Texto 1D
            self.text_cols = [c for c in self.df.columns if c not in cols_excluded]
            self.text_cols.sort() # Ordenar para consistencia
            
            if mode == 'train':
                print(f"   -> Usando TODAS las features de texto disponibles: {len(self.text_cols)}")
        
        # Guardamos la dimensión para configurar la red neuronal más tarde
        self.text_1d_dim = len(self.text_cols)
        
        if self.text_1d_dim == 0 and mode == 'train':
            print(f"⚠️ Aviso: No se detectaron columnas TF-IDF. Se usará un vector de ceros (dim 100).")
            self.text_1d_dim = 100 # Dimensión default de seguridad

        # 5. Mapeo de Etiquetas (CORREGIDO)
        self.label_map = {
            'Q1_Happy': 0, 
            'Q2_Angry': 1, 
            'Q3_Sad': 2, 
            'Q4_Relaxed': 3
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # ==========================
        # A. PROCESAMIENTO DE AUDIO
        # ==========================
        
        # 1. Audio 2D: Espectrograma
        spec_path = Path("data/processed") / str(row['path_spectrogram'])
        # Ajuste de ruta si ya incluye 'processed'
        if not spec_path.exists():
             spec_path = Path("data") / str(row['path_spectrogram'])

        try:
            if spec_path.exists():
                spec = np.load(spec_path)
                # Convertir a Tensor y agregar dimensión de canal: (H, W) -> (1, H, W)
                # Verificamos si ya tiene canal
                if spec.ndim == 2:
                    spec = spec[np.newaxis, ...]
                spec = torch.tensor(spec, dtype=torch.float32)
            else:
                spec = torch.zeros((1, 128, 128), dtype=torch.float32)
        except Exception as e:
            # Fallback de seguridad
            # print(f"Error cargando {spec_path}: {e}")
            spec = torch.zeros((1, 128, 128), dtype=torch.float32)
        
        # 2. Audio 1D: Features Estadísticos
        audio_vals = row[self.audio_cols].values.astype(np.float32)
        audio_1d = torch.tensor(audio_vals, dtype=torch.float32)
        
        # ==========================
        # B. PROCESAMIENTO DE TEXTO
        # ==========================
        
        # 1. Texto 2D: Embeddings BERT
        # (Corrección de dimensión para Conv1d: quitamos unsqueeze(0))
        if self.all_embeddings is not None:
            idx_emb = int(row['embedding_idx'])
            text_emb = self.all_embeddings[idx_emb] # Shape: (Seq, 768)
            # Transponer para Conv1d: (Seq, Channels) -> (Channels, Seq) -> (768, 256)
            text_2d = torch.tensor(text_emb, dtype=torch.float32).transpose(0, 1)
        else:
            text_2d = torch.zeros((768, 256), dtype=torch.float32)
        
        # 2. Texto 1D: TF-IDF (Filtrado o Completo)
        if len(self.text_cols) > 0:
            text_vals = row[self.text_cols].values.astype(np.float32)
            text_1d = torch.tensor(text_vals, dtype=torch.float32)
        else:
            # Vector de ceros si no hay datos
            text_1d = torch.zeros(self.text_1d_dim, dtype=torch.float32)
        
        # ==========================
        # C. ETIQUETA (TARGET)
        # ==========================
        label_str = str(row['label_quadrant'])
        # .get() devuelve 0 por defecto si hay algún error en el string
        label = torch.tensor(self.label_map.get(label_str, 0), dtype=torch.long)
        
        return {
            'spotify_id': self.df.iloc[idx]['spotify_id'], 
            'audio_2d': spec,      # [1, 128, 128]
            'audio_1d': audio_1d,  # [34]
            'text_2d': text_2d,    # [768, Seq]
            'text_1d': text_1d,    # [N_TFIDF]
            'label': label         # Escalar
        }

def get_dataloaders(batch_size=16, use_chi2=True, k_features=500):
    """
    Función auxiliar para instanciar los 3 dataloaders automáticamente.
    Args:
        batch_size: Tamaño del lote.
        use_chi2: Si True, filtra las columnas TF-IDF usando Chi-cuadrado en el set de Train.
        k_features: Número de top palabras a mantener.
    Retorna: train_loader, val_loader, test_loader
    """
    base = Path("data/processed")
    csv_path = base / "master_dataset.csv"
    emb_path = base / "features_2d/embeddings/embeddings_2d.npy"
    
    # Verificación básica de archivos
    if not csv_path.exists():
        # Intento ruta Colab
        csv_path = Path("/content/drive/MyDrive/Proyecto_Final_MIR/data/processed/master_dataset.csv")
    
    if not csv_path.exists():
        print(f"❌ ERROR CRÍTICO: No se encontró {csv_path}")
        print("   Ejecuta primero 'create_master_dataset.py'")
        return None, None, None
        
    if not emb_path.exists():
        # Intento ruta Colab
        emb_path = Path("/content/drive/MyDrive/Proyecto_Final_MIR/data/processed/features_2d/embeddings/embeddings_2d.npy")
        if not emb_path.exists():
             print(f"⚠️ Advertencia: No se encontró {emb_path}, se cargará dummy en el dataset.")

    print(f"🚀 Creando DataLoaders (Batch: {batch_size}) | Chi2: {use_chi2}")
    
    # --- LOGICA CHI-CUADRADO ---
    selected_cols = None
    
    if use_chi2:
        print("🔍 Ejecutando selección de características (Chi^2) sobre TRAIN...")
        
        # 1. Leer CSV completo para análisis
        df_temp = pd.read_csv(csv_path)
        df_train = df_temp[df_temp['split'] == 'train'].copy()
        
        # 2. Identificar candidatos (Columnas que empiecen con 'tfidf_')
        # Es la forma más segura de distinguir texto de metadatos numéricos
        tfidf_candidates = [c for c in df_train.columns if c.startswith('tfidf_')]
        
        if len(tfidf_candidates) == 0:
            print("⚠️ No se encontraron columnas 'tfidf_', omitiendo selección.")
        else:
            # 3. Preparar X e y
            label_map = {'Q1_Happy': 0, 'Q2_Angry': 1, 'Q3_Sad': 2, 'Q4_Relaxed': 3}
            
            # Limpiar datos
            df_train = df_train.dropna(subset=['label_quadrant'])
            X_train = df_train[tfidf_candidates].fillna(0)
            y_train = df_train['label_quadrant'].map(label_map).fillna(0)
            
            # 4. Ajustar SelectKBest
            k = min(k_features, len(tfidf_candidates))
            selector = SelectKBest(chi2, k=k)
            selector.fit(X_train, y_train)
            
            # 5. Obtener nombres de columnas
            selected_cols = list(selector.get_feature_names_out(input_features=tfidf_candidates))
            print(f"✅ Selección completada: {len(tfidf_candidates)} -> {len(selected_cols)} features más relevantes.")

    # --- INSTANCIAR DATASETS ---
    # Pasamos 'selected_cols' a todos. Si es None, usan todo.
    # Si tiene lista, Val y Test usarán SOLO esas columnas (evitando data leakage).
    
    train_ds = MultimodalDataset(csv_path, emb_path, mode='train', specific_text_cols=selected_cols)
    val_ds = MultimodalDataset(csv_path, emb_path, mode='val', specific_text_cols=selected_cols)
    test_ds = MultimodalDataset(csv_path, emb_path, mode='test', specific_text_cols=selected_cols)
    
    # Creamos los loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"   ✅ Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    
    return train_loader, val_loader, test_loader