# ===============================================================
# generate_meta_features.py
# ===============================================================
# Genera predicciones de los expertos de Audio y Texto (1D+2D)
# para los splits TRAIN / VAL / TEST.
# Crea los CSV necesarios para train_fusion.py
# ===============================================================

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import sys

from definitions import AudioNetwork, TextNetwork
from utils import get_dataloaders


# ---------------------------------------------------------------
# 1. Obtener rutas del proyecto (Robusto para LOCAL + COLAB)
# ---------------------------------------------------------------
def get_project_paths():
    """
    Detecta automáticamente la carpeta raíz del proyecto.
    Funciona tanto en LOCAL como en COLAB/Drive.
    """

    cwd = Path.cwd()

    # Caso 1: local — buscar carpeta "Proyecto Final MIR"
    for parent in [cwd] + list(cwd.parents):
        if (parent / "Proyecto Final MIR").exists():
            root = parent / "Proyecto Final MIR"
            break
    else:
        # Caso 2: Colab — buscar en drive
        drive_base = Path("/content/drive/MyDrive")
        if (drive_base / "Proyecto Final MIR").exists():
            root = drive_base / "Proyecto Final MIR"
        else:
            raise FileNotFoundError(
                "❌ No se encontró la carpeta 'Proyecto Final MIR' en local ni Colab."
            )

    paths = {
        "ROOT": root,
        "REPORTS": root / "Reports",
        "AUDIO_PRED": root / "Reports" / "audio_expert",
        "TEXT_PRED": root / "Reports" / "text_expert",
        "SAVED": root / "src" / "saved_models",
        "MODELS": root / "src" / "Models",
        "DATA": root / "data",
    }

    return paths


# ---------------------------------------------------------------
# 2. Inferir sobre Un Experto
# ---------------------------------------------------------------
def infer_expert(model, loader, device="cpu", prefix="audio"):
    """
    Devuelve un DataFrame con:
    spotify_id, prob_Q1..prob_Q4, true_label
    """
    model.eval()

    ids = []
    probs = []
    labels = []

    with torch.no_grad():
        for batch in loader:

            spotify_ids = batch.get("spotify_id")
            ids.extend(spotify_ids)

            if prefix == "audio":
                x2d = batch["audio_2d"].to(device)
                x1d = batch["audio_1d"].to(device)
                logits = model(x2d, x1d)

            else:  # texto
                x2d = batch["text_2d"].to(device)
                x1d = batch["text_1d"].to(device)
                logits = model(x2d, x1d)

            pr = torch.softmax(logits, dim=1).cpu().numpy()
            probs.extend(pr)

            labels.extend(batch["label"].cpu().numpy())

    df = pd.DataFrame(
        {
            "spotify_id": ids,
            f"prob_{prefix}_Q1": np.array(probs)[:, 0],
            f"prob_{prefix}_Q2": np.array(probs)[:, 1],
            f"prob_{prefix}_Q3": np.array(probs)[:, 2],
            f"prob_{prefix}_Q4": np.array(probs)[:, 3],
            "true_label": labels,
        }
    )
    return df


# ---------------------------------------------------------------
# 3. Main — Generar Meta Features
# ---------------------------------------------------------------
def generate_meta_features(batch_size=32):
    paths = get_project_paths()

    print("\n🔍 Generando meta-features...")
    print(paths)

    # --- Asegurar que sys.path contiene src y Models ---
    sys.path.append(str(paths["ROOT"] / "src"))
    sys.path.append(str(paths["MODELS"]))

    # --- Cargar DataLoaders ---
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, use_chi2=True, k_features=500
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Dispositivo: {device}")

    # ---------------------------------------------------
    # Cargar modelos expertos (ARCHIVOS CORRECTOS)
    # ---------------------------------------------------
    audio_model_path = paths["SAVED"] / "audio_network_best.pth"
    text_model_path = paths["SAVED"] / "text_network_best.pth"

    print("\n📥 Cargando modelos expertos...")

    # AUDIO
    state = torch.load(audio_model_path, map_location=device)
    audio_sd = state["model_state_dict"] if "model_state_dict" in state else state
    audio_model = AudioNetwork(num_classes=4, audio_1d_dim=34).to(device)
    audio_model.load_state_dict(audio_sd)

    # TEXTO
    tstate = torch.load(text_model_path, map_location=device)
    text_sd = tstate["model_state_dict"] if "model_state_dict" in tstate else tstate
    text_1d_dim = tstate.get("text_1d_dim", 500) if isinstance(tstate, dict) else 500
    text_model = TextNetwork(num_classes=4, text_1d_dim=text_1d_dim).to(device)
    text_model.load_state_dict(text_sd)

    print("✔ Modelos cargados.")

    # ---------------------------------------------------
    # Inferir para TRAIN / VAL / TEST
    # ---------------------------------------------------
    splits = {
        "TRAIN": train_loader,
        "VAL": val_loader,
        "TEST": test_loader,
    }

    audio_out = paths["AUDIO_PRED"]
    text_out = paths["TEXT_PRED"]

    audio_out.mkdir(parents=True, exist_ok=True)
    text_out.mkdir(parents=True, exist_ok=True)

    for split, loader in splits.items():
        print(f"\n➡ Procesando {split}...")

        df_a = infer_expert(audio_model, loader, device=device, prefix="audio")
        df_t = infer_expert(text_model, loader, device=device, prefix="text")

        df_a.to_csv(audio_out / f"predicciones_audio_{split}.csv", index=False)
        df_t.to_csv(text_out / f"predicciones_text_{split}.csv", index=False)

        print(f"   ✔ {split}: CSVs generados.")

    print("\n🎉 ¡Meta-features creados exitosamente!")
