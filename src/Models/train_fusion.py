# ==========================================================
# train_fusion.py — Multimodal Stacking (Audio + Texto)
# ==========================================================

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------
# get_project_paths (misma versión robusta que en meta_features)
# ----------------------------------------------------------
def get_project_paths():
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "Proyecto Final MIR").exists():
            root = parent / "Proyecto Final MIR"
            break
    else:
        drive = Path("/content/drive/MyDrive")
        if (drive / "Proyecto Final MIR").exists():
            root = drive / "Proyecto Final MIR"
        else:
            raise FileNotFoundError("❌ No se encontró la carpeta del proyecto.")

    return {
        "ROOT": root,
        "REPORTS": root / "Reports",
        "AUDIO_PRED": root / "Reports" / "audio_expert",
        "TEXT_PRED": root / "Reports" / "text_expert",
        "SAVED": root / "src" / "saved_models",
        "MODELS": root / "src" / "Models",
    }


# ----------------------------------------------------------
# Cargar meta-features
# ----------------------------------------------------------
def load_and_merge(split, paths):
    df_a = pd.read_csv(paths["AUDIO_PRED"] / f"predicciones_audio_{split}.csv")
    df_t = pd.read_csv(paths["TEXT_PRED"] / f"predicciones_text_{split}.csv")

    df = df_a.merge(df_t, on="spotify_id")

    X_a = df[[c for c in df.columns if c.startswith("prob_audio_")]].to_numpy(np.float32)
    X_t = df[[c for c in df.columns if c.startswith("prob_text_")]].to_numpy(np.float32)

    X = np.concatenate([X_a, X_t], axis=1)
    y = df["true_label"].to_numpy(np.int64)

    return X, y, df["spotify_id"].tolist()


# ----------------------------------------------------------
# Subclasificador
# ----------------------------------------------------------
class FusionNet(nn.Module):
    def __init__(self, in_features=8, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


# ----------------------------------------------------------
# Train / Eval
# ----------------------------------------------------------
def run_epoch(model, loader, criterion, optim=None, device="cpu"):
    if optim:
        model.train()
    else:
        model.eval()

    total_loss, total_correct, total_samples = 0,0,0

    with torch.set_grad_enabled(optim is not None):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            if optim:
                optim.zero_grad()
                loss.backward()
                optim.step()

            preds = logits.argmax(1)
            total_correct += (preds == yb).sum().item()
            total_loss += loss.item() * yb.size(0)
            total_samples += yb.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def train_fusion(epochs=40, batch_size=128, lr=1e-3, device=None):
    paths = get_project_paths()

    # Agregar rutas
    sys.path.append(str(paths["ROOT"] / "src"))
    sys.path.append(str(paths["MODELS"]))

    # Cargar features
    X_train, y_train, _ = load_and_merge("TRAIN", paths)
    X_val, y_val, _     = load_and_merge("VAL", paths)
    X_test, y_test, _   = load_and_merge("TEST", paths)

    num_classes = len(np.unique(y_train))
    print("✔ Clases detectadas:", num_classes)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FusionNet(in_features=8, num_classes=num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=batch_size)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val)),   batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test)),  batch_size=batch_size)

    best_val = 0
    best_state = None

    for ep in range(1, epochs+1):
        tl, ta = run_epoch(model, train_loader, crit, optim, device)
        vl, va = run_epoch(model, val_loader,   crit, None, device)

        if va > best_val:
            best_val = va
            best_state = model.state_dict()

        print(f"Ep {ep:03d} | Train Acc {ta:.3f} | Val Acc {va:.3f}")

    model.load_state_dict(best_state)

    tl, ta = run_epoch(model, train_loader, crit, None, device)
    vl, va = run_epoch(model, val_loader, crit, None, device)
    te, teacc = run_epoch(model, test_loader, crit, None, device)

    print("\nResultados finales")
    print("Train:", ta)
    print("Val:  ", va)
    print("Test:", teacc)

    save_path = paths["SAVED"] / "fusion_best.pth"
    torch.save(model.state_dict(), save_path)
    print("Modelo guardado en", save_path)


# ----------------------------------------------------------
# ENTRENAMIENTO XGBOOST SIMPLE
# ----------------------------------------------------------
def train_fusion_xgb_simple():
    paths = get_project_paths()

    print("📁 Proyecto:", paths["ROOT"])

    # === 1. Cargar datos ===
    X_train, y_train, _ = load_and_merge("TRAIN", paths)
    X_val, y_val, _     = load_and_merge("VAL",   paths)
    X_test, y_test, ids_test = load_and_merge("TEST", paths)

    # Concatenar TRAIN + VAL
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    print("✔ X_full:", X_full.shape, "| y_full:", y_full.shape)

    # === 2. Modelo XGBoost ===
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
    )

    print("🚀 Entrenando XGBoost Fusion Model...")
    xgb.fit(X_full, y_full)

    # === 3. Evaluar en TEST ===
    y_pred = xgb.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print("\n🎯 TEST Accuracy:", round(test_acc, 4))

    report = classification_report(y_test, y_pred, digits=4)
    print("\n📊 CLASSIFICATION REPORT:\n", report)

    # === 4. Guardar reportes ===
    out_dir = paths["REPORTS"] / "fusion_xgboost_simple"
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    quadrant_names = ["Q1_Happy", "Q2_Angry", "Q3_Sad", "Q4_Relaxed"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=quadrant_names,
                yticklabels=quadrant_names,
                cmap="Blues")
    plt.title("Matriz de Confusión — XGBoost Fusion")
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # === 5. Guardar modelo ===
    model_path = paths["SAVED"] / "fusion_xgb_simple.json"
    xgb.save_model(model_path)
    print("📁 Modelo guardado en:", model_path)


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
if __name__ == "__main__":
    train_fusion()
    train_fusion_xgb_simple()