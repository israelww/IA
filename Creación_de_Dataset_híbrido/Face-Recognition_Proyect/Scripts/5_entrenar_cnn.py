"""
5_entrenar_cnn.py
=================
Fase 5  Entrenamiento de CNN para Reconocimiento Facial Multiclase
Transfer learning con ResNet18 o EfficientNet-B0, entrenado sobre
imagenes preprocesadas de 160x160. HOLA.

Funcionalidades:
  - Dataset personalizado desde CSV (con correccion automatica de rutas)
  - Data augmentation solo en train
  - Manejo de desbalance con WeightedRandomSampler
  - Early stopping por val_loss
  - Guardado de best_model.pth, last_model.pth, history.json, metrics_test.json
  - Reporte de accuracy top-1 en train/val/test + matriz de confusion

Uso:
    python Scripts/5_entrenar_cnn.py
    python Scripts/5_entrenar_cnn.py --model_name efficientnet_b0 --epochs 50 --batch_size 64 --lr 0.0003

Autor: Generado automaticamente para el proyecto Face-Recognition Proyect
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Forzar UTF-8 en stdout/stderr para evitar errores de encoding en Windows (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

#  Rutas del proyecto 

PROJECT_ROOT = Path(__file__).resolve().parent.parent          # Face-Recognition Proyect/
SPLITS_DIR   = PROJECT_ROOT / "splits"
MODELS_DIR   = PROJECT_ROOT / "models"
CLASES_JSON  = SPLITS_DIR / "clases.json"

# Raiz del repositorio (un nivel arriba de Face-Recognition Proyect)
REPO_ROOT = PROJECT_ROOT.parent


# 
#  1) DATASET PERSONALIZADO
# 

class FaceDataset(Dataset):
    """
    Lee un CSV con columnas [ruta, clase, clase_idx, grupo] y carga
    las imagenes con un transform dado.

    Corrige automaticamente rutas absolutas viejas: busca la parte
    relativa a partir de 'Face-Recognition Proyect/' y la recompone
    usando la raiz real del proyecto actual.
    """

    ANCHOR = "Face-Recognition Proyect"

    def __init__(self, csv_path: str, transform=None, img_size: int = 160, grupos: list = None):
        self.df        = pd.read_csv(csv_path)
        self.transform = transform
        self.img_size  = img_size

        #  Filtrar por grupos si se especifica 
        if grupos:
            grupos_norm = [g.strip() for g in grupos]
            self.df = self.df[self.df["grupo"].isin(grupos_norm)].reset_index(drop=True)
            if len(self.df) == 0:
                disponibles = sorted(pd.read_csv(csv_path)["grupo"].unique())
                raise ValueError(
                    f"No hay imagenes para los grupos: {grupos_norm}.\n"
                    f"Grupos disponibles: {disponibles}"
                )

        #  Re-mapear clase_idx a indices consecutivos 0..N-1 
        # Necesario cuando se trabaja con un subconjunto de clases.
        clases_unicas     = sorted(self.df["clase"].unique())
        self.class_names  = clases_unicas
        self.clase_to_idx = {c: i for i, c in enumerate(clases_unicas)}
        self.df["clase_idx"] = self.df["clase"].map(self.clase_to_idx)
        self.num_classes  = len(clases_unicas)

        # Pre-corregir todas las rutas al construir el dataset
        self.df["ruta"] = self.df["ruta"].apply(self._normalizar_ruta)

        # Verificacion rapida: contar imagenes accesibles
        muestra = self.df["ruta"].head(20)
        accesibles = sum(1 for r in muestra if Path(r).exists())
        if accesibles == 0:
            print(f"   ADVERTENCIA: Ninguna de las primeras 20 rutas es accesible.")
            print(f"    Ejemplo: {muestra.iloc[0]}")
            print(f"    Revisa que Dataset_aumentado/ exista en {PROJECT_ROOT}")
        elif accesibles < len(muestra):
            print(f"   {len(muestra) - accesibles}/{len(muestra)} rutas inaccesibles en muestra.")

    def _normalizar_ruta(self, ruta_original: str) -> str:
        """
        Si la ruta contiene 'Face-Recognition Proyect', extrae la parte
        relativa desde ahi y la recompone con la raiz real del proyecto.
        Maneja tanto / como \\ como separadores.
        """
        ruta_str = str(ruta_original).replace("\\", "/")

        if self.ANCHOR in ruta_str:
            # Extraer la parte relativa a partir del anchor
            idx  = ruta_str.index(self.ANCHOR)
            rel  = ruta_str[idx + len(self.ANCHOR):].lstrip("/")
            ruta = PROJECT_ROOT / rel
            return str(ruta)

        # Si la ruta ya es relativa o no contiene el anchor, probar como esta
        if Path(ruta_original).exists():
            return str(ruta_original)

        # Ultimo recurso: tratar como relativa al repo
        return str(REPO_ROOT / ruta_original)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        ruta = row["ruta"]
        label = int(row["clase_idx"])

        try:
            img = Image.open(ruta).convert("RGB")
        except Exception as e:
            # Imagen corrupta o inaccesible: devolver tensor negro
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, label


# 
#  2) TRANSFORMS (DATA AUGMENTATION SOLO EN TRAIN)
# 

def get_transforms(img_size: int = 160):
    """
    Retorna transforms para train (con augmentation) y val/test (solo normalizacion).
    Normalizacion con media y std de ImageNet (requerido por modelos pretrained).
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, eval_transform


# 
#  3) MODELO CNN CON TRANSFER LEARNING
# 

def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Construye el modelo de transfer learning.
    Opciones: resnet18, efficientnet_b0
    """
    model_name = model_name.lower().strip()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        # Reemplazar la capa FC final
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )
        print(f"   Modelo: ResNet18 (fc: {in_features} -> 256 -> {num_classes})")

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )
        print(f"   Modelo: EfficientNet-B0 (classifier: {in_features} -> 256 -> {num_classes})")

    else:
        raise ValueError(
            f"Modelo '{model_name}' no soportado. Usa 'resnet18' o 'efficientnet_b0'."
        )

    return model


# 
#  4) WEIGHTED RANDOM SAMPLER PARA DESBALANCE
# 

def make_weighted_sampler(dataset: FaceDataset) -> WeightedRandomSampler:
    """
    Crea un WeightedRandomSampler que sobremuestrea las clases minoritarias.
    Cada muestra recibe un peso inversamente proporcional a la frecuencia de su clase.
    """
    labels = dataset.df["clase_idx"].values
    class_counts = Counter(labels)
    num_samples  = len(labels)

    # Peso por clase = total / (n_clases * conteo_clase)
    n_classes = len(class_counts)
    class_weights = {
        cls: num_samples / (n_classes * count)
        for cls, count in class_counts.items()
    }

    # Peso por muestra
    sample_weights = np.array([class_weights[label] for label in labels], dtype=np.float64)
    sample_weights = torch.from_numpy(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )

    # Reporte del desbalance
    min_cls = min(class_counts, key=class_counts.get)
    max_cls = max(class_counts, key=class_counts.get)
    ratio   = class_counts[max_cls] / class_counts[min_cls]
    print(f"   WeightedRandomSampler activo (ratio desbalance: {ratio:.1f}x)")
    print(f"    Clase mas grande: idx={max_cls} ({class_counts[max_cls]} imgs)")
    print(f"    Clase mas chica:  idx={min_cls} ({class_counts[min_cls]} imgs)")

    return sampler


# 
#  5) EARLY STOPPING
# 

class EarlyStopping:
    """
    Para el entrenamiento si val_loss no mejora en `patience` epocas consecutivas.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, verbose: bool = True):
        self.patience  = patience
        self.min_delta = min_delta
        self.verbose   = verbose
        self.counter   = 0
        self.best_loss = None
        self.triggered = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"     EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.triggered = True
                return True

        return False


# 
#  6) FUNCIONES DE ENTRENAMIENTO Y EVALUACION
# 

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Entrena una epoca y retorna (loss_promedio, accuracy)."""
    model.train()
    running_loss    = 0.0
    correct         = 0
    total           = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evalua el modelo y retorna (loss_promedio, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def get_all_predictions(model, loader, device):
    """Retorna (y_true, y_pred) para todo el dataloader."""
    model.eval()
    all_labels = []
    all_preds  = []

    for images, labels in loader:
        images  = images.to(device, non_blocking=True)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


# 
#  7) MATRIZ DE CONFUSION EN CONSOLA
# 

def print_confusion_matrix(y_true, y_pred, class_names):
    """Imprime la matriz de confusion formateada en consola."""
    cm = confusion_matrix(y_true, y_pred)
    n  = len(class_names)

    # Truncar nombres largos para que quepan
    max_len   = 18
    short     = [name[:max_len] for name in class_names]
    col_width = max(max_len, 5) + 1

    print(f"\n{'-'*70}")
    print("  MATRIZ DE CONFUSION (TEST)")
    print(f"{'-'*70}")

    # Encabezado: solo los indices para no ocupar tanto espacio
    header = f"{'':>{max_len}}  " + " ".join(f"{i:>4}" for i in range(n))
    print(header)
    print(f"{'' * max_len}" + "" * (5 * n))

    for i in range(n):
        row_str = " ".join(f"{cm[i, j]:>4}" for j in range(n))
        print(f"{short[i]:>{max_len}}  {row_str}")

    print(f"\n  Leyenda de indices:")
    for i, name in enumerate(class_names):
        print(f"    {i:>2}: {name}")
    print()


# 
#  8) FUNCION PRINCIPAL DE ENTRENAMIENTO
# 

def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "" * 70)
    print("  FASE 5  ENTRENAMIENTO DE CNN PARA RECONOCIMIENTO FACIAL")
    print("" * 70)

    #  Dispositivo 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"    GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("    CPU (sin GPU detectada  el entrenamiento sera mas lento)")

    #  Directorio de modelos (ruta unica del proyecto) 
    # Se guarda directamente en models/ para simplificar uso de scripts.
    run_models_dir = MODELS_DIR
    run_models_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Salida: {run_models_dir}")

    #  Hiperparametros 
    grupos_label = ", ".join(args.grupos) if args.grupos else "TODOS"
    print(f"\n{'-'*70}")
    print(f"  HIPERPARAMETROS")
    print(f"{'-'*70}")
    print(f"  Modelo:       {args.model_name}")
    print(f"  Grupos:       {grupos_label}")
    print(f"  Epocas:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Learning rate:{args.lr}")
    print(f"  Img size:     {args.img_size}x{args.img_size}")
    print(f"  Workers:      {args.num_workers}")
    print(f"  Dispositivo:  {device}")
    print(f"{'-'*70}")

    #  Transforms 
    train_transform, eval_transform = get_transforms(args.img_size)

    #  Datasets 
    print(f"\n  Cargando datasets...")
    grupos_arg = args.grupos if args.grupos else None
    train_ds = FaceDataset(str(SPLITS_DIR / "train.csv"), transform=train_transform, img_size=args.img_size, grupos=grupos_arg)
    val_ds   = FaceDataset(str(SPLITS_DIR / "val.csv"),   transform=eval_transform,  img_size=args.img_size, grupos=grupos_arg)
    test_ds  = FaceDataset(str(SPLITS_DIR / "test.csv"),  transform=eval_transform,  img_size=args.img_size, grupos=grupos_arg)

    # Obtener class_names y num_classes del dataset filtrado
    class_names = train_ds.class_names
    num_classes = train_ds.num_classes

    print(f"   Train: {len(train_ds):,} imagenes")
    print(f"   Val:   {len(val_ds):,} imagenes")
    print(f"   Test:  {len(test_ds):,} imagenes")
    print(f"   Clases ({num_classes}): {', '.join(class_names)}")

    #  WeightedRandomSampler para manejar desbalance 
    print(f"\n  Configurando balanceo de clases...")
    sampler = make_weighted_sampler(train_ds)

    #  DataLoaders 
    # Nota: shuffle=False porque el sampler ya se encarga de la aleatorizacion
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    #  Modelo 
    print(f"\n  Construyendo modelo...")
    model = build_model(args.model_name, num_classes, pretrained=True)
    model = model.to(device)

    total_params   = sum(p.numel() for p in model.parameters())
    trainable      = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parametros totales:     {total_params:>12,}")
    print(f"   Parametros entrenables: {trainable:>12,}")

    #  Loss, optimizer, scheduler 
    # Class weights adicionales en la loss (complementario al sampler)
    labels_train  = train_ds.df["clase_idx"].values
    class_counts  = np.bincount(labels_train, minlength=num_classes).astype(np.float64)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes  # normalizar
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    early_stopping = EarlyStopping(patience=12, min_delta=1e-4, verbose=True)

    #  Historial de entrenamiento 
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr":         [],
    }

    best_val_loss = float("inf")
    best_epoch    = 0

    # 
    #  BUCLE DE ENTRENAMIENTO
    # 
    print(f"\n{'-'*70}")
    print(f"  ENTRENAMIENTO")
    print(f"{'-'*70}\n")

    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr  = optimizer.param_groups[0]["lr"]

        #  Train 
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        #  Validate 
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        #  Scheduler step 
        scheduler.step()

        epoch_time = time.time() - epoch_start

        #  Logging 
        history["train_loss"].append(round(train_loss, 5))
        history["train_acc"].append(round(train_acc, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_acc"].append(round(val_acc, 5))
        history["lr"].append(round(current_lr, 8))

        marker = ""

        #  Guardar mejor modelo 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "val_loss":     val_loss,
                "val_acc":      val_acc,
                "model_name":   args.model_name,
                "num_classes":  num_classes,
                "img_size":     args.img_size,
                "class_names":  class_names,
                "grupos":       grupos_arg,
            }, run_models_dir / "best_model.pth")
            marker = "  BEST"

        #  Imprimir progreso 
        print(
            f"  Epoca {epoch:>3}/{args.epochs}    "
            f"train_loss: {train_loss:.4f}  acc: {train_acc:.4f}    "
            f"val_loss: {val_loss:.4f}  acc: {val_acc:.4f}    "
            f"lr: {current_lr:.2e}    "
            f"{epoch_time:.1f}s{marker}"
        )

        #  Early stopping 
        if early_stopping(val_loss):
            print(f"\n   Early stopping en epoca {epoch} (mejor: epoca {best_epoch})")
            break

    total_time = time.time() - total_start

    #  Guardar ultimo modelo 
    torch.save({
        "epoch":        epoch,
        "model_state":  model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "val_loss":     val_loss,
        "val_acc":      val_acc,
        "model_name":   args.model_name,
        "num_classes":  num_classes,
        "img_size":     args.img_size,
        "class_names":  class_names,
        "grupos":       grupos_arg,
    }, run_models_dir / "last_model.pth")

    #  Guardar historial 
    history["config"] = {
        "model_name":  args.model_name,
        "grupos":      grupos_arg,
        "epochs_run":  epoch,
        "batch_size":  args.batch_size,
        "lr":          args.lr,
        "img_size":    args.img_size,
        "num_classes": num_classes,
        "class_names": class_names,
        "best_epoch":  best_epoch,
        "best_val_loss": round(best_val_loss, 5),
        "total_time_s":  round(total_time, 1),
        "device":      str(device),
        "timestamp":   timestamp,
    }

    with open(run_models_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\n   best_model.pth   epoca {best_epoch} (val_loss={best_val_loss:.4f})")
    print(f"   last_model.pth   epoca {epoch}")
    print(f"   history.json guardado")
    print(f"    Tiempo total: {total_time/60:.1f} min")

    # 
    #  EVALUACION EN TEST CON EL MEJOR MODELO
    # 
    print(f"\n{'-'*70}")
    print(f"  EVALUACION FINAL (best_model.pth)")
    print(f"{'-'*70}")

    # Cargar best model
    checkpoint = torch.load(run_models_dir / "best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])

    #  Accuracy en cada split 
    print(f"\n  Calculando accuracy en cada split...")

    _, train_acc_final = evaluate(model, train_loader, criterion, device)
    _, val_acc_final   = evaluate(model, val_loader,   criterion, device)

    y_true_test, y_pred_test = get_all_predictions(model, test_loader, device)
    test_acc_final = accuracy_score(y_true_test, y_pred_test)

    print(f"\n  {'Split':<10}  {'Accuracy Top-1':>14}")
    print(f"  {'-'*28}")
    print(f"  {'Train':<10}  {train_acc_final:>13.4f}")
    print(f"  {'Val':<10}  {val_acc_final:>13.4f}")
    print(f"  {'Test':<10}  {test_acc_final:>13.4f}")

    #  Metricas detalladas en test 
    test_precision = precision_score(y_true_test, y_pred_test, average="weighted", zero_division=0)
    test_recall    = recall_score(y_true_test, y_pred_test, average="weighted", zero_division=0)
    test_f1        = f1_score(y_true_test, y_pred_test, average="weighted", zero_division=0)

    print(f"\n  Metricas globales en TEST (weighted avg):")
    print(f"    Precision: {test_precision:.4f}")
    print(f"    Recall:    {test_recall:.4f}")
    print(f"    F1-score:  {test_f1:.4f}")

    #  Classification report 
    print(f"\n  Classification Report (TEST):")
    print(classification_report(y_true_test, y_pred_test, target_names=class_names, zero_division=0))

    #  Matriz de confusion 
    print_confusion_matrix(y_true_test, y_pred_test, class_names)

    #  Guardar metricas de test 
    cm = confusion_matrix(y_true_test, y_pred_test).tolist()

    per_class_report = classification_report(
        y_true_test, y_pred_test,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics_test = {
        "timestamp":       timestamp,
        "model_name":      args.model_name,
        "best_epoch":      best_epoch,
        "accuracy": {
            "train": round(train_acc_final, 5),
            "val":   round(val_acc_final, 5),
            "test":  round(test_acc_final, 5),
        },
        "test_metrics": {
            "precision_weighted": round(test_precision, 5),
            "recall_weighted":    round(test_recall, 5),
            "f1_weighted":        round(test_f1, 5),
        },
        "per_class": per_class_report,
        "confusion_matrix": cm,
        "class_names": class_names,
        "config": {
            "epochs_run":  epoch,
            "batch_size":  args.batch_size,
            "lr":          args.lr,
            "img_size":    args.img_size,
            "total_time_s": round(total_time, 1),
            "device":      str(device),
        },
    }

    with open(run_models_dir / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(metrics_test, f, indent=2, ensure_ascii=False)

    print(f"   metrics_test.json guardado en {run_models_dir}")

    #  Resumen final 
    print(f"\n{'-'*70}")
    print(f"  RESUMEN FINAL")
    print(f"{'-'*70}")
    print(f"  Modelo:        {args.model_name}")
    print(f"  Mejor epoca:   {best_epoch}/{epoch}")
    print(f"  Test Accuracy: {test_acc_final:.4f}")
    print(f"  Test F1:       {test_f1:.4f}")
    print(f"  Tiempo total:  {total_time/60:.1f} min")
    print(f"")
    print(f"  Archivos guardados en: {run_models_dir}")
    print(f"     best_model.pth")
    print(f"     last_model.pth")
    print(f"     history.json")
    print(f"     metrics_test.json")
    print(f"{'-'*70}\n")


# 
#  ENTRY POINT
# 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 5  Entrenamiento CNN para Reconocimiento Facial",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",      type=int,   default=50,    help="Numero maximo de epocas")
    parser.add_argument("--batch_size",  type=int,   default=32,    help="Tamano del batch")
    parser.add_argument("--lr",          type=float, default=3e-4,  help="Learning rate inicial")
    parser.add_argument("--img_size",    type=int,   default=160,   help="Tamano de imagen (lado)")
    parser.add_argument("--num_workers", type=int,   default=2,     help="Workers para DataLoader")
    parser.add_argument("--model_name",  type=str,   default="resnet18",
                        choices=["resnet18", "efficientnet_b0"],
                        help="Arquitectura base de transfer learning")
    parser.add_argument("--grupos",      type=str,   default=None,  nargs="+",
                        help="Grupos a incluir en el entrenamiento (ej: --grupos Alumnos Actores). "
                             "Si no se especifica, entrena con todos los grupos.")

    args = parser.parse_args()
    main(args)
