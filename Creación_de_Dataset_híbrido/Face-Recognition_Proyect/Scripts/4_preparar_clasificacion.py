"""
4_preparar_clasificacion.py
===========================
Fase 4 - Preparacion de splits train/val/test para clasificacion multiclase.

Estructura soportada (estricta):
  Dataset_aumentado/Persona/*.jpg
"""


import cv2
import json
import random
import argparse
import csv
import re
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "Dataset_aumentado"
SPLIT_DIR = PROJECT_ROOT / "splits"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
GRUPO_DEFAULT = "Directo"


def _tiene_imagenes_directas(carpeta: Path) -> bool:
    return any(f.is_file() and f.suffix.lower() in EXTS for f in carpeta.iterdir())


def _iterar_clases(base: Path):
    """
    Yields (clase_dir) para estructura plana por persona.
    Si detecta estructura por grupos, lanza ValueError.
    """
    nivel1 = sorted([d for d in base.iterdir() if d.is_dir()])
    if not nivel1:
        return

    es_plana = any(_tiene_imagenes_directas(d) for d in nivel1)
    if not es_plana:
        raise ValueError(
            "Estructura no compatible: se detectaron grupos.\n"
            "Usa solo Dataset_aumentado/<Persona>/<imagenes>."
        )
    for clase_dir in nivel1:
        yield clase_dir


def recolectar_muestras(base: Path):
    """
    Recolecta muestras con campos:
      ruta, clase, grupo, base_id
    """
    muestras = []
    clases = set()

    for clase_dir in _iterar_clases(base):
        clase = clase_dir.name
        clases.add(clase)
        for img in clase_dir.iterdir():
            if img.is_file() and img.suffix.lower() in EXTS:
                muestras.append(
                    {
                        "ruta": str(img),
                        "clase": clase,
                        "grupo": GRUPO_DEFAULT,
                        "base_id": construir_base_id(img),
                    }
                )

    return muestras, sorted(clases)


def construir_base_id(ruta_img: Path) -> str:
    """
    Obtiene el identificador base de la imagen para mantener juntas
    las variantes aumentadas (_augXX) y la imagen original.
    """
    stem = ruta_img.stem
    stem_base = re.sub(r"_aug\d+$", "", stem, flags=re.IGNORECASE)
    return stem_base


def dividir_estratificado_por_base(muestras, train_r, val_r, test_r, seed=SEED):
    """
    Split estratificado por clase, agrupando por base_id para evitar fuga de datos.
    Todas las variantes de una misma imagen base van al mismo split.
    """
    random.seed(seed)
    por_clase = defaultdict(list)
    for m in muestras:
        por_clase[m["clase"]].append(m)

    train, val, test = [], [], []
    for _, items_clase in por_clase.items():
        grupos = defaultdict(list)
        for item in items_clase:
            grupos[item["base_id"]].append(item)

        grupos_lista = list(grupos.values())
        random.shuffle(grupos_lista)

        n_grupos = len(grupos_lista)
        n_tr = int(n_grupos * train_r)
        n_val = int(n_grupos * val_r)

        grupos_train = grupos_lista[:n_tr]
        grupos_val = grupos_lista[n_tr:n_tr + n_val]
        grupos_test = grupos_lista[n_tr + n_val:]

        for g in grupos_train:
            train.extend(g)
        for g in grupos_val:
            val.extend(g)
        for g in grupos_test:
            test.extend(g)

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def guardar_csv(muestras, ruta, mapeo_clases):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ruta", "clase", "clase_idx", "grupo"])
        for m in muestras:
            writer.writerow([
                m["ruta"],
                m["clase"],
                mapeo_clases[m["clase"]],
                m.get("grupo", GRUPO_DEFAULT),
            ])


def verificar_imagen(ruta: str) -> bool:
    img = cv2.imread(ruta)
    return img is not None and img.shape[:2] == (160, 160)


def imprimir_reporte(train, val, test, clases):
    total = len(train) + len(val) + len(test)
    print(f"\n{'='*62}")
    print("  SPLITS GENERADOS")
    print(f"{'-'*62}")
    print(f"  {'Split':<10} {'Imagenes':>10}  {'%':>6}")
    print(f"  {'-'*30}")
    for nombre, split in [("Train", train), ("Val", val), ("Test", test)]:
        pct = (len(split) / total * 100) if total else 0
        print(f"  {nombre:<10} {len(split):>10,}  {pct:>5.1f}%")
    print(f"  {'-'*30}")
    print(f"  {'TOTAL':<10} {total:>10,}  100.0%")
    print(f"\n  Clases ({len(clases)}):")
    conteos = Counter(m["clase"] for m in train + val + test)
    for clase in clases:
        n = conteos[clase]
        barra = "#" * min(n // 10, 30)
        print(f"    {clase:24s} {n:5d}  {barra}")
    print(f"{'='*62}\n")


def preparar(train_r=0.70, val_r=0.15, test_r=0.15):
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "Los ratios deben sumar 1.0"

    if not BASE_DIR.exists():
        print(f"\n  ERROR: No se encontro {BASE_DIR}")
        print("  Ejecuta primero Fase 2 y Fase 3.\n")
        return

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Recolectando imagenes de {BASE_DIR}...")
    try:
        muestras, clases = recolectar_muestras(BASE_DIR)
    except ValueError as ex:
        print(f"\n  ERROR: {ex}\n")
        return
    print(f"  Total: {len(muestras):,} imagenes | {len(clases)} clases")

    mapeo_clases = {c: i for i, c in enumerate(clases)}
    mapeo_inverso = {i: c for c, i in mapeo_clases.items()}

    print(f"\n  Dividiendo dataset por imagen base (train={train_r:.0%} / val={val_r:.0%} / test={test_r:.0%})...")
    train, val, test = dividir_estratificado_por_base(muestras, train_r, val_r, test_r)

    guardar_csv(train, SPLIT_DIR / "train.csv", mapeo_clases)
    guardar_csv(val, SPLIT_DIR / "val.csv", mapeo_clases)
    guardar_csv(test, SPLIT_DIR / "test.csv", mapeo_clases)
    print(f"  ok train.csv, val.csv, test.csv -> {SPLIT_DIR}")

    config = {
        "n_clases": len(clases),
        "clases": mapeo_inverso,
        "clases_nombre": clases,
        "resolucion": "160x160",
        "total_imagenes": len(muestras),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
    }
    with open(SPLIT_DIR / "clases.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  ok clases.json -> {SPLIT_DIR / 'clases.json'}")

    imprimir_reporte(train, val, test, clases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()
    preparar(train_r=args.train, val_r=args.val, test_r=args.test)
