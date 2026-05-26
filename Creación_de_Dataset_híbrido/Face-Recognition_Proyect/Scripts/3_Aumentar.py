"""
3_Aumentar.py
=============
Fase 3 - Aumentacion de datos.

Estructura soportada (estricta):
  Dataset_procesado/Persona/*.jpg
"""


import cv2
import numpy as np
import argparse
import random
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "Dataset_procesado"
OUT_DIR = PROJECT_ROOT / "Dataset_aumentado"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _tiene_imagenes_directas(carpeta: Path) -> bool:
    return any(f.is_file() and f.suffix.lower() in EXTS for f in carpeta.iterdir())


def leer_imagen_compatible(ruta: Path):
    """
    Lee imagen de forma robusta en rutas Unicode (Windows/Linux).
    """
    try:
        data = np.fromfile(str(ruta), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def guardar_imagen_compatible(ruta: Path, imagen: np.ndarray, calidad=95) -> bool:
    """
    Guarda imagen de forma robusta en rutas Unicode (Windows/Linux).
    """
    ext = ruta.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, int(calidad)] if ext in {".jpg", ".jpeg"} else []
    ok, buffer = cv2.imencode(ext, imagen, params)
    if not ok:
        return False
    try:
        buffer.tofile(str(ruta))
        return True
    except Exception:
        return False


def iterar_clases(base: Path):
    """
    Yields (clase_dir, etiqueta) para estructura plana por persona.
    Si detecta estructura por grupos, lanza ValueError.
    """
    nivel1 = sorted([d for d in base.iterdir() if d.is_dir()])
    if not nivel1:
        return

    es_plana = any(_tiene_imagenes_directas(d) for d in nivel1)
    if not es_plana:
        raise ValueError(
            "Estructura no compatible: se detectaron grupos.\n"
            "Usa solo Dataset_procesado/<Persona>/<imagenes>."
        )
    for clase_dir in nivel1:
        yield clase_dir, clase_dir.name


def rotacion(img):
    ang = random.uniform(-15, 15)
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), ang, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


def cambio_brillo(img):
    factor = random.uniform(0.50, 1.60)
    img_f = img.astype(np.float32) * factor
    return np.clip(img_f, 0, 255).astype(np.uint8)


def espejo(img):
    return cv2.flip(img, 1)


def ruido_gaussiano(img):
    sigma = random.uniform(5, 20)
    ruido = np.random.normal(0, sigma, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)


def recorte_zoom(img):
    h, w = img.shape[:2]
    margen = int(random.uniform(0.05, 0.15) * min(h, w))
    top = random.randint(0, margen)
    left = random.randint(0, margen)
    bot = random.randint(0, margen)
    right = random.randint(0, margen)
    rec = img[top:h - bot, left:w - right]
    return cv2.resize(rec, (w, h), interpolation=cv2.INTER_LANCZOS4)


def ajuste_contraste(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clip = random.uniform(1.0, 4.0)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def blur_leve(img):
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)


def cambio_tono(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-18, 18)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.75, 1.25), 0, 255)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


BASICAS = [rotacion, cambio_brillo, espejo]
ADICIONALES = [ruido_gaussiano, recorte_zoom, ajuste_contraste, blur_leve, cambio_tono]


def generar_variantes(img, n, solo_basicas):
    pool = BASICAS if solo_basicas else BASICAS + ADICIONALES
    variantes = [espejo(img)]
    for _ in range(n - 1):
        out = img.copy()
        ops = random.sample([f for f in pool if f != espejo], k=random.randint(1, min(3, len(pool) - 1)))
        for op in ops:
            out = op(out)
        variantes.append(out)
    return variantes


def aumentar_dataset(factor=6, solo_basicas=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  AUMENTACION DE DATOS")
    print(f"  Factor: x{factor} (1 original + {factor-1} sinteticas)")
    print(f"  Fuente:  {BASE_DIR}")
    print(f"  Destino: {OUT_DIR}")
    print(f"{'='*60}\n")

    total_orig = 0
    total_gen = 0

    try:
        clases_iter = list(iterar_clases(BASE_DIR))
    except ValueError as ex:
        print(f"\n  ERROR: {ex}\n")
        return

    for clase_dir, etiqueta in clases_iter:
        imagenes = [f for f in clase_dir.iterdir() if f.suffix.lower() in EXTS]
        if not imagenes:
            continue

        carpeta_out = OUT_DIR / clase_dir.name
        carpeta_out.mkdir(parents=True, exist_ok=True)

        conteo = 0
        for ruta in tqdm(imagenes, desc=f"  {etiqueta}", leave=False):
            img = leer_imagen_compatible(ruta)
            if img is None:
                continue

            if guardar_imagen_compatible(carpeta_out / ruta.name, img, calidad=95):
                conteo += 1

            variantes = generar_variantes(img, factor - 1, solo_basicas)
            for i, var in enumerate(variantes):
                nombre = f"{ruta.stem}_aug{i+1:02d}.jpg"
                if guardar_imagen_compatible(carpeta_out / nombre, var, calidad=92):
                    conteo += 1

        orig = len(imagenes)
        gen = conteo - orig
        total_orig += orig
        total_gen += gen
        print(f"  ok {etiqueta:35s} {orig:4d} orig +{gen:5d} sint = {conteo:5d}")

    total = total_orig + total_gen
    factor_real = (total / total_orig) if total_orig else 0.0
    print(f"\n{'='*60}")
    print(f"  Imagenes originales: {total_orig}")
    print(f"  Imagenes generadas:  {total_gen}")
    print(f"  TOTAL:               {total}")
    print(f"  Factor real:         x{factor_real:.1f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=int, default=6, help="Multiplicador total por imagen")
    parser.add_argument("--solo_basicas", action="store_true", help="Usar solo rotacion/brillo/espejo")
    args = parser.parse_args()
    aumentar_dataset(factor=args.factor, solo_basicas=args.solo_basicas)
