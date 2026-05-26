"""
2_Preprocesar.py
================
Fase 2 - Preprocesamiento facial.

Estructura soportada (estricta):
  Dataset/PersonaA/*.jpg
"""


import cv2
import numpy as np
import argparse
import json
import shutil
import tempfile
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

try:
    from mtcnn import MTCNN
    TIENE_MTCNN = True
except ImportError:
    TIENE_MTCNN = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "Dataset"
OUT_DIR = PROJECT_ROOT / "Dataset_procesado"
LOG_DIR = PROJECT_ROOT / "logs"
TAMANO = (160, 160)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_HAAR_WARNED = False


def json_safe(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


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


def cargar_haar_compatible():
    """
    Carga Haar Cascade de forma robusta.
    En Windows, cv2 puede fallar con rutas Unicode; se intenta copiar el XML
    a una ruta temporal ASCII.
    """
    cascade_name = "haarcascade_frontalface_default.xml"
    candidatos = []

    try:
        candidatos.append(Path(cv2.data.haarcascades) / cascade_name)
    except Exception:
        pass

    try:
        candidatos.append(Path(cv2.__file__).resolve().parent / "data" / cascade_name)
    except Exception:
        pass

    for ruta in candidatos:
        if not ruta.exists():
            continue

        # Intento directo
        cascade = cv2.CascadeClassifier(str(ruta))
        if not cascade.empty():
            return cascade

        # Fallback: copiar a ruta ASCII temporal
        try:
            temp_dir = Path(tempfile.gettempdir()) / "opencv_haar_cache"
            temp_dir.mkdir(parents=True, exist_ok=True)
            ruta_temp = temp_dir / cascade_name
            shutil.copyfile(str(ruta), str(ruta_temp))
            cascade = cv2.CascadeClassifier(str(ruta_temp))
            if not cascade.empty():
                return cascade
        except Exception:
            continue

    return None


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
            "Usa solo Dataset/<Persona>/<imagenes>."
        )
    for clase_dir in nivel1:
        yield clase_dir, clase_dir.name


class ProcesadorMTCNN:
    def __init__(self, umbral_confianza=0.90):
        self.model = MTCNN()
        self.umbral = umbral_confianza

    def procesar(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resultados = self.model.detect_faces(img_rgb)
        if not resultados:
            return None, None

        det = max(resultados, key=lambda r: r["confidence"])
        if det["confidence"] < self.umbral:
            return None, None

        kp = det["keypoints"]
        ojo_izq = np.array(kp["left_eye"], dtype=np.float32)
        ojo_der = np.array(kp["right_eye"], dtype=np.float32)

        dy = ojo_der[1] - ojo_izq[1]
        dx = ojo_der[0] - ojo_izq[0]
        angulo = np.degrees(np.arctan2(dy, dx))

        centro = tuple(((ojo_izq + ojo_der) / 2).astype(int))
        h, w = img_bgr.shape[:2]
        m = cv2.getRotationMatrix2D(centro, angulo, scale=1.0)
        img_alineada = cv2.warpAffine(
            img_bgr,
            m,
            (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT,
        )

        x, y, bw, bh = det["box"]
        margen = int(0.20 * max(bw, bh))
        x1 = max(0, x - margen)
        y1 = max(0, y - margen)
        x2 = min(w, x + bw + margen)
        y2 = min(h, y + bh + margen)

        recorte = img_alineada[y1:y2, x1:x2]
        if recorte.size == 0:
            return None, None

        recorte_160 = cv2.resize(recorte, TAMANO, interpolation=cv2.INTER_LANCZOS4)
        meta = {
            "confianza": round(det["confidence"], 4),
            "angulo_correccion": round(angulo, 2),
            "bbox_original": [x, y, bw, bh],
        }
        return recorte_160, meta


class ProcesadorHaar:
    def __init__(self, **kwargs):
        self.cascade = cargar_haar_compatible()
        if self.cascade is None:
            print("  ⚠ No se pudo cargar Haar Cascade; se omitirá detección Haar.")

    def procesar(self, img_bgr):
        global _HAAR_WARNED
        if self.cascade is None:
            if not _HAAR_WARNED:
                print("  ⚠ Haar no disponible en esta ejecución.")
                _HAAR_WARNED = True
            return None, None

        gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        try:
            rostros = self.cascade.detectMultiScale(gris, 1.1, 5, minSize=(60, 60))
        except cv2.error:
            return None, None
        if len(rostros) == 0:
            return None, None
        x, y, bw, bh = max(rostros, key=lambda r: r[2] * r[3])
        margen = int(0.20 * max(bw, bh))
        h, w = img_bgr.shape[:2]
        x1 = max(0, x - margen)
        y1 = max(0, y - margen)
        x2 = min(w, x + bw + margen)
        y2 = min(h, y + bh + margen)
        recorte = img_bgr[y1:y2, x1:x2]
        if recorte.size == 0:
            return None, None
        recorte_160 = cv2.resize(recorte, TAMANO, interpolation=cv2.INTER_LANCZOS4)
        meta = {"confianza": 1.0, "angulo_correccion": 0.0, "bbox_original": [x, y, bw, bh]}
        return recorte_160, meta


def procesar_dataset(umbral: float = 0.90):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if TIENE_MTCNN:
        print(f"\n  Detector: MTCNN (umbral={umbral})")
        procesador = ProcesadorMTCNN(umbral_confianza=umbral)
    else:
        print("\n  Detector: Haar Cascade (instala mtcnn para mejor calidad)")
        procesador = ProcesadorHaar()

    print(f"  Fuente: {BASE_DIR}")
    print(f"  Destino: {OUT_DIR}")
    print(f"  Resolucion: {TAMANO[0]}x{TAMANO[1]}\n")

    log_global = {
        "timestamp": datetime.now().isoformat(),
        "detector": "MTCNN" if TIENE_MTCNN else "Haar",
        "resolucion": f"{TAMANO[0]}x{TAMANO[1]}",
        "categorias": {},
    }

    total_ok = 0
    total_skip = 0

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

        ok = 0
        skip = 0
        metas = []

        for ruta in tqdm(imagenes, desc=f"  {etiqueta}", leave=False):
            img = leer_imagen_compatible(ruta)
            if img is None:
                skip += 1
                continue

            rostro, meta = procesador.procesar(img)
            if rostro is None:
                skip += 1
                continue

            nombre_out = ruta.stem + "_proc.jpg"
            ruta_out = carpeta_out / nombre_out
            if guardar_imagen_compatible(ruta_out, rostro, calidad=95):
                metas.append({**meta, "archivo": ruta.name})
                ok += 1
            else:
                skip += 1

        total_ok += ok
        total_skip += skip
        estado = "ok" if skip == 0 else f"ok ({skip} sin rostro)"
        print(f"  {estado:16s} {etiqueta:35s} {ok:4d} procesadas")

        log_global["categorias"][etiqueta] = {
            "procesadas": ok,
            "omitidas": skip,
            "muestras": metas[:5],
        }

    log_path = LOG_DIR / "preprocesamiento.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log_global, f, indent=2, ensure_ascii=False, default=json_safe)

    print(f"\n{'='*60}")
    print("  PREPROCESAMIENTO COMPLETO")
    print(f"  Imagenes procesadas: {total_ok}")
    print(f"  Imagenes omitidas:   {total_skip}")
    print(f"  Log guardado en:     {log_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confianza", type=float, default=0.90, help="Umbral MTCNN")
    args = parser.parse_args()
    procesar_dataset(umbral=args.confianza)
