"""
1_Captura.py
============
Fase 1 - Captura de rostros para dataset (estructura plana por persona).

Estructura esperada:
  Dataset/PersonaA/*.jpg

Ejemplos:
  python Scripts/1_Captura.py --persona Joey
  python Scripts/1_Captura.py --persona Joey --meta 300 --cooldown 0.7
  python Scripts/1_Captura.py --persona "Juan Perez" --fuente 1
"""


import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from facenet_pytorch import MTCNN

    TIENE_FACENET = True
except ImportError:
    TIENE_FACENET = False

try:
    import torch
except ImportError:
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
EXTENSION = ".jpg"


def parse_args():
    """Define y parsea los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(
        description="Captura automatica de rostros para una persona."
    )
    parser.add_argument(
        "--persona",
        type=str,
        default="Joey",
        help="Nombre de la persona (carpeta destino en Dataset).",
    )
    parser.add_argument(
        "--meta",
        type=int,
        default=220,
        help="Cantidad de fotos NUEVAS a capturar en esta sesion.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.65,
        help="Segundos minimos entre una foto y la siguiente.",
    )
    parser.add_argument(
        "--fuente",
        type=int,
        default=0,
        help="Indice de camara a usar (por defecto 0).",
    )
    return parser.parse_args()


def _normalizar_nombre_archivo(nombre: str) -> str:
    """Normaliza el nombre de persona para usarlo en archivos."""
    base = re.sub(r"\s+", "_", nombre.strip())
    base = re.sub(r"[^A-Za-z0-9_]", "", base)
    return base or "persona"


def _siguiente_indice(carpeta_persona: Path, prefijo: str) -> int:
    """
    Devuelve el siguiente indice numerico disponible segun archivos existentes.
    Espera formato: prefijo_123.jpg
    """
    patron = re.compile(rf"^{re.escape(prefijo)}_(\d+){re.escape(EXTENSION)}$")
    max_idx = 0
    for archivo in carpeta_persona.glob(f"*{EXTENSION}"):
        m = patron.match(archivo.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _crear_detector():
    """
    Crea detector principal (facenet_pytorch MTCNN) o fallback Haar Cascade.
    Retorna (detector, modo, device).
    """
    if TIENE_FACENET:
        device = "cpu"
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
        detector = MTCNN(keep_all=True, device=device)
        return detector, "mtcnn", device

    xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(xml)
    return detector, "haar", "cpu"


def _detectar_rostro_mayor(frame_bgr: np.ndarray, detector, modo: str):
    """
    Detecta rostros y retorna el de mayor area como bbox (x1, y1, x2, y2).
    Si no detecta, retorna None.
    """
    if modo == "mtcnn":
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, _ = detector.detect(frame_rgb)
        if boxes is None or len(boxes) == 0:
            return None
        boxes = np.asarray(boxes, dtype=np.float32)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
        x1, y1, x2, y2 = boxes[idx].astype(int).tolist()
        return x1, y1, x2, y2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    caras = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(caras) == 0:
        return None
    x, y, w, h = max(caras, key=lambda r: r[2] * r[3])
    return x, y, x + w, y + h


def _recortar_con_margen(frame_bgr: np.ndarray, bbox, margen_relativo=0.20):
    """Recorta rostro con margen relativo alrededor del bbox."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    margen = int(max(bw, bh) * margen_relativo)

    rx1 = max(0, x1 - margen)
    ry1 = max(0, y1 - margen)
    rx2 = min(w, x2 + margen)
    ry2 = min(h, y2 + margen)

    recorte = frame_bgr[ry1:ry2, rx1:rx2]
    if recorte.size == 0:
        return None
    return recorte


def _guardar_imagen_compatible(ruta: Path, imagen_bgr: np.ndarray) -> bool:
    """
    Guarda imagen de forma robusta en rutas con caracteres Unicode (Windows/Linux).
    cv2.imwrite puede fallar con acentos en algunas instalaciones de Windows.
    """
    ext = ruta.suffix.lower()
    ok, buffer = cv2.imencode(ext if ext else ".jpg", imagen_bgr)
    if not ok:
        return False
    try:
        buffer.tofile(str(ruta))
        return True
    except Exception:
        return False


def capturar_imagenes(persona: str, meta: int, cooldown: float, fuente: int):
    """
    Ejecuta la captura automatica de imagenes de una persona.
    """
    if meta <= 0:
        print("Error: --meta debe ser un entero mayor a 0.")
        return 1
    if cooldown < 0:
        print("Error: --cooldown no puede ser negativo.")
        return 1

    carpeta_persona = DATASET_DIR / persona
    carpeta_persona.mkdir(parents=True, exist_ok=True)

    prefijo = _normalizar_nombre_archivo(persona)
    existentes = len(list(carpeta_persona.glob(f"*{EXTENSION}")))
    siguiente_idx = _siguiente_indice(carpeta_persona, prefijo)

    detector, modo_detector, device = _crear_detector()
    if modo_detector == "mtcnn":
        print(f"Detector: MTCNN (facenet_pytorch) | device={device}")
    else:
        print("Detector: Haar Cascade (fallback, instala facenet_pytorch para mejorar)")

    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir la camara con indice {fuente}.")
        return 1

    print("\n=== CAPTURA DE DATASET ===")
    print(f"Persona: {persona}")
    print(f"Destino: {carpeta_persona}")
    print(f"Fotos existentes: {existentes}")
    print(f"Meta sesion (nuevas): {meta}")
    print(f"Cooldown entre fotos: {cooldown:.2f}s")
    print("Controles: presiona 'q' para salir.\n")

    capturadas = 0
    fallidas = 0
    ultima_captura = 0.0
    inicio = time.time()

    try:
        while capturadas < meta:
            ok, frame = cap.read()
            if not ok:
                print("Advertencia: no se pudo leer frame de la camara.")
                break

            frame_vis = frame.copy()
            bbox = _detectar_rostro_mayor(frame, detector, modo_detector)
            ahora = time.time()
            restante = max(0.0, cooldown - (ahora - ultima_captura))

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(
                    frame_vis,
                    "Rostro detectado",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )

                if restante <= 0.0:
                    recorte = _recortar_con_margen(frame, bbox, margen_relativo=0.20)
                    if recorte is not None:
                        nombre = f"{prefijo}_{siguiente_idx}{EXTENSION}"
                        ruta_salida = carpeta_persona / nombre
                        if _guardar_imagen_compatible(ruta_salida, recorte):
                            capturadas += 1
                            siguiente_idx += 1
                            ultima_captura = ahora
                            restante = cooldown
                        else:
                            fallidas += 1
                            print(
                                f"Advertencia: no se pudo guardar {ruta_salida.name}"
                            )

            total_actual = existentes + capturadas
            barra = (
                f"Nuevas: {capturadas}/{meta}  |  "
                f"Total en carpeta: {total_actual}  |  "
                f"Cooldown: {restante:.2f}s"
            )
            cv2.putText(
                frame_vis,
                barra,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Captura Dataset - Presiona Q para salir", frame_vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nCaptura interrumpida por el usuario.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    duracion = max(0.001, time.time() - inicio)
    print("\n=== RESUMEN ===")
    print(f"Nuevas capturas: {capturadas}")
    print(f"Guardados fallidos: {fallidas}")
    print(f"Total final en carpeta: {existentes + capturadas}")
    print(f"Tiempo total: {duracion:.1f}s")
    print(f"Promedio: {capturadas / duracion:.2f} fotos/s")

    return 0


def main():
    args = parse_args()
    codigo = capturar_imagenes(
        persona=args.persona,
        meta=args.meta,
        cooldown=args.cooldown,
        fuente=args.fuente,
    )
    sys.exit(codigo)


if __name__ == "__main__":
    main()
