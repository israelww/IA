# -*- coding: utf-8 -*-
"""
demo.py
=======
Demo unificado del sistema de reconocimiento facial.

Este script es el punto de entrada para usuario final y reutiliza la logica ya
implementada en:
  - Fase 6 (inferencia de imagen y utilidades del modelo)
  - Fase 7 (pipeline completo de camara)
"""

import sys
import argparse
import warnings
import inspect
import shutil
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import cv2
import torch
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# Forzar UTF-8 en stdout/stderr para compatibilidad en Windows/Linux
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTADOS_DIR = PROJECT_ROOT / "resultados"
DEFAULT_MODEL = MODELS_DIR / "best_model.pth"


# ---------------------------------------------------------------------------
# 1) IMPORTACION DINAMICA OBLIGATORIA (mismo patron de fases previas)
# ---------------------------------------------------------------------------
def _importar_modulo(nombre, archivo):
    scripts_dir = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(nombre, scripts_dir / archivo)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


_fase6 = _importar_modulo("inferencia_demo", "6_inferencia.py")
_fase7 = _importar_modulo("camara_demo", "7_pipeline_camara.py")

# Funciones reutilizadas de Fase 6
load_model = _fase6.load_model
predict = _fase6.predict
detectar_todos_los_rostros = _fase6.detectar_todos_los_rostros
predict_imagen_subida = _fase6.predict_imagen_subida

# Funcion reutilizada de Fase 7
run_pipeline = _fase7.run_pipeline


# ---------------------------------------------------------------------------
# 2) UTILIDADES
# ---------------------------------------------------------------------------
def _formatear_nombre_modelo(raw_name):
    """Convierte nombre de checkpoint a version legible para banner."""
    if not raw_name:
        return "Desconocido"
    txt = str(raw_name).strip().lower()
    if txt == "resnet18":
        return "ResNet18"
    if txt == "efficientnet_b0":
        return "EfficientNet-B0"
    return str(raw_name)


def _leer_meta_checkpoint(modelo_path):
    """
    Lee metadatos basicos del checkpoint para mostrar bienvenida sin
    reconstruir el modelo completo.
    """
    ckpt = torch.load(modelo_path, map_location="cpu", weights_only=False)
    model_name = _formatear_nombre_modelo(ckpt.get("model_name", "Desconocido"))
    num_classes = int(ckpt.get("num_classes", 0))
    threshold = float(ckpt.get("confidence_threshold", 0.70))
    return model_name, num_classes, threshold


def _mostrar_bienvenida(model_name, num_classes, threshold):
    """Imprime pantalla de bienvenida del demo."""
    print()
    print("")
    print("        SISTEMA DE RECONOCIMIENTO FACIAL  DEMO                  ")
    print(
        f"        Modelo: {model_name:<9}   "
        f"Clases: {num_classes:<3}    Umbral: {threshold:0.2f}         "
    )
    print("")
    print()
    print("  Modos disponibles:")
    print("    --camara          Reconocimiento en tiempo real con la camara")
    print("    --imagen foto.jpg Analizar una imagen subida")
    print()
    print("  Opciones adicionales:")
    print("    --grabar          Grabar el video de la sesion  (solo con --camara)")
    print("    --mostrar         Abrir ventana con la imagen anotada (solo con --imagen)")
    print("    --modelo path     Usar un modelo alternativo")
    print("    --fuente N        Indice de camara a usar (por defecto 0)")
    print()


def _validar_camara_disponible(fuente):
    """Valida que la camara solicitada puede abrirse y la cierra enseguida."""
    cap = cv2.VideoCapture(fuente)
    try:
        if not cap.isOpened():
            print(f"  ERROR: No se pudo abrir la camara con indice {fuente}.")
            print("  Verifica conexion/permisos y vuelve a intentar.")
            sys.exit(1)
    finally:
        cap.release()


def _mostrar_imagen_anotada(ruta_img):
    """Abre con OpenCV la imagen anotada guardada."""
    ruta_img = Path(ruta_img)
    if not ruta_img.exists():
        print(f"  Aviso: no se encontro la imagen anotada para mostrar: {ruta_img}")
        return

    # Carga robusta para rutas Unicode (Windows)
    img_bytes = np.fromfile(str(ruta_img), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  Aviso: no se pudo abrir la imagen anotada: {ruta_img}")
        return

    window = "Demo - Imagen anotada"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, img)
    print("  Presiona cualquier tecla para cerrar la vista previa...")
    cv2.waitKey(0)
    cv2.destroyWindow(window)


def _imprimir_resultado_imagen(result):
    """Imprime resultado de inferencia en formato amigable de demo."""
    persona = result.get("clase", "Desconocido")
    confianza = float(result.get("confianza", 0.0))
    conocido = bool(result.get("conocido", False))
    rostro = bool(result.get("face_detected", False))
    top3 = result.get("top3", [])
    img_out = result.get("imagen_anotada", "")

    try:
        rel_out = Path(img_out).resolve().relative_to(PROJECT_ROOT.resolve())
        out_txt = str(rel_out).replace("\\", "/")
    except Exception:
        out_txt = str(img_out)

    print("  ")
    print("    Resultado")
    print("  ")
    print(f"    Persona:    {persona}")
    print(f"    Confianza:  {confianza*100:.1f}%")
    print(f"    Conocido:   {'Si' if conocido else 'No'}")
    print(f"    Rostro:     {'Detectado' if rostro else 'No detectado'}")
    print()
    print("    Top-3:")
    for i, pred in enumerate(top3[:3], start=1):
        nombre = str(pred.get("clase", "N/D"))
        conf = float(pred.get("confianza", 0.0))
        print(f"      {i}. {nombre:<18} {conf*100:5.1f}%")
    print()
    print("    Imagen guardada en:")
    print(f"    {out_txt}")
    print("  ")


def _mover_imagen_a_carpeta_demo(result):
    """
    Reubica la imagen anotada a resultados/imagenes_anotadas para mantener
    la estructura final del proyecto desde el punto de entrada demo.
    """
    ruta_actual = Path(result.get("imagen_anotada", ""))
    if not ruta_actual.exists():
        return result

    destino_dir = RESULTADOS_DIR / "imagenes_anotadas"
    destino_dir.mkdir(parents=True, exist_ok=True)
    destino = destino_dir / ruta_actual.name

    try:
        shutil.move(str(ruta_actual), str(destino))
        result["imagen_anotada"] = str(destino)
    except Exception:
        # Si no se puede mover (p. ej. mismo archivo), mantener la ruta original.
        result["imagen_anotada"] = str(ruta_actual)

    return result


def _ejecutar_modo_camara(args, model, class_names, img_size, confidence_threshold, device):
    """
    Ejecuta el pipeline de camara usando la funcion existente de Fase 7
    sin reimplementar su loop.
    """
    model_display_name = type(model).__name__
    firma = inspect.signature(run_pipeline)
    params = list(firma.parameters.keys())

    # Compatibilidad con firma real de Fase 7 (run_pipeline(args, ...))
    if params and params[0] == "args":
        args_pipeline = SimpleNamespace(
            fuente=args.fuente,
            skip_frames=2,
            resize_factor=0.5,
            grabar=args.grabar,
        )
        run_pipeline(
            args_pipeline,
            model,
            class_names,
            img_size,
            confidence_threshold,
            device,
            model_display_name,
        )
        return

    # Fallback de compatibilidad si en otra version existe firma directa.
    run_pipeline(
        model,
        class_names,
        img_size,
        confidence_threshold,
        device,
        fuente=args.fuente,
        skip_frames=2,
        resize_factor=0.5,
        grabar=args.grabar,
    )


def _limpiar_recursos():
    """Libera recursos graficos de OpenCV."""
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 3) FLUJO PRINCIPAL
# ---------------------------------------------------------------------------
def main(args):
    modelo_path = Path(args.modelo) if args.modelo else DEFAULT_MODEL

    # Validacion 1: checkpoint existe
    if not modelo_path.exists():
        print(f"  ERROR: No se encontro el checkpoint: {modelo_path}")
        print("  Debes ejecutar primero la Fase 5 (5_entrenar_cnn.py).")
        sys.exit(1)

    # Validacion 2: facenet_pytorch disponible
    try:
        import facenet_pytorch  # noqa: F401
    except Exception:
        print("  Aviso: facenet_pytorch no esta instalado.")
        print("  Se usara Haar Cascade como fallback para deteccion facial.")

    # Validacion 3: camara disponible cuando aplica
    if args.camara:
        _validar_camara_disponible(args.fuente)

    # Metadatos para pantalla de bienvenida
    model_name, num_classes, threshold_meta = _leer_meta_checkpoint(modelo_path)
    _mostrar_bienvenida(model_name, num_classes, threshold_meta)

    # Sin argumentos de modo: solo bienvenida y salir
    if not args.camara and not args.imagen:
        return

    # Validaciones de coherencia de CLI
    if args.camara and args.imagen:
        print("  ERROR: Usa solo un modo a la vez: --camara o --imagen.")
        sys.exit(1)

    if args.grabar and not args.camara:
        print("  Aviso: --grabar solo aplica en modo camara; se ignorara.")

    if args.mostrar and not args.imagen:
        print("  Aviso: --mostrar solo aplica con --imagen; se ignorara.")

    # Cargar modelo con la funcion oficial de Fase 6
    model, class_names, img_size, confidence_threshold, device = load_model(str(modelo_path))

    # Modo camara
    if args.camara:
        _ejecutar_modo_camara(
            args=args,
            model=model,
            class_names=class_names,
            img_size=img_size,
            confidence_threshold=confidence_threshold,
            device=device,
        )
        return

    # Modo imagen
    imagen_path = Path(args.imagen)
    if not imagen_path.exists():
        print(f"  ERROR: No se encontro la imagen: {imagen_path}")
        print("  Verifica la ruta e intentalo de nuevo.")
        sys.exit(1)

    result = predict_imagen_subida(
        str(imagen_path),
        model,
        class_names,
        img_size,
        confidence_threshold,
        device,
    )
    result = _mover_imagen_a_carpeta_demo(result)
    _imprimir_resultado_imagen(result)

    if args.mostrar:
        _mostrar_imagen_anotada(result.get("imagen_anotada", ""))


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Demo unificado del sistema de reconocimiento facial",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camara", action="store_true", help="Reconocimiento en vivo con camara")
    parser.add_argument("--imagen", type=str, default=None, help="Ruta de imagen para analizar")
    parser.add_argument("--grabar", action="store_true", help="Grabar sesion de camara")
    parser.add_argument("--mostrar", action="store_true", help="Mostrar imagen anotada al finalizar")
    parser.add_argument(
        "--modelo",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Ruta al checkpoint .pth del modelo",
    )
    parser.add_argument("--fuente", type=int, default=0, help="Indice de camara")
    return parser


if __name__ == "__main__":
    try:
        cli_args = _build_parser().parse_args()
        main(cli_args)
    except KeyboardInterrupt:
        _limpiar_recursos()
        print("\n  Sesion interrumpida por el usuario.")
    finally:
        _limpiar_recursos()


