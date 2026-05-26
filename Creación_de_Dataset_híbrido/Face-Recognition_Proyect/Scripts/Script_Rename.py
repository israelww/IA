from __future__ import annotations

import argparse
import sys
from pathlib import Path


EXTENSIONES_IMAGEN = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
    ".heic",
}

# ------------------------------------------------------------
# CONFIGURACION
# ------------------------------------------------------------
CARPETA_OBJETIVO = Path(
    r"C:\Users\joeyk\OneDrive\Desktop\CNN\Face-recognition-proyect\Face-Recognition Proyect\Dataset\Villanos\Joker"
)
NOMBRE_BASE_OBJETIVO = "Lara_Croft"
INICIO_OBJETIVO = 1
SIMULAR_CAMBIOS = False  # True = solo mostrar, False = renombrar


def obtener_imagenes(carpeta: Path) -> list[Path]:
    archivos = [
        p
        for p in carpeta.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONES_IMAGEN
    ]
    return sorted(archivos, key=lambda p: p.name.lower())


def renombrar_imagenes(carpeta: Path, nombre_base: str, inicio: int, simular: bool) -> None:
    imagenes = obtener_imagenes(carpeta)
    if not imagenes:
        print("No se encontraron imagenes para renombrar.")
        return

    # Paso 1: mover a nombres temporales para evitar conflictos.
    temporales: list[Path] = []
    for i, imagen in enumerate(imagenes, start=1):
        temp = carpeta / f"__tmp_renombre_{i:06d}{imagen.suffix.lower()}"
        temporales.append(temp)
        if simular:
            print(f"[SIMULACION] {imagen.name} -> {temp.name}")
        else:
            imagen.rename(temp)

    # Paso 2: nombre final (ej. imagen_1.png).
    for i, temp in enumerate(temporales, start=inicio):
        nuevo_nombre = f"{nombre_base}_{i}{temp.suffix.lower()}"
        destino = carpeta / nuevo_nombre
        if simular:
            print(f"[SIMULACION] {temp.name} -> {destino.name}")
        else:
            temp.rename(destino)
            print(f"{temp.name} -> {destino.name}")

    if not simular:
        print(f"\nListo. Se renombraron {len(temporales)} imagen(es).")


def validar_entrada(carpeta: Path, inicio: int, nombre_base: str) -> None:
    if not carpeta.exists() or not carpeta.is_dir():
        raise SystemExit(f"La carpeta no existe o no es valida: {carpeta}")
    if inicio < 0:
        raise SystemExit("El valor de inicio no puede ser negativo.")
    if not nombre_base.strip():
        raise SystemExit("El nombre base no puede estar vacio.")


def ejecutar_desde_variables() -> None:
    carpeta = CARPETA_OBJETIVO.resolve()
    nombre_base = NOMBRE_BASE_OBJETIVO.strip()
    inicio = INICIO_OBJETIVO
    simular = SIMULAR_CAMBIOS

    validar_entrada(carpeta, inicio, nombre_base)
    renombrar_imagenes(carpeta, nombre_base, inicio, simular)


def ejecutar_desde_argumentos() -> None:
    parser = argparse.ArgumentParser(
        description="Renombra imagenes de una carpeta con formato: nombre_base_1.png, nombre_base_2.jpg, etc."
    )
    parser.add_argument("carpeta", type=Path, help="Ruta de la carpeta que contiene las imagenes.")
    parser.add_argument("nombre_base", help="Nombre base deseado. Ejemplo: imagen")
    parser.add_argument(
        "--inicio",
        type=int,
        default=1,
        help="Numero inicial de la secuencia (por defecto: 1).",
    )
    parser.add_argument(
        "--simular",
        action="store_true",
        help="Muestra los cambios sin renombrar archivos realmente.",
    )
    args = parser.parse_args()

    carpeta = args.carpeta.resolve()
    nombre_base = args.nombre_base.strip()
    validar_entrada(carpeta, args.inicio, nombre_base)
    renombrar_imagenes(carpeta, nombre_base, args.inicio, args.simular)


if __name__ == "__main__":
    # Si se pasan argumentos: modo terminal.
    # Si no se pasan argumentos: modo variables de este archivo.
    if len(sys.argv) > 1:
        ejecutar_desde_argumentos()
    else:
        ejecutar_desde_variables()
