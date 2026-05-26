<div align="center">

# Comentarista IA de Videojuegos

</div>

---

Sistema de IA que analiza clips de Skywars de minecraft en `.mp4` y genera comentarios automáticos en voz estilo narrador deportivo, orquestando Visión Artificial, NLP y Síntesis de Voz en una interfaz web con Streamlit.

## Equipo

- Rodriguez Rojo Israel Josue - 22170799
- Samano Machado Kevin Jasiel - 22170815
- Quevedo Castellon Joey Kelvin - 22170777

## Arquitectura elegida

El sistema sigue un pipeline lineal donde cada módulo transforma los datos y los pasa al siguiente:

```
Video .mp4 → Captura → Visión (BLIP) → Heurística → Secuencias
→ Acciones (CLIP) → Narración (NLP) → Voz (MMS-TTS) → Video final (FFmpeg)
```

Se eligió la **Opción B** de visión: modelos preentrenados de Hugging Face (BLIP y CLIP) en lugar de entrenar una CNN propia.

## Modelos de Hugging Face utilizados

| Módulo | Modelo | Tarea |
|---|---|---|
| Visión | `Salesforce/blip-image-captioning-base` | Descripción textual de frames |
| Acciones | `openai/clip-vit-base-patch32` | Clasificación zero-shot de acciones de gameplay |
| Audio | `facebook/mms-tts-spa` | Síntesis de voz en español |

## Descripción de cada archivo

**`app.py`** — Interfaz principal en Streamlit. Carga los modelos, recibe el `.mp4`, ejecuta el pipeline completo al presionar el botón y muestra el video final junto con un timeline de comentarios generados.

**`modules/capture.py`** — Abre el video con OpenCV y extrae frames a la frecuencia configurada (`fps_sample`). Guarda el segundo exacto de cada frame para sincronizar después la voz con el video.

**`modules/vision.py`** — Convierte cada frame a imagen PIL y lo pasa a BLIP, que devuelve una descripción en inglés de lo que ve en pantalla, por ejemplo `"a minecraft player fighting on a bridge"`.

**`modules/heuristic.py`** — Compara captions consecutivos con `SequenceMatcher` y decide cuáles merecen comentario. Aplica cooldown para no comentar demasiado seguido y fuerza comentario inmediato si detecta palabras críticas como `kill`, `victory` o `void`.

**`modules/commentary_pipeline.py`** — Agrupa los eventos seleccionados en secuencias con contexto temporal. Limpia captions repetidas, detecta la fase de la partida (inicio, medio, cierre) y señales visuales como combate, loot o peligro de vacío.

**`modules/action_vision.py`** — Módulo opcional que usa CLIP para clasificar frames contra etiquetas específicas de gameplay como `"bridging over void"` o `"close pvp combat"`, complementando la lectura de BLIP.

**`modules/narrator.py`** — Recibe el contexto de cada secuencia, detecta la categoría de la escena (combate, puenteo, victoria, eliminación, etc.) y elige una plantilla narrativa en español evitando repetir frases recientes. Produce comentarios cortos como `"Cruce expuesto; un golpe puede mandarlo al vacío"`.

**`modules/tts.py`** — Convierte cada comentario en texto a un archivo `.wav` usando MMS-TTS en español. Calcula en qué segundo debe sonar cada audio dentro del video para evitar solapamientos.

**`modules/video_mixer.py`** — Usa FFmpeg para retrasar cada `.wav` al segundo correcto, mezclar las voces con el audio original del juego y exportar el `.mp4` final comentado.

**`requirements.txt`** — Dependencias Python del proyecto.

## Cómo ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

> FFmpeg y FFprobe deben estar instalados y disponibles en el PATH del sistema.
