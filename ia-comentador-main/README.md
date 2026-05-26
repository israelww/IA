# AI Game Commentator

## 1. Título y descripción general

**Nombre del proyecto:** AI Game Commentator

AI Game Commentator es un sistema de inteligencia artificial que analiza clips de videojuegos en formato `.mp4`, detecta lo que ocurre en cada escena mediante visión computacional, decide cuando aparece un momento suficientemente relevante para ser comentado, genera un comentario creativo con estilo de narrador deportivo y convierte ese comentario en voz sintética. Todo el flujo está orquestado desde una interfaz web interactiva construida con Streamlit, donde el usuario puede subir un video, ajustar parámetros del análisis, ejecutar el pipeline completo y revisar el resultado final.

La idea central del proyecto es transformar un video de gameplay en una pieza narrada automáticamente. En lugar de limitarse a extraer imágenes o describir frames individuales, el sistema intenta construir una lectura de la partida: identifica cambios visuales, agrupa eventos cercanos, resume el contexto, categoriza el tipo de acción y produce frases cortas que suenan como comentarios de caster. En la versión actual del repositorio, la aplicación está especialmente orientada a Minecraft Skywars, con categorías como inicio, loot, puenteo, centro, combate, eliminación, peligro de vacío, cierre y victoria. Aún así, la arquitectura modular permite adaptar el proyecto a otros juegos cambiando prompts, etiquetas de acción, heurísticas y plantillas narrativas.

El resultado final es un video `.mp4` comentado automáticamente. El sistema conserva el video original, genera audios `.wav` para cada comentario, calcula en que segundo debe sonar cada narración y mezcla la voz del comentarista con el audio original del juego usando FFmpeg. Además, la interfaz muestra un timeline de comentarios para inspeccionar que vio el sistema, que categoría asigno, que texto genero y que audio produjo.

Este proyecto combina tres ramas principales de la inteligencia artificial:

- **Visión Artificial:** analiza frames del video y convierte información visual en texto o etiquetas de acción. El modelo principal es BLIP para image captioning. Opcionalmente se usa CLIP para clasificación zero-shot de acciones específicas de gameplay.
- **Procesamiento de Lenguaje Natural, NLP:** interpreta captions, limpia ruido, detecta palabras clave, clasifica escenas, arma contexto narrativo y genera comentarios en español. En el código actual, esta parte se implementa con heurísticas, categorías ponderadas, plantillas y memoria de frases recientes, no con un modelo generativo BART cargado desde HuggingFace.
- **Síntesis de Voz, TTS:** convierte los comentarios en audio usando un modelo MMS-TTS en español. Cada comentario se guarda como `.wav` y luego se sincroniza con el video.

También se combinan herramientas clasicas de multimedia y programacion: OpenCV para leer videos y extraer frames, NumPy para representar imágenes, PyTorch y Transformers para ejecutar modelos, SciPy para leer y escribir WAV, y FFmpeg/FFprobe para mezclar audio y video.

## 2. Arquitectura general del sistema

### Diagrama ASCII del flujo completo

```text
+-------------------+
| Video .mp4        |
| subido por usuario|
+---------+---------+
          |
          | ruta temporal en disco: temp_video.mp4
          v
+-------------------+
| Captura           |
| modules/capture.py|
+---------+---------+
          |
          | list[dict]
          | [{ sample_idx, source_frame_idx, time_s, frame: np.ndarray RGB }, ...]
          v
+-------------------+
| Visión BLIP       |
| modules/vision.py |
+---------+---------+
          |
          | list[dict]
          | [{ frame_idx, source_frame_idx, time_s, caption: str }, ...]
          v
+-------------------+
| Heuristica        |
| modules/heuristic.py
+---------+---------+
          |
          | list[dict]
          | [{ frame_idx, time_s, caption, reason }, ...]
          v
+-------------------------------+
| Secuencias con contexto       |
| modules/commentary_pipeline.py|
+---------+---------------------+
          |
          | list[dict]
          | [{ sequence_id, start_time_s, end_time_s, key_frame_idx,
          |    captions, context_caption, context_summary_es, phase_hint }, ...]
          v
+-------------------------------+
| Visión de acciones opcional  |
| modules/action_vision.py      |
+---------+---------------------+
          |
          | list[dict]
          | [{ ..., action_tags: [{ label, alias, score }] }, ...]
          v
+-------------------------------+
| Narración NLP                 |
| modules/narrator.py           |
+---------+---------------------+
          |
          | list[dict]
          | [{ ..., comment: str, category: str }, ...]
          v
+-------------------+
| TTS MMS           |
| modules/tts.py    |
+---------+---------+
          |
          | list[dict]
          | [{ ..., audio_path: .wav, audio_duration_s,
          |    narration_time_s, narration_end_s, narration_delay_s }, ...]
          v
+-------------------------------+
| Mezcla multimedia             |
| modules/video_mixer.py        |
+---------+---------------------+
          |
          | archivo .mp4 final con narración mezclada
          v
+-------------------------------+
| Interfaz Streamlit            |
| app.py                        |
+-------------------------------+
```

### Explicación de cada flecha

**Video `.mp4` -> Captura**

El usuario selecciona un archivo `.mp4` en el `file_uploader` de Streamlit. La aplicación escribe los bytes recibidos en `temp_video.mp4`. Lo que viaja hacia el módulo de captura es una ruta de archivo:

```python
"temp_video.mp4"
```

Este diseno simplifica la integracion con OpenCV y FFmpeg, ya que ambas herramientas trabajan de forma natural con archivos en disco.

**Captura -> Visión BLIP**

`modules/capture.py` abre el video con OpenCV, calcula los FPS reales, decide que frames muestrear según `fps_sample`, convierte cada frame de BGR a RGB y opcionalmente lo redimensiona. La salida es una lista de diccionarios:

```python
{
    "sample_idx": 0,
    "source_frame_idx": 0,
    "time_s": 0.0,
    "frame": np.ndarray
}
```

El campo `frame` contiene una imagen RGB como array NumPy con forma aproximada `(alto, ancho, 3)`. Los metadatos `source_frame_idx` y `time_s` permiten sincronizar más tarde los comentarios con el video original.

**Visión BLIP -> Heuristica**

`modules/vision.py` recibe los frames, convierte cada array NumPy en una imagen PIL y usa BLIP para generar una descripción textual. La salida conserva los metadatos y agrega `caption`:

```python
{
    "frame_idx": 5,
    "source_frame_idx": 150,
    "time_s": 5.0,
    "caption": "a minecraft player fighting on a bridge"
}
```

Aqui ocurre una transformacion importante: la información visual se convierte en texto. Esto permite que los módulos posteriores trabajen con similitud textual, palabras clave, categorías y plantillas narrativas.

**Heuristica -> Secuencias con contexto**

`modules/heuristic.py` recibe captions por frame y selecciona momentos relevantes. Compara la caption actual con la ultima caption seleccionada usando `SequenceMatcher`, aplica un cooldown y fuerza eventos criticos si aparecen palabras como `victory`, `kill`, `void`, `falling`, `top 2` o `1v1`. La salida agrega una razon:

```python
{
    "frame_idx": 12,
    "time_s": 8.0,
    "caption": "...",
    "reason": "cambio de acción"
}
```

La lista resultante es más pequena que la lista de frames analizados. Su objetivo es evitar comentar frames repetitivos.

**Secuencias con contexto -> Visión de acciones opcional**

`modules/commentary_pipeline.py` no narra cada frame aislado. Toma los eventos seleccionados y crea ventanas con frames anteriores y posteriores. Luego limpia captions, deduplica texto, descarta ruido, detecta senales visuales y calcula la fase temporal del video. Cada secuencia puede incluir:

```python
{
    "sequence_id": 0,
    "start_frame_idx": 0,
    "end_frame_idx": 6,
    "key_frame_idx": 3,
    "key_time_s": 1.0,
    "start_time_s": 0.0,
    "end_time_s": 2.0,
    "captions": ["caption 1", "caption 2"],
    "context_caption": "Secuencia de skywars...",
    "reason": "inicio de la partida",
    "has_visual_context": True,
    "visual_signals": ["loot", "movilidad"],
    "phase_hint": "inicio",
    "progress_pct": 0,
    "context_summary_es": "Fase: inicio de partida..."
}
```

Si CLIP está disponible, `modules/action_vision.py` clasifica frames cercanos al frame clave y devuelve etiquetas de acción:

```python
{
    "label": "minecraft player placing blocks to bridge over void",
    "alias": "bridging over void",
    "score": 0.28
}
```

Estas etiquetas se agregan a la secuencia como `action_tags` y enriquecen `context_caption`.

**Visión de acciones opcional -> Narración NLP**

`modules/narrator.py` recibe el contexto textual de la secuencia, detecta categoría, selecciona plantillas y genera un comentario en español. La salida agrega:

```python
{
    "comment": "Cruce expuesto; un golpe puede mandarlo al vacío.",
    "category": "puenteo"
}
```

La categoría permite mostrar el tipo de momento en la interfaz y aplicar un estilo narrativo consistente.

**Narración NLP -> TTS MMS**

`modules/tts.py` recibe strings de comentario, los limpia, limita longitud, calcula velocidad de habla y genera audios `.wav`. El resultado se agrega al timeline:

```python
{
    "audio_path": "outputs/audio_timeline/sequence_003.wav",
    "audio_duration_s": 2.31,
    "narration_time_s": 14.2,
    "narration_end_s": 16.51,
    "narration_delay_s": 0.0
}
```

**TTS MMS -> Mezcla multimedia**

`modules/video_mixer.py` recibe el video original y el timeline con audios. Usa FFmpeg para retrasar cada WAV con `adelay`, ajustar volumen, mezclar voces entre si y mezclar la voz final con el audio del juego. Si el video no tiene audio, genera una pista silenciosa base. La salida es un `.mp4` final.

**Mezcla multimedia -> Interfaz Streamlit**

`app.py` muestra el video final con `st.video`, muestra cada secuencia del timeline, enseña el frame clave, el contexto detectado, las acciones probables, el comentario generado y un reproductor `st.audio` para escuchar cada WAV individual.

## 3. Descripción detallada de cada módulo

### `app.py`: interfaz principal y orquestador

**Qué hace**

`app.py` es el punto de entrada de la aplicación. Construye la interfaz Streamlit, carga los modelos, define los controles del sidebar, recibe el archivo `.mp4`, ejecuta el pipeline completo y muestra el video final junto con el timeline de comentarios.

**Cómo funciona internamente**

Al iniciar, configura la pagina con `st.set_page_config`, muestra título y descripción, y carga modelos dentro de `st.session_state`:

- `VisionModel`, para BLIP.
- `NarratorModel`, para comentarios en español.
- `TTSModel`, para voz sintética.
- `ActionVisionModel`, para acciones visuales con CLIP.

El uso de `st.session_state` evita recargar modelos pesados en cada cambio de la interfaz. Esto es importante porque BLIP, CLIP y MMS-TTS pueden tardar bastante en inicializarse.

Cuando el usuario sube un archivo, la app lo guarda como `temp_video.mp4` y lo muestra con `st.video`. El pipeline solo se ejecuta al presionar `Analizar y Renderizar Video Narrado`.

Orden real de ejecución:

1. Extrae frames con `extract_sampled_frames`.
2. Analiza frames con BLIP mediante `analyze_frames`.
3. Filtra eventos con `filter_frames_to_comment`.
4. Construye secuencias con `build_commentary_sequences`.
5. Enriquece con acciones visuales usando `enrich_sequences_with_action_tags`.
6. Genera narración con `narrate_sequences`.
7. Sintetiza voces con `synthesize_sequence_comments`.
8. Mezcla audio y video con `overlay_commentary_on_video`.
9. Muestra el video final y el timeline.

**Input**

- Archivo `.mp4` subido desde la interfaz.
- Valores de sliders y selectbox.
- Modelos cargados en memoria.

**Output**

- Video final en `outputs/final/`.
- Audios WAV en `outputs/audio_timeline/`.
- Timeline visual en la interfaz.
- Mensajes de progreso, exito, advertencia o error.

**Decisiones técnicas y por que**

- **Streamlit:** permite una demo web rápida sin construir frontend separado.
- **`st.session_state`:** reduce latencia al no recargar modelos.
- **`temp_video.mp4`:** facilita integracion con OpenCV y FFmpeg.
- **Sliders:** permiten ajustar sensibilidad y ritmo sin tocar código.
- **Spinners:** hacen visible el progreso de operaciónes lentas.
- **Manejo global de excepciones:** evita que la app muera sin explicar el error.

### `modules/capture.py`: captura y muestreo de frames

**Qué hace**

Extrae frames representativos de un video `.mp4` y conserva metadatos temporales para sincronizar después la narración.

**Funciones**

- `extract_frames(video_path, fps_sample=2)`: devuelve solo frames RGB.
- `extract_sampled_frames(video_path, fps_sample=2, resize_to=None)`: devuelve frames y metadatos.

**Cómo funciona internamente**

El módulo abre el video con `cv2.VideoCapture`, lee los FPS reales con `cv2.CAP_PROP_FPS`, calcula `frame_interval = int(video_fps / fps_sample)` y recorre el video. Cuando el indice del frame cumple el intervalo, convierte de BGR a RGB y opcionalmente redimensiona.

**Input**

```python
video_path: str
fps_sample: int
resize_to: tuple[int, int] | None
```

**Output**

```python
list[dict]
```

Ejemplo:

```python
{
    "sample_idx": 0,
    "source_frame_idx": 0,
    "time_s": 0.0,
    "frame": np.ndarray
}
```

**Decisiones técnicas y por que**

- **Muestreo por FPS:** evita analizar cada frame del video, lo cual seria muy costoso.
- **RGB:** PIL y los modelos de vision esperan imágenes RGB, no BGR.
- **Metadatos de tiempo:** son imprescindibles para colocar la voz en el segundo correcto.
- **Redimension opcional:** permite acelerar el análisis en CPU o con videos grandes.
- **Validaciones:** si el video no se abre o no tiene FPS válido, se lanza un error claro.

### `modules/vision.py`: visión artificial con BLIP

**Qué hace**

Convierte frames RGB en captions textuales usando el modelo `Salesforce/blip-image-captioning-base`.

**Clases y funciones**

- `VisionModel`: carga procesador y modelo BLIP.
- `describe_frame(frame)`: genera una caption para un frame.
- `analyze_frames(frames, model)`: analiza una lista de frames o diccionarios con frames.

**Cómo funciona internamente**

`VisionModel` detecta si hay CUDA. Si hay GPU, usa `torch.float16`, `torch.autocast` y optimizaciones de CUDA. Si no hay GPU, ejecuta en CPU. Cada frame se convierte a `PIL.Image` y se procesa con `BlipProcessor`. Luego `BlipForConditionalGeneration.generate` produce texto.

El prompt usado para orientar al modelo es:

```text
minecraft skywars first person gameplay showing pvp action, items, movement and map context:
```

Parámetros de generación relevantes:

- `max_new_tokens=34`, para captions cortas.
- `num_beams=4`, para mejorar busqueda.
- `repetition_penalty=1.2`, para reducir repeticion.
- `length_penalty=0.95`, para controlar longitud.
- `no_repeat_ngram_size=3`, para evitar n-gramas repetidos.

**Input**

```python
list[np.ndarray]
```

o:

```python
list[dict]
```

con `frame`, `sample_idx`, `source_frame_idx` y `time_s`.

**Output**

```python
list[dict]
```

Ejemplo:

```python
{
    "frame_idx": 3,
    "source_frame_idx": 90,
    "time_s": 3.0,
    "caption": "a minecraft player standing on a bridge"
}
```

**Decisiones técnicas y por que**

- **BLIP:** permite image captioning sin entrenar desde cero.
- **Prompt de dominio:** intenta guiar al modelo hacia PvP, items, movimiento y mapa.
- **Inferencia sin gradientes:** `torch.inference_mode()` reduce memoria y acelera.
- **Fallback por frame:** si falla un frame, se marca como `error al procesar frame` y el pipeline sigue.

### `modules/heuristic.py`: detección de eventos comentables

**Qué hace**

Selecciona los frames que parecen importantes para narrar. Reduce la lista completa de captions a una lista de eventos.

**Funcion principal**

```python
filter_frames_to_comment(analyzed_frames, cooldown_frames=4, similarity_threshold=0.7)
```

**Cómo funciona internamente**

El módulo compara cada caption con la ultima caption seleccionada usando `SequenceMatcher`. También revisa palabras clave criticas:

```text
victory, winner, you win, kill, killed, eliminated, void, falling, fell,
endgame, top 2, top 3, 1v1
```

Reglas principales:

- El primer frame se selecciona como `inicio de la partida`.
- Si aparece una palabra clave critica y ya paso más de un frame, se selecciona como `evento clave detectado`.
- Si la similitud baja mucho, se selecciona como `cambio de acción`.
- Si paso el cooldown y la escena cambio lo suficiente, se selecciona como `nuevo momento detectado`.

**Input**

```python
analyzed_frames: list[dict]
cooldown_frames: int
similarity_threshold: float
```

**Output**

```python
list[dict]
```

Ejemplo:

```python
{
    "frame_idx": 8,
    "time_s": 4.0,
    "caption": "...",
    "reason": "nuevo momento detectado"
}
```

**Decisiones técnicas y por que**

- **Similitud textual:** es rápida y suficiente para detectar muchos cambios de escena.
- **Cooldown:** evita que el narrador hable demasiado seguido.
- **Eventos forzados:** una kill, una caida o una victoria son importantes aunque ocurran cerca de otro evento.
- **Umbral configurable:** distintos juegos tienen ritmos distintos.

### `modules/commentary_pipeline.py`: secuencias narrables y contexto

**Qué hace**

Convierte eventos sueltos en secuencias narrables con contexto temporal, captions limpias, resumen visual, fase del video y acciones opcionales.

**Funciones principales**

- `build_commentary_sequences(...)`
- `enrich_sequences_with_action_tags(...)`
- `narrate_sequences(...)`

**Cómo funciona internamente**

Primero indexa los frames por `frame_idx`. Luego toma los eventos seleccionados y crea ventanas usando `context_before` y `context_after`. Si varias ventanas se tocan o se solapan, se fusionan. Esto evita que el sistema genere varios comentarios para la misma jugada.

Si hay pocos eventos, inserta anclajes temporales repartidos en el video para asegurar una narración mínima. Esta decision hace que el resultado no quede silencioso cuando BLIP produce captions repetitivas o ambiguas.

Limpieza de captions:

- Elimina strings vacíos.
- Descarta `error al procesar frame`.
- Filtra captions con patrones de ruido.
- Deduplica captions similares.
- Quita afirmaciones poco confiables sobre arco o flechas cuando podrian ensuciar la narración.

Contexto generado:

- `captions`: captions útiles de la secuencia.
- `context_caption`: texto compacto que se entrega al narrador.
- `context_summary_es`: resumen en español.
- `phase_hint`: `inicio`, `medio` o `cierre`.
- `progress_pct`: porcentaje aproximado del video.
- `visual_signals`: senales como combate, proyectiles, loot, movilidad, riesgo de vacío, curacion, cierre o victoria.

**Input**

```python
analyzed_frames: list[dict]
selected_frames: list[dict]
context_before: int
context_after: int
max_captions_per_sequence: int
max_sequences: int
target_commentary_count: int | None
```

**Output**

```python
list[dict]
```

Ejemplo:

```python
{
    "sequence_id": 4,
    "start_frame_idx": 20,
    "end_frame_idx": 26,
    "key_frame_idx": 23,
    "key_time_s": 7.6,
    "start_time_s": 6.7,
    "end_time_s": 8.7,
    "captions": ["..."],
    "context_caption": "Secuencia de skywars...",
    "reason": "cambio de acción",
    "trigger_count": 1,
    "has_visual_context": True,
    "visual_signals": ["combate", "riesgo_vacío"],
    "phase_hint": "medio",
    "progress_pct": 35,
    "context_summary_es": "Fase: juego medio..."
}
```

**Decisiones técnicas y por que**

- **Secuencias en vez de frames:** los comentarios deportivos necesitan contexto.
- **Ventanas fusionadas:** evitan duplicados.
- **Anclajes temporales:** garantizan narración cuando la detección es pobre.
- **Limpieza agresiva:** reduce captions genericas, repetidas o incorrectas.
- **Fase temporal:** impide que conceptos de cierre o victoria aparezcan demasiado pronto.

### `modules/action_vision.py`: acciones visuales con CLIP

**Qué hace**

Clasifica frames con CLIP zero-shot para detectar acciones específicas de gameplay.

**Clase principal**

```python
ActionVisionModel
```

**Funcion principal**

```python
classify_frame(frame, top_k=4, min_score=0.12)
```

**Cómo funciona internamente**

Carga un pipeline de HuggingFace:

```python
pipeline(
    task="zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
    device=0 or -1
)
```

Compara la imagen contra labels candidatos como:

- `minecraft player fighting another player with sword`
- `minecraft player close melee pvp combat`
- `minecraft player throwing snowballs at enemy`
- `minecraft player eating golden apple`
- `minecraft player opening chest and looting`
- `minecraft player placing blocks to bridge over void`
- `minecraft player near the void edge in danger`
- `minecraft skywars final duel 1v1`
- `minecraft skywars victory screen`

Cada label tiene un alias más corto, por ejemplo `bridging over void` o `close pvp combat`.

**Input**

```python
frame: np.ndarray
top_k: int
min_score: float
```

**Output**

```python
list[dict]
```

Ejemplo:

```python
{
    "label": "minecraft player close melee pvp combat",
    "alias": "close pvp combat",
    "score": 0.24
}
```

**Decisiones técnicas y por que**

- **Zero-shot:** no hace falta entrenar un dataset propio.
- **Labels de dominio:** mejoran la lectura de Minecraft Skywars.
- **Score mínimo:** reduce falsos positivos.
- **Módulo opcional:** si falla, el pipeline sigue funcionando con BLIP.

### `modules/narrator.py`: generación de comentarios estilo caster

**Qué hace**

Genera el comentario final en español y asigna una categoría narrativa. Es el nucleo de NLP del proyecto actual.

**Clase principal**

```python
NarratorModel
```

**Funcion principal**

```python
generate_comment(caption) -> tuple[str, str]
```

También existe `narrate_frames(filtered_frames, model)` para narrar frames individuales, aunque la aplicación actual usa narración por secuencias.

**Cómo funciona internamente**

El narrador usa reglas y plantillas, no un LLM ni BART. Su flujo interno es:

1. Normaliza texto con `únicodedata` para comparar sin depender de acentos.
2. Detecta captions inválidas o de error.
3. Extrae items como espada, snowballs, huevos, pocion o manzana dorada cuando el contexto lo permite.
4. Extrae acciones visuales marcadas por el pipeline, por ejemplo `duelo melee` o `puenteo sobre vacío`.
5. Extrae senales tacticas como combate, movilidad, loot, riesgo de vacío o curacion.
6. Detecta fase temporal: inicio, medio o cierre.
7. Calcula scores por categoría usando keywords y pesos.
8. Selecciona la categoría ganadora.
9. Escoge una plantilla narrativa evitando repetir frases recientes.
10. Agrega detalles o transiciones si aportan ritmo.
11. Compacta el comentario para respetar `max_comment_words`.

Categorías principales:

- `inicio`: spawn, apertura, primeros segundos, preparacion.
- `loot`: cofres, inventario, armadura, espada, recursos.
- `puenteo`: bloques, bridge, cruce entre islas, riesgo de vacío.
- `centro`: mid, centro, control de rutas.
- `combate`: PvP, melee, trades, proyectiles, presion.
- `eliminación`: kill, rival eliminado, baja confirmada.
- `peligro_vacío`: edge, void, falling, knockback, clutch.
- `cierre`: endgame, top 2, top 3, duelo final.
- `victoria`: winner, victory, you win, pantalla final.
- `general`: fallback cuando no hay senal clara.

Ejemplos de comentarios generados:

```text
Cruce expuesto; un golpe puede mandarlo al vacío.
Duelo abierto; el primer combo pesa mucho.
Baja confirmada; se abre el mapa.
Tramo final; cada rotación vale oro.
Win confirmado; ejecución limpia.
```

**Input**

```python
caption: str
```

En la práctica, este `caption` suele ser `context_caption`, que ya contiene resumen visual, captions útiles, fase temporal, progreso y acciones probables.

**Output**

```python
(comment: str, category: str)
```

Ejemplo:

```python
("Puenteo con presion, cada bloque cuenta.", "puenteo")
```

**Decisiones técnicas y por que**

- **Plantillas controladas:** evitan frases demasiado largas o alucinadas.
- **Pesos por categoría:** priorizan eventos importantes como victoria, eliminación o peligro de vacío.
- **Memoria de comentarios recientes:** reduce repeticion.
- **Limite de palabras:** mejora sincronización con TTS.
- **Español nativo en plantillas:** evita traducir al final y da más control de estilo.
- **Sin BART actualmente:** mejora velocidad y consistencia para una demo local; un modelo generativo podria integrarse más adelante en `generate_comment`.

### `modules/tts.py`: síntesis de voz con MMS-TTS

**Qué hace**

Convierte comentarios de texto en archivos de audio `.wav` usando `facebook/mms-tts-spa`.

**Clase principal**

```python
TTSModel
```

**Funciones principales**

- `synthesize(text, output_path, speaking_rate=None)`
- `get_wav_duration_s(audio_path)`
- `synthesize_comments(narrated_frames, model, output_dir="outputs/audio")`
- `synthesize_sequence_comments(narrated_sequences, model, output_dir="outputs/audio_timeline", min_spacing_s=0.1, speaking_rate=1.25)`

**Cómo funciona internamente**

`TTSModel` carga `VitsTokenizer` y `VitsModel` desde `facebook/mms-tts-spa`. Detecta si CUDA está disponible y mueve el modelo a GPU o CPU.

Antes de sintetizar, limpia el texto:

- Compacta espacios.
- Corrige espacios antes de signos de puntuacion.
- Limita la cantidad de palabras.
- Agrega punto final si el texto fue recortado.

La velocidad de habla se calcula con `_dynamic_rate`. Si el comentario tiene muchas palabras, el módulo aumenta ligeramente la velocidad para que el audio no se haga demasiado largo.

El waveform producido por el modelo se convierte a NumPy, se recorta al rango `[-1.0, 1.0]`, se convierte a PCM de 16 bits y se guarda con `scipy.io.wavfile.write`.

**Input**

```python
text: str
output_path: str
speaking_rate: float | None
```

Para timeline:

```python
narrated_sequences: list[dict]
model: TTSModel
output_dir: str
min_spacing_s: float
speaking_rate: float
```

**Output**

Archivo WAV y diccionario enriquecido:

```python
{
    "audio_path": "outputs/audio_timeline/sequence_000.wav",
    "audio_duration_s": 2.14,
    "audio_error": None,
    "narration_time_s": 0.0,
    "narration_end_s": 2.14,
    "narration_delay_s": 0.0
}
```

**Decisiones técnicas y por que**

- **MMS-TTS en español:** permite TTS local sin depender de APIs externas.
- **WAV PCM 16-bit:** alta compatibilidad con FFmpeg.
- **Timeline calculado después de medir duración real:** evita superposiciones no deseadas.
- **`min_spacing_s`:** da respiracion entre comentarios.
- **Manejo de errores por secuencia:** una falla de TTS no necesariamente destruye todo el análisis.

### `modules/video_mixer.py`: mezcla final de audio y video

**Qué hace**

Crea el video final mezclando el audio original del gameplay con los audios WAV de narración.

**Funcion principal**

```python
overlay_commentary_on_video(
    input_video_path,
    commentary_timeline,
    output_video_path="outputs/final/commentated_video.mp4",
    game_audio_volume=0.8,
    commentary_volume=1.35,
)
```

**Cómo funciona internamente**

Primero localiza `ffmpeg` con `shutil.which`. Usa `ffprobe` para leer información del video, detectar duración y saber si tiene stream de audio. Luego filtra comentarios válidos, es decir, aquellos con `audio_path` existente.

Si no hay comentarios válidos, copia el video original a la salida. Si el video original no trae audio, crea una pista silenciosa con `anullsrc`.

Para cada comentario:

1. Agrega el WAV como input de FFmpeg.
2. Calcula delay en milisegundos desde `narration_time_s`.
3. Aplica `adelay`.
4. Ajusta volumen con `volume`.
5. Mezcla voces con `amix`.
6. Mezcla voz final con audio del juego.
7. Exporta MP4 con video copiado y audio AAC.

**Input**

```python
input_video_path: str
commentary_timeline: list[dict]
output_video_path: str
game_audio_volume: float
commentary_volume: float
```

**Output**

```python
str
```

Ruta del video final.

**Decisiones técnicas y por que**

- **FFmpeg:** herramienta robusta para audio/video.
- **FFprobe:** inspeccion precisa de streams y duración.
- **`adelay`:** sincroniza comentarios con la línea de tiempo.
- **`amix`:** mezcla varias pistas.
- **`-c:v copy`:** evita recodificar video, ahorra tiempo y mantiene calidad.
- **AAC 192 kbps:** compatible con MP4 y navegadores.
- **Pista silenciosa si no hay audio:** mantiene el pipeline funcionando con videos mudos.

### `requirements.txt`: dependencias Python

**Qué hace**

Lista las librerías Python necesarias:

```text
streamlit>=1.30
opencv-python>=4.8
numpy>=1.24
transformers>=4.38
torch>=2.0
scipy>=1.11
Pillow>=10.0
mss>=9.0
```

**Decisiones técnicas y por que**

- Versiónes mínimas modernas para APIs estables.
- `torch` y `transformers` son esenciales para BLIP, CLIP y MMS.
- `mss` está instalado para posibles flujos de captura de pantalla, aunque el flujo actual usa subida de video.

### `setup_gpu.ps1`: preparacion opcional de GPU

**Qué hace**

Instala PyTorch con CUDA desde el indice oficial de PyTorch y verifica disponibilidad de GPU.

**Comando principal**

```powershell
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch
```

**Salida esperada**

El script imprime versión de PyTorch, si CUDA está disponible, numero de GPUs y nombre del dispositivo. También hace una operación simple con matrices en GPU para verificar que funciona.

**Decisiones técnicas y por que**

- BLIP, CLIP y MMS-TTS son costosos en CPU.
- CUDA puede reducir mucho el tiempo de análisis.
- Separar esto en un script evita complicar `app.py`.

## 4. Modelos de HuggingFace utilizados

### `Salesforce/blip-image-captioning-base`

**Módulo:** `modules/vision.py`

**Tipo:** image captioning múltimodal.

**Clases usadas:**

- `BlipProcessor`
- `BlipForConditionalGeneration`

**Funcion en el proyecto**

Genera captions para frames del video. Es el puente entre la imagen y el texto que después procesan las heurísticas y el narrador.

**Entrada**

- Imagen PIL creada desde `np.ndarray` RGB.
- Prompt textual especializado en Minecraft Skywars.

**Salida**

- String con descripción de la escena.

**Ventajas**

- No requiere entrenamiento propio.
- Compatible con CPU y GPU.
- Facil de integrar con Transformers.

**Limitaciones**

- Puede producir captions genericas.
- Puede equivocarse con items pequenos o HUD.
- Analiza frames individuales, no entiende por si solo la secuencia temporal.

### `openai/clip-vit-base-patch32`

**Módulo:** `modules/action_vision.py`

**Tipo:** CLIP para zero-shot image classification.

**Interfaz usada:**

```python
transformers.pipeline(task="zero-shot-image-classification")
```

**Funcion en el proyecto**

Clasifica frames contra etiquetas candidatas de acciones de Skywars. Complementa a BLIP cuando se necesita una lectura más concreta.

**Entrada**

- Imagen PIL.
- Lista de labels candidatos.
- Plantilla de hipotesis: `This is a screenshot of {}.`

**Salida**

- Lista de labels con score, alias y texto original.

**Ventajas**

- No necesita dataset entrenado.
- Labels faciles de modificar.
- Detecta acciones más específicas que una caption libre.

**Limitaciones**

- Los scores no son probabilidades perfectas.
- Puede confundir acciones visualmente parecidas.
- Requiere umbrales y filtros temporales.

### `facebook/mms-tts-spa`

**Módulo:** `modules/tts.py`

**Tipo:** text-to-speech en español basado en VITS.

**Clases usadas:**

- `VitsTokenizer`
- `VitsModel`

**Funcion en el proyecto**

Convierte comentarios en español en waveform de audio y después en archivos `.wav`.

**Entrada**

- Texto en español.
- Velocidad de narración opcional.

**Salida**

- Waveform.
- Archivo WAV en disco.

**Ventajas**

- Funciona localmente.
- No requiere API externa.
- Esta orientado a español.
- Se integra con PyTorch.

**Limitaciones**

- Puede pronunciar raro algunos terminos de videojuegos.
- Comentarios largos pueden sonar menos naturales.
- Requiere postprocesamiento para guardar WAV compatible.

### Nota sobre BART u otros modelos generativos

La arquitectura conceptual podria usar BART, T5, Llama, Mistral u otro modelo generativo para redactar comentarios. Sin embargo, este repositorio no carga actualmente BART. La generación narrativa está implementada en `modules/narrator.py` con reglas y plantillas. Esto reduce consumo de memoria, baja la latencia, evita alucinaciones y mantiene los comentarios cortos para TTS.

Si se quisiera incorporar BART en el futuro, el punto natural de integracion seria reemplazar o complementar `NarratorModel.generate_comment`.

## 5. Librerías utilizadas

### Streamlit

Se usa para construir toda la interfaz web. Permite subir archivos, configurar parámetros, ejecutar el pipeline con un boton y visualizar resultados sin crear una aplicación frontend separada.

Usos principales:

- `st.set_page_config`, configuracion de pagina.
- `st.title` y `st.caption`, encabezado de la app.
- `st.sidebar`, panel de configuracion.
- `st.slider`, controles numericos.
- `st.selectbox`, seleccion de resolución.
- `st.file_uploader`, carga del `.mp4`.
- `st.video`, vista previa y video final.
- `st.button`, ejecución del pipeline.
- `st.spinner`, estados de carga.
- `st.success`, `st.warning`, `st.error`, feedback al usuario.
- `st.container`, `st.columns`, `st.image`, `st.audio`, timeline visual.

### OpenCV, `opencv-python`

Se usa para abrir videos y extraer frames.

Usos principales:

- `cv2.VideoCapture`, lectura del video.
- `cv2.CAP_PROP_FPS`, obtencion de FPS.
- `cv2.cvtColor`, conversión BGR a RGB.
- `cv2.resize`, redimension opcional.

### NumPy

Se usa para representar frames como arrays y para procesar waveforms de audio. Los frames viajan por el pipeline como `np.ndarray` hasta que se convierten a imagen PIL para BLIP o CLIP.

### PyTorch

Se usa para ejecutar modelos de HuggingFace, detectar CUDA, mover modelos a GPU o CPU, ejecutar inferencia sin gradientes y usar optimizaciones como `float16` en GPU.

Usos principales:

- `torch.cuda.is_available()`.
- `torch.device("cuda" if ... else "cpu")`.
- `torch.inference_mode()`.
- `torch.autocast(...)`.
- Tensores enviados a GPU con `.to(self.device)`.

### HuggingFace Transformers

Se usa para cargar los modelos de vision, clasificación y TTS.

Componentes usados:

- `BlipProcessor`.
- `BlipForConditionalGeneration`.
- `pipeline` para CLIP zero-shot.
- `VitsTokenizer`.
- `VitsModel`.

### Pillow

Se usa para convertir arrays NumPy en imágenes PIL mediante `Image.fromarray(frame)`, formato compatible con los modelos de vision.

### SciPy

Se usa para escribir y leer archivos WAV.

Componentes usados:

- `scipy.io.wavfile.write`, guardar audio sintético.
- `scipy.io.wavfile.read`, calcular duración real de WAV.

### FFmpeg y FFprobe

Son dependencias del sistema, no paquetes de Python. Deben estar instaladas y disponibles en el `PATH`.

FFprobe se usa para inspeccionar el video:

- Duración total.
- Streams disponibles.
- Existencia de audio original.

FFmpeg se usa para renderizar:

- Agregar WAVs como inputs.
- Retrasar comentarios con `adelay`.
- Ajustar volumen con `volume`.
- Mezclar pistas con `amix`.
- Exportar MP4 final con audio AAC.

### pathlib

Se usa para manejar rutas y crear carpetas de salida como `outputs/audio_timeline` y `outputs/final`.

### subprocess

Se usa para ejecutar comandos `ffmpeg` y `ffprobe` desde Python.

### json

Se usa para interpretar la salida JSON de FFprobe.

### shutil

Se usa para localizar `ffmpeg` en el sistema mediante `shutil.which`.

### difflib.SequenceMatcher

Se usa para comparar captions y medir si una escena cambio lo suficiente como para justificar otro comentario.

### re

Se usa para limpieza de texto, extracción de marcadores, busqueda de patrones y compactacion de comentarios.

### únicodedata

Se usa para normalizar texto y comparar keywords sin que los acentos afecten el resultado.

### dataclasses

Se usa para definir `SceneContext`, una estructura que agrupa información semantica de la escena.

### random

Se usa para variar plantillas narrativas y evitar que el narrador suene siempre igual.

### collections.Counter

Se usa para encontrar la razon dominante cuando una secuencia contiene varios triggers.

### mss

Aparece en `requirements.txt`. En el flujo actual de la app, basado en subir `.mp4`, no es un componente central. Puede servir para una futura versión con captura de pantalla en vivo.

## 6. Estructura del proyecto

Estructura esperada del proyecto:

```text
ai-commentator/
|-- README.md
|-- app.py
|-- requirements.txt
|-- setup_gpu.ps1
|-- temp_video.mp4
|-- modules/
|   |-- action_vision.py
|   |-- capture.py
|   |-- commentary_pipeline.py
|   |-- heuristic.py
|   |-- narrator.py
|   |-- tts.py
|   |-- video_mixer.py
|   |-- vision.py
|-- outputs/
|   |-- audio_timeline/
|   |   |-- sequence_000.wav
|   |   |-- sequence_001.wav
|   |   |-- ...
|   |-- final/
|   |   |-- commentated_YYYYMMDD_HHMMSS.mp4
|-- __pycache__/
|-- modules/__pycache__/
|-- .gitignore
```

### `README.md`

Documento principal del proyecto. Explica proposito, arquitectura, módulos, modelos, librerías, estructura y uso de la interfaz.

### `app.py`

Aplicación Streamlit. Es el orquestador de todo el sistema y el archivo que se ejecuta con `streamlit run app.py`.

### `requirements.txt`

Dependencias Python necesarias para instalar el entorno.

### `setup_gpu.ps1`

Script opcional de PowerShell para instalar PyTorch con CUDA y verificar GPU.

### `temp_video.mp4`

Archivo temporal usado por la app para guardar el video subido. Puede sobrescribirse en cada ejecución.

### `modules/capture.py`

Extracción de frames y metadatos temporales.

### `modules/vision.py`

Análisis visual con BLIP y generación de captions.

### `modules/heuristic.py`

Seleccion de momentos relevantes mediante similitud textual, cooldown y palabras clave.

### `modules/commentary_pipeline.py`

Construccion de secuencias, limpieza de captions, contexto, fases temporales, acciones y llamada al narrador.

### `modules/action_vision.py`

Clasificación opcional de acciones visuales con CLIP.

### `modules/narrator.py`

Generación de comentarios en español con categorías, pesos, plantillas y memoria.

### `modules/tts.py`

Síntesis de voz con MMS-TTS y construccion del timeline de audios.

### `modules/video_mixer.py`

Mezcla de audio original y narración con FFmpeg para crear el MP4 final.

### `outputs/audio_timeline/`

Carpeta de audios WAV generados por comentario. Los nombres siguen el formato:

```text
sequence_000.wav
sequence_001.wav
sequence_002.wav
```

### `outputs/final/`

Carpeta de videos finales renderizados. Los nombres siguen el formato:

```text
commentated_20260524_191314.mp4
```

### `__pycache__/`

Carpetas generadas automáticamente por Python. No forman parte de la logica fuente.

### `.gitignore`

Archivo para excluir caches, salidas temporales y artefactos pesados del control de versiónes.

## 7. Cómo usar la interfaz

### Preparacion inicial

1. Instala dependencias Python:

```powershell
pip install -r requirements.txt
```

2. Verifica que FFmpeg y FFprobe existan en el `PATH`:

```powershell
ffmpeg -versión
ffprobe -versión
```

3. Si tienes GPU NVIDIA, prepara PyTorch con CUDA de forma opcional:

```powershell
.\setup_gpu.ps1
```

4. Ejecuta Streamlit:

```powershell
streamlit run app.py
```

5. Abre la URL local que Streamlit indique, normalmente:

```text
http://localhost:8501
```

### Guia detallada de uso

1. **Estado de GPU**

   En el sidebar, la app indica si hay GPU CUDA activa. Si hay GPU, el procesamiento de BLIP, CLIP y TTS puede ser mucho más rápido. Si no hay GPU, la aplicación funciona en CPU, pero puede tardar bastante con videos largos o muchos frames.

2. **Detector visual de acciones**

   La app intenta cargar CLIP como detector opcional. Si no está disponible, aparece una advertencia y el sistema usa solo captions de BLIP. Esto no rompe el flujo, solo reduce el detalle de acciones específicas.

3. **File uploader: `Sube un video .mp4 de Skywars`**

   Este componente permite elegir un archivo `.mp4`. Al subirlo, Streamlit lo guarda como `temp_video.mp4` y lo muestra en pantalla.

   Recomendaciones:

   - Para pruebas iniciales, usa clips de 20 a 90 segundos.
   - Para CPU, evita videos largos o resoluciónes muy altas.
   - Para mejores captions, usa clips con buena visibilidad y acción clara.
   - Aunque la app dice Skywars, la arquitectura puede procesar otros videos, pero las categorías y plantillas estan optimizadas para ese tipo de gameplay.

4. **Slider: `Frames por segundo a analizar`**

   Controla cuantos frames por segundo se extraen del video para pasarlos por BLIP.

   Rango en la app: `1` a `8`.

   Efecto:

   - Valor bajo: menos costo, menos detalle temporal.
   - Valor alto: más precisión en acciones rápidas, más tiempo de procesamiento.

   Recomendaciones:

   - Acción rápida: `3` a `6`.
   - Acción muy rápida y GPU disponible: `5` a `8`.
   - Exploracion lenta: `1` a `2`.
   - CPU: comenzar con `1` o `2`.

5. **Slider: `Cooldown entre eventos`**

   Controla cuantos frames muestreados deben pasar antes de aceptar otro evento normal.

   Rango en la app: `1` a `12`.

   Efecto:

   - Valor bajo: más comentarios y mayor sensibilidad.
   - Valor alto: menos comentarios y narración más espaciada.

   Recomendaciones:

   - Acción rápida: `3` a `5`.
   - Partidas caoticas: `2` a `4`.
   - Exploracion lenta: `6` a `10`.
   - Video con demasiada voz: `8` a `12`.

6. **Slider: `Umbral de similitud para cambio de escena`**

   Define que tan diferentes deben ser dos captions para considerar que hay un nuevo momento.

   Rango en la app: `0.0` a `1.0`.

   Efecto:

   - Valor bajo: solo cambios fuertes disparan eventos.
   - Valor alto: cambios pequenos también pueden disparar eventos.

   Recomendaciones:

   - Skywars o acción rápida: `0.65` a `0.80`.
   - Valor inicial razonable: `0.70`.
   - Exploracion lenta: `0.50` a `0.65`.
   - Si hay pocos comentarios, subirlo.
   - Si hay demasiados comentarios, bajarlo.

7. **Slider: `Frames de contexto antes del evento`**

   Define cuantos frames muestreados anteriores al evento se incluyen en la secuencia.

   Rango en la app: `0` a `5`.

   Recomendaciones:

   - Acción rápida: `1` a `2`.
   - Skywars: `2`.
   - Exploracion lenta: `2` a `4`.

8. **Slider: `Frames de contexto después del evento`**

   Define cuantos frames posteriores se incluyen. Sirve para capturar la consecuencia de una acción.

   Rango en la app: `1` a `8`.

   Recomendaciones:

   - Acción rápida: `3` a `5`.
   - Skywars: `4`.
   - Exploracion lenta: `4` a `8`.

9. **Slider: `Máximo de comentarios en timeline`**

   Limita cuantas secuencias narradas pueden aparecer como máximo.

   Rango en la app: `6` a `30`.

   Recomendaciones:

   - Clip de 30 segundos: `6` a `10`.
   - Clip de 1 minuto: `10` a `18`.
   - Clip de 2 minutos o más: `18` a `30`.
   - Exploracion lenta: `6` a `12`.

10. **Slider: `Mínimo de comentarios deseados`**

    Indica cuantos comentarios intenta garantizar el pipeline. Si la heuristica encuentra pocos eventos, `commentary_pipeline.py` agrega anclajes temporales.

    Rango en la app: `3` a `20`.

    Recomendaciones:

    - Acción rápida: `8` a `14`.
    - Skywars corto: `6` a `10`.
    - Exploracion lenta: `3` a `6`.
    - Si el video queda silencioso, subirlo.
    - Si comenta momentos poco importantes, bajarlo.

11. **Slider: `Acciones visuales por secuencia`**

    Controla cuantas etiquetas CLIP se conservan por secuencia.

    Rango en la app: `2` a `6`.

    Recomendaciones:

    - Acción rápida: `3` a `4`.
    - Exploracion lenta: `2` a `3`.
    - Si hay falsos positivos, usar `2`.

12. **Slider: `Ventana temporal de acción (+/- frames)`**

    Define cuantos frames alrededor del frame clave se analizan con CLIP.

    Rango en la app: `1` a `4`.

    Recomendaciones:

    - Acción rápida: `2` a `3`.
    - Exploracion lenta: `3` a `4`.
    - Si CLIP toma acciones de momentos vecinos, reducirlo.

13. **Slider: `Umbral de confianza de acciones`**

    Define el score mínimo para aceptar una acción visual de CLIP.

    Rango en la app: `0.05` a `0.40`.

    Recomendaciones:

    - Skywars: `0.20`.
    - Acción rápida: `0.18` a `0.25`.
    - Exploracion lenta: `0.20` a `0.30`.
    - Si no detecta acciones, bajar a `0.12` o `0.15`.
    - Si detecta acciones incorrectas, subir a `0.25` o `0.30`.

14. **Slider: `Máximo de captions por secuencia`**

    Limita cuantas captions limpias se incluyen en el contexto del narrador.

    Rango en la app: `3` a `10`.

    Recomendaciones:

    - Acción rápida: `5` a `7`.
    - Skywars: `7`.
    - Exploracion lenta: `4` a `6`.
    - Si el narrador parece confundido, reducirlo.

15. **Selectbox: `Resolución para vision`**

    Opciones:

    - `original`.
    - `960x540`.
    - `640x360`.

    Efecto:

    - `original`: mayor detalle, mayor costo.
    - `960x540`: balance entre detalle y velocidad.
    - `640x360`: más rápido, util en CPU.

    Recomendaciones:

    - GPU buena: `original` o `960x540`.
    - CPU: `640x360`.
    - HUD o texto pequeno importante: `original`.
    - Exploracion lenta: `640x360` o `960x540`.

16. **Slider: `Separacion mínima entre comentarios (s)`**

    Controla la distancia mínima entre narraciónes consecutivas.

    Rango en la app: `0.0` a `2.0` segundos.

    Recomendaciones:

    - Acción rápida: `0.05` a `0.30`.
    - Skywars: `0.10`.
    - Exploracion lenta: `0.75` a `1.50`.
    - Si las voces se enciman, subirlo.

17. **Slider: `Velocidad de narración (voz)`**

    Controla la velocidad del TTS.

    Rango en la app: `0.8` a `2.0`.

    Recomendaciones:

    - Acción rápida: `1.25` a `1.45`.
    - Skywars: `1.32`.
    - Exploracion lenta: `0.95` a `1.15`.
    - Si la voz suena atropellada, bajarlo.
    - Si los comentarios no caben, subirlo poco a poco.

18. **Slider: `Palabras máximas por comentario`**

    Controla cuantas palabras puede usar el narrador por comentario.

    Rango en la app: `10` a `32`.

    Recomendaciones:

    - Acción rápida: `12` a `18`.
    - Skywars: `18`.
    - Exploracion lenta: `20` a `28`.
    - Si la voz satura el video, bajarlo.

19. **Slider: `Volumen del juego`**

    Controla el volumen del audio original en la mezcla final.

    Rango en la app: `0.1` a `1.5`.

    Recomendaciones:

    - Gameplay ruidoso: `0.5` a `0.8`.
    - Skywars: `0.8`.
    - Exploracion lenta: `0.8` a `1.1`.
    - Si la voz no se entiende, bajarlo.

20. **Slider: `Volumen del narrador`**

    Controla el volumen de los WAV generados por TTS.

    Rango en la app: `0.5` a `2.5`.

    Recomendaciones:

    - Acción rápida: `1.3` a `1.8`.
    - Skywars: `1.35`.
    - Exploracion lenta: `1.0` a `1.3`.
    - Si distorsiona, bajarlo.
    - Si queda tapado por el juego, subirlo.

21. **Lista de categorías detectadas**

    El sidebar muestra las categorías narrativas disponibles. Sirven para interpretar el timeline:

    - Inicio.
    - Loot.
    - Puenteo.
    - Centro.
    - Combate.
    - Eliminación.
    - Peligro de vacío.
    - Cierre.
    - Victoria.
    - General.

22. **Boton: `Analizar y Renderizar Video Narrado`**

    Este boton inicia todo el pipeline. Al presionarlo, la aplicación ejecuta:

    - Extracción de frames.
    - Análisis con BLIP.
    - Detección de eventos.
    - Construccion de secuencias.
    - Detección opcional de acciones con CLIP.
    - Generación de comentarios.
    - Síntesis de voz.
    - Mezcla final con FFmpeg.
    - Visualizacion del video final y timeline.

23. **Mensajes de progreso**

    Durante la ejecución, la app muestra conteos importantes:

    - `Frames muestreados`: cuantos frames se extrajeron.
    - `Frames analizados`: cuantos frames recibieron caption.
    - `Eventos detectados`: cuantos momentos selecciono la heuristica.
    - `Secuencias narrables`: cuantos bloques se enviaran al narrador.

    Interpretacion:

    - Muchos frames y pocos eventos puede indicar captions repetitivas.
    - Pocos frames puede indicar `fps_sample` demasiado bajo.
    - Muchas secuencias puede producir una narración saturada.
    - Cero audios válidos indica problema en TTS o en la generación de WAV.

24. **Seccion `Video Final Narrado`**

    Cuando termina el render, Streamlit muestra el MP4 final. También se guarda en:

```text
outputs/final/
```

    El nombre se genera con timestamp:

```text
commentated_YYYYMMDD_HHMMSS.mp4
```

25. **Seccion `Timeline de Comentarios`**

    Cada tarjeta del timeline representa una secuencia narrada. Puede incluir:

    - Frame clave.
    - Categoría en mayusculas.
    - Razon del evento.
    - Inicio en video.
    - Momento exacto donde se narra.
    - Duración de la voz.
    - Indicador de contexto visual válido.
    - Contexto detectado.
    - Acciones visuales probables.
    - Comentario generado.
    - Reproductor de audio.
    - Error de audio si ocurrio.

26. **Campo `Inicio en video`**

    Indica el segundo donde comienza visualmente la secuencia detectada. Sale de `start_time_s`.

27. **Campo `Narrado en`**

    Indica el segundo donde se coloco el audio de narración. Puede ser posterior al inicio visual si el sistema tuvo que separar comentarios para evitar solapamiento.

28. **Campo `Duración voz`**

    Muestra la duración real del WAV generado. Si es demasiado larga, conviene reducir palabras máximas o subir ligeramente velocidad de narración.

29. **Campo `Contexto visual válido`**

    Indica si la secuencia tenia captions útiles después de limpieza. Si aparece como `no`, el comentario pudo generarse con un fallback temporal o por razon del evento.

30. **Campo `Contexto detectado`**

    Resume lo que el pipeline cree que ocurrio. Puede incluir fase de partida, progreso, senales visuales y captions útiles.

31. **Campo `Acciones visuales`**

    Muestra etiquetas de CLIP con scores. Ejemplo:

```text
duelo melee (0.31), puenteo sobre vacío (0.24)
```

    Estos scores son indicadores relativos. No deben interpretarse como verdad absoluta, sino como apoyo para el narrador.

32. **Comentario generado**

    Es la frase final producida por `NarratorModel`. Esta frase es la que se convierte a audio. Debe ser corta, clara y con energia de narrador deportivo.

33. **Reproductor de audio individual**

    Si `audio_path` existe, Streamlit muestra un reproductor. Desde ahi puedes escuchar cada comentario aislado. Los WAV se guardan en:

```text
outputs/audio_timeline/
```

34. **Configuracion recomendada para juego de acción rápida**

```text
Frames por segundo a analizar: 3 a 6
Cooldown entre eventos: 3 a 5
Umbral de similitud: 0.65 a 0.80
Frames de contexto antes: 1 a 2
Frames de contexto después: 3 a 5
Máximo de comentarios: 12 a 24
Mínimo de comentarios deseados: 8 a 14
Acciones visuales por secuencia: 3 a 4
Ventana temporal de acción: 2 a 3
Umbral de confianza de acciones: 0.18 a 0.25
Máximo de captions por secuencia: 5 a 7
Resolución para vision: original o 960x540 con GPU
Separacion mínima: 0.05 a 0.30 s
Velocidad de narración: 1.25 a 1.45
Palabras máximas: 12 a 18
Volumen del juego: 0.5 a 0.8
Volumen del narrador: 1.3 a 1.8
```

35. **Configuracion recomendada para exploracion lenta**

```text
Frames por segundo a analizar: 1 a 2
Cooldown entre eventos: 6 a 10
Umbral de similitud: 0.50 a 0.65
Frames de contexto antes: 2 a 4
Frames de contexto después: 4 a 8
Máximo de comentarios: 6 a 12
Mínimo de comentarios deseados: 3 a 6
Acciones visuales por secuencia: 2 a 3
Ventana temporal de acción: 3 a 4
Umbral de confianza de acciones: 0.20 a 0.30
Máximo de captions por secuencia: 4 a 6
Resolución para vision: 640x360 o 960x540
Separacion mínima: 0.75 a 1.50 s
Velocidad de narración: 0.95 a 1.15
Palabras máximas: 20 a 28
Volumen del juego: 0.8 a 1.1
Volumen del narrador: 1.0 a 1.3
```

36. **Si el análisis es demasiado lento**

    Prueba:

    - Bajar `Frames por segundo a analizar`.
    - Usar `640x360`.
    - Reducir `Máximo de comentarios en timeline`.
    - Reducir `Acciones visuales por secuencia`.
    - Usar clips más cortos.
    - Ejecutar con GPU CUDA.

37. **Si hay pocos comentarios**

    Prueba:

    - Subir `Mínimo de comentarios deseados`.
    - Subir `Frames por segundo a analizar`.
    - Bajar `Cooldown entre eventos`.
    - Subir un poco `Umbral de similitud para cambio de escena`.
    - Bajar `Umbral de confianza de acciones` si CLIP no detecta nada.

38. **Si hay demasiados comentarios**

    Prueba:

    - Bajar `Máximo de comentarios en timeline`.
    - Bajar `Mínimo de comentarios deseados`.
    - Subir `Cooldown entre eventos`.
    - Bajar `Umbral de similitud para cambio de escena`.
    - Subir `Separacion mínima entre comentarios`.

39. **Si la voz se escucha tapada o distorsionada**

    Para voz tapada:

    - Bajar `Volumen del juego`.
    - Subir `Volumen del narrador`.

    Para distorsion:

    - Bajar `Volumen del narrador`.
    - Evitar valores extremos si el audio original ya es fuerte.
    - Bajar también el volumen del juego si la mezcla completa satura.

40. **Si los comentarios no coinciden con la acción**

    Prueba:

    - Subir `Frames por segundo a analizar`.
    - Subir `Frames de contexto después del evento`.
    - Usar resolución `original` o `960x540`.
    - Reducir `Ventana temporal de acción` si CLIP toma acciones vecinas.
    - Subir `Umbral de confianza de acciones` para ser más conservador.

41. **Resultado final esperado**

    Una ejecución correcta produce:

    - Video final comentado en la interfaz.
    - Archivo `.mp4` en `outputs/final/`.
    - Audios `.wav` en `outputs/audio_timeline/`.
    - Timeline con frame clave, categoría, contexto, comentario y reproductor de audio.



