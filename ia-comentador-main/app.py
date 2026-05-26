"""Comentarista IA de Minecraft Skywars - aplicacion principal Streamlit."""

import io
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch

os.environ["TQDM_DISABLE"] = "1"

from modules.capture import extract_sampled_frames
from modules.commentary_pipeline import (
    build_commentary_sequences,
    enrich_sequences_with_action_tags,
    narrate_sequences,
)
from modules.heuristic import filter_frames_to_comment
from modules.narrator import CATEGORY_EMOJI, NarratorModel
from modules.tts import TTSModel, synthesize_sequence_comments
from modules.video_mixer import overlay_commentary_on_video
from modules.vision import VisionModel, analyze_frames
from modules.action_vision import ActionVisionModel


def _load_safe(factory):
    """Carga un modelo redirigiendo stdout/stderr para evitar ruido en consola."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        return factory()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


st.set_page_config(
    page_title="Comentarista IA de Minecraft Skywars",
    layout="wide",
    page_icon="🎮",
)

if "vision_model" not in st.session_state:
    with st.spinner("Cargando modelo de vision (BLIP)..."):
        st.session_state.vision_model = _load_safe(VisionModel)

if "narrator_model" not in st.session_state:
    st.session_state.narrator_model = NarratorModel()

if "tts_model" not in st.session_state:
    with st.spinner("Cargando modelo de voz (MMS-TTS espanol)..."):
        st.session_state.tts_model = _load_safe(TTSModel)

if "action_model" not in st.session_state:
    try:
        with st.spinner("Cargando detector visual de acciones (CLIP)..."):
            st.session_state.action_model = _load_safe(ActionVisionModel)
    except Exception:
        st.session_state.action_model = None

vision_model = st.session_state.vision_model
narrator_model = st.session_state.narrator_model
tts_model = st.session_state.tts_model
action_model = st.session_state.action_model

st.title("🎮 Comentarista IA de Minecraft Skywars")
st.caption(
    "Analiza secuencias de juego, genera narracion con contexto y entrega un video final "
    "con voz de caster sobrepuesta al audio original."
)

st.sidebar.header("Configuracion")
if torch.cuda.is_available():
    st.sidebar.success(f"GPU activa: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.warning("Sin GPU CUDA: ejecutando en CPU (mas lento).")
if action_model is None:
    st.sidebar.warning("Detector visual de acciones no disponible, usando solo captions.")

fps_sample = st.sidebar.slider("Frames por segundo a analizar", 1, 8, 3)
cooldown_frames = st.sidebar.slider("Cooldown entre eventos", 1, 12, 4)
similarity_threshold = st.sidebar.slider(
    "Umbral de similitud para cambio de escena",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05,
)

st.sidebar.divider()
st.sidebar.subheader("Narracion por Secuencia")
context_before = st.sidebar.slider("Frames de contexto antes del evento", 0, 5, 2)
context_after = st.sidebar.slider("Frames de contexto despues del evento", 1, 8, 4)
max_sequences = st.sidebar.slider("Maximo de comentarios en timeline", 6, 30, 18)
min_commentary_count = st.sidebar.slider("Minimo de comentarios deseados", 3, 20, 8)
top_action_tags = st.sidebar.slider("Acciones visuales por secuencia", 2, 6, 3)
action_window_radius = st.sidebar.slider("Ventana temporal de accion (+/- frames)", 1, 4, 2)
action_score_threshold = st.sidebar.slider(
    "Umbral de confianza de acciones",
    min_value=0.05,
    max_value=0.40,
    value=0.20,
    step=0.01,
)
max_captions_per_sequence = st.sidebar.slider(
    "Maximo de captions por secuencia",
    3,
    10,
    7,
)
resize_mode = st.sidebar.selectbox(
    "Resolucion para vision",
    options=["original", "960x540", "640x360"],
    index=0,
)
min_spacing_s = st.sidebar.slider(
    "Separacion minima entre comentarios (s)",
    min_value=0.0,
    max_value=2.0,
    value=0.10,
    step=0.05,
)
speaking_rate = st.sidebar.slider(
    "Velocidad de narracion (voz)",
    min_value=0.8,
    max_value=2.0,
    value=1.32,
    step=0.02,
)
max_comment_words = st.sidebar.slider(
    "Palabras maximas por comentario",
    min_value=10,
    max_value=32,
    value=18,
    step=1,
)

st.sidebar.divider()
st.sidebar.subheader("Mezcla de Audio")
game_audio_volume = st.sidebar.slider("Volumen del juego", 0.1, 1.5, 0.8, 0.05)
commentary_volume = st.sidebar.slider("Volumen del narrador", 0.5, 2.5, 1.35, 0.05)

st.sidebar.divider()
st.sidebar.markdown(
    "**Categorias detectadas:**\n"
    "- 🚀 Inicio\n"
    "- 📦 Loot\n"
    "- 🧱 Puenteo\n"
    "- 🎯 Centro\n"
    "- ⚔️ Combate\n"
    "- 💀 Eliminacion\n"
    "- 🌌 Peligro vacio\n"
    "- 🔥 Cierre\n"
    "- 🏆 Victoria\n"
    "- 🎮 General"
)

uploaded_file = st.file_uploader("Sube un video .mp4 de Skywars", type=["mp4"])
if uploaded_file is None:
    st.info("Sube tu video para generar la narracion automatica con overlay de audio.")
    st.stop()

temp_video_path = "temp_video.mp4"
with open(temp_video_path, "wb") as handle:
    handle.write(uploaded_file.read())

st.video(temp_video_path)

if st.button("Analizar y Renderizar Video Narrado", type="primary"):
    try:
        with st.spinner("Extrayendo frames con metadatos de tiempo..."):
            resize_to = None
            if resize_mode == "960x540":
                resize_to = (960, 540)
            elif resize_mode == "640x360":
                resize_to = (640, 360)
            sampled_frames = extract_sampled_frames(
                temp_video_path,
                fps_sample=fps_sample,
                resize_to=resize_to,
            )
            frames = [item["frame"] for item in sampled_frames]
        st.success(f"Frames muestreados: {len(sampled_frames)}")

        with st.spinner("Analizando frames con BLIP..."):
            analyzed = analyze_frames(sampled_frames, vision_model)
        st.success(f"Frames analizados: {len(analyzed)}")

        with st.spinner("Detectando eventos clave..."):
            filtered = filter_frames_to_comment(
                analyzed,
                cooldown_frames=cooldown_frames,
                similarity_threshold=similarity_threshold,
            )
        st.success(f"Eventos detectados: {len(filtered)}")

        with st.spinner("Construyendo secuencias con contexto..."):
            sequences = build_commentary_sequences(
                analyzed_frames=analyzed,
                selected_frames=filtered,
                context_before=context_before,
                context_after=context_after,
                max_captions_per_sequence=max_captions_per_sequence,
                max_sequences=max_sequences,
                target_commentary_count=min_commentary_count,
            )
        st.success(f"Secuencias narrables: {len(sequences)}")

        with st.spinner("Generando narracion estilo caster por secuencia..."):
            sequences = enrich_sequences_with_action_tags(
                sequences=sequences,
                frames=frames,
                action_model=action_model,
                top_k=top_action_tags,
                min_score=action_score_threshold,
                window_radius=action_window_radius,
            )
            narrator_model.max_comment_words = max_comment_words
            narrated_sequences = narrate_sequences(sequences, narrator_model)

        with st.spinner("Sintetizando voces y timeline de comentarios..."):
            timeline = synthesize_sequence_comments(
                narrated_sequences,
                tts_model,
                output_dir="outputs/audio_timeline",
                min_spacing_s=min_spacing_s,
                speaking_rate=speaking_rate,
            )
        valid_audio_count = sum(1 for item in timeline if item.get("audio_path"))
        if valid_audio_count == 0:
            st.error(
                "No se pudo sintetizar audio en ninguna secuencia. Revisa errores por secuencia "
                "abajo en el timeline."
            )

        output_name = f"commentated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_video_path = str(Path("outputs/final") / output_name)

        with st.spinner("Mezclando comentario sobre el video final..."):
            rendered_video = overlay_commentary_on_video(
                input_video_path=temp_video_path,
                commentary_timeline=timeline,
                output_video_path=output_video_path,
                game_audio_volume=game_audio_volume,
                commentary_volume=commentary_volume,
            )

        st.divider()
        st.subheader("Video Final Narrado")
        st.video(rendered_video)
        st.success(f"Render completado: {rendered_video}")

        st.divider()
        st.subheader("Timeline de Comentarios")

        if not timeline:
            st.warning("No se genero timeline de comentarios.")
        else:
            for item in timeline:
                category = item.get("category", "general")
                emoji = CATEGORY_EMOJI.get(category, "🎮")
                key_frame_idx = int(item.get("key_frame_idx", 0))
                preview_caption = str(item.get("context_summary_es", "")).strip()
                if not preview_caption:
                    preview_caption = " | ".join(item.get("captions", [])[:3]).strip()
                if not preview_caption:
                    preview_caption = item.get("context_caption", "sin contexto")

                with st.container(border=True):
                    col_img, col_text = st.columns([1, 2])
                    with col_img:
                        if 0 <= key_frame_idx < len(frames):
                            st.image(frames[key_frame_idx], caption=f"Frame {key_frame_idx}")
                    with col_text:
                        st.markdown(f"**{emoji} {category.upper()}** - _{item.get('reason', 'secuencia')}_")
                        st.markdown(
                            f"**Inicio en video:** {item.get('start_time_s', 0.0):.2f}s  \n"
                            f"**Narrado en:** {item.get('narration_time_s', 0.0):.2f}s  \n"
                            f"**Duracion voz:** {item.get('audio_duration_s', 0.0):.2f}s  \n"
                            f"**Contexto visual valido:** {'si' if item.get('has_visual_context') else 'no'}"
                        )
                        st.markdown(f"**Contexto detectado:** {preview_caption}")
                        if item.get("action_tags"):
                            tags_line = ", ".join(
                                f"{tag['alias']} ({tag['score']:.2f})"
                                for tag in item["action_tags"]
                            )
                            st.markdown(f"**Acciones visuales:** {tags_line}")
                        st.markdown(f"### 🎙️ {item.get('comment', '')}")
                        if item.get("audio_path"):
                            st.audio(item["audio_path"])
                        if item.get("audio_error"):
                            st.warning(f"Error de audio: {item['audio_error']}")

        st.balloons()

    except Exception as exc:
        st.error(f"Error durante el pipeline: {exc}")
