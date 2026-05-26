from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _probe_video_info(video_path: str) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _get_video_duration_s(video_path: str) -> float:
    info = _probe_video_info(video_path)
    duration_raw = info.get("format", {}).get("duration", "0")
    try:
        return max(0.0, float(duration_raw))
    except (TypeError, ValueError):
        return 0.0


def _has_audio_stream(video_path: str) -> bool:
    info = _probe_video_info(video_path)
    streams = info.get("streams", [])
    return any(stream.get("codec_type") == "audio" for stream in streams)


def overlay_commentary_on_video(
    input_video_path: str,
    commentary_timeline: list[dict],
    output_video_path: str = "outputs/final/commentated_video.mp4",
    game_audio_volume: float = 0.8,
    commentary_volume: float = 1.35,
) -> str:
    """Sobrepone audios de comentario con delay y mezcla sobre el video original."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("No se encontro ffmpeg en PATH.")

    valid_comments = [
        item
        for item in commentary_timeline
        if item.get("audio_path") and Path(item["audio_path"]).exists()
    ]

    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Si no hay comentarios sintetizados, devolver copia directa.
    if not valid_comments:
        cmd = [ffmpeg_path, "-y", "-i", input_video_path, "-c", "copy", str(output_path)]
        subprocess.run(cmd, check=True)
        return str(output_path)

    has_audio = _has_audio_stream(input_video_path)
    video_duration = _get_video_duration_s(input_video_path)

    cmd: list[str] = [ffmpeg_path, "-y", "-i", input_video_path]

    base_audio_input_idx = 0
    if not has_audio:
        # Crear pista silenciosa base si el video no trae audio.
        cmd.extend(
            [
                "-f",
                "lavfi",
                "-t",
                f"{video_duration:.3f}",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
            ]
        )
        base_audio_input_idx = 1

    comment_input_start = base_audio_input_idx + 1
    for comment in valid_comments:
        cmd.extend(["-i", str(comment["audio_path"])])

    filter_parts: list[str] = []
    filter_parts.append(
        f"[{base_audio_input_idx}:a]volume={game_audio_volume:.3f}[game]"
    )

    voice_labels: list[str] = []
    for local_idx, comment in enumerate(valid_comments, start=0):
        input_idx = comment_input_start + local_idx
        delay_ms = max(0, int(round(float(comment.get("narration_time_s", 0.0)) * 1000)))
        voice_label = f"v{local_idx + 1}"
        voice_labels.append(f"[{voice_label}]")
        filter_parts.append(
            f"[{input_idx}:a]adelay={delay_ms}:all=1,"
            f"volume={commentary_volume:.3f}[{voice_label}]"
        )

    if len(voice_labels) == 1:
        filter_parts.append(f"{voice_labels[0]}anull[voice]")
    else:
        joined = "".join(voice_labels)
        filter_parts.append(
            f"{joined}amix=inputs={len(voice_labels)}:duration=longest:normalize=0[voice]"
        )

    filter_parts.append("[game][voice]amix=inputs=2:duration=first:normalize=0[aout]")
    filter_complex = ";".join(filter_parts)

    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )

    subprocess.run(cmd, check=True)
    return str(output_path)
