import cv2
import numpy as np


def extract_frames(video_path: str, fps_sample: int = 2) -> list:
    """Extrae frames RGB redimensionados de un video MP4."""
    sampled = extract_sampled_frames(video_path, fps_sample=fps_sample)
    return [entry["frame"] for entry in sampled]


def extract_sampled_frames(
    video_path: str,
    fps_sample: int = 2,
    resize_to: tuple[int, int] | None = None,
) -> list[dict]:
    """Extrae frames RGB junto con metadatos de tiempo para sincronizacion."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el archivo de video: {video_path}")

    sampled_frames: list[dict] = []
    frame_idx = 0
    sample_idx = 0

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            raise ValueError("No se pudo obtener un FPS valido del video.")
        if fps_sample <= 0:
            raise ValueError("fps_sample debe ser mayor que 0.")

        frame_interval = int(video_fps / fps_sample)
        if frame_interval <= 0:
            frame_interval = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = (
                    cv2.resize(frame_rgb, resize_to) if resize_to is not None else frame_rgb
                )
                sampled_frames.append(
                    {
                        "sample_idx": sample_idx,
                        "source_frame_idx": frame_idx,
                        "time_s": frame_idx / video_fps,
                        "frame": frame_resized,
                    }
                )
                sample_idx += 1

            frame_idx += 1
    finally:
        cap.release()

    return sampled_frames
