from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import torch
from scipy.io.wavfile import read, write
from transformers import VitsModel, VitsTokenizer


class TTSModel:
    """Modelo TTS para convertir texto en audio WAV."""

    def __init__(self, default_speaking_rate: float = 1.25):
        """Carga tokenizer y modelo TTS en GPU si existe; si no, usa CPU."""
        model_name = "facebook/mms-tts-spa"
        self.tokenizer = VitsTokenizer.from_pretrained(model_name)
        self.model = VitsModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_speaking_rate = max(0.8, min(2.0, float(default_speaking_rate)))
        self.model.to(self.device)
        self.model.eval()

    def _sanitize_text(self, text: str, max_words: int = 34) -> str:
        clean = re.sub(r"\s+", " ", (text or "").strip())
        clean = re.sub(r"\s+([,.;:!?])", r"\1", clean)
        words = clean.split()
        if len(words) <= max_words:
            return clean
        trimmed = " ".join(words[:max_words]).rstrip(",;:")
        return f"{trimmed}."

    def _dynamic_rate(self, text: str, speaking_rate: float | None = None) -> float:
        rate = self.default_speaking_rate if speaking_rate is None else float(speaking_rate)
        word_count = len((text or "").split())
        if word_count >= 30:
            rate += 0.08
        if word_count >= 42:
            rate += 0.08
        return max(0.8, min(2.0, rate))

    def synthesize(self, text: str, output_path: str, speaking_rate: float | None = None) -> str:
        """Sintetiza audio desde texto y lo guarda en un archivo WAV."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        clean_text = self._sanitize_text(text)
        effective_rate = self._dynamic_rate(clean_text, speaking_rate=speaking_rate)

        inputs = self.tokenizer(clean_text, return_tensors="pt")
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        with torch.inference_mode():
            try:
                waveform = self.model(**inputs, speaking_rate=effective_rate).waveform
            except TypeError:
                waveform = self.model(**inputs).waveform

        waveform_np = waveform.cpu().numpy().squeeze()
        waveform_np = np.asarray(waveform_np, dtype=np.float32)
        waveform_np = np.clip(waveform_np, -1.0, 1.0)
        # Guardar como PCM 16-bit para compatibilidad amplia.
        waveform_int16 = (waveform_np * 32767.0).astype(np.int16)

        write(
            output_path,
            rate=self.model.config.sampling_rate,
            data=waveform_int16,
        )
        return output_path


def get_wav_duration_s(audio_path: str) -> float:
    sample_rate, data = read(audio_path)
    if sample_rate <= 0:
        return 0.0
    frame_count = int(data.shape[0]) if hasattr(data, "shape") else len(data)
    return frame_count / float(sample_rate)


def synthesize_comments(
    narrated_frames: list[dict],
    model: TTSModel,
    output_dir: str = "outputs/audio",
) -> list[dict]:
    """Genera audios para comentarios narrados y agrega su ruta."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    synthesized_frames: list[dict] = []

    for frame in narrated_frames:
        updated = dict(frame)
        try:
            output_path = f"{output_dir}/frame_{frame['frame_idx']}.wav"
            updated["audio_path"] = model.synthesize(frame["comment"], output_path)
            updated["audio_duration_s"] = get_wav_duration_s(updated["audio_path"])
            updated["audio_error"] = None
        except Exception as exc:
            updated["audio_path"] = None
            updated["audio_duration_s"] = 0.0
            updated["audio_error"] = str(exc)
        synthesized_frames.append(updated)

    return synthesized_frames


def synthesize_sequence_comments(
    narrated_sequences: list[dict],
    model: TTSModel,
    output_dir: str = "outputs/audio_timeline",
    min_spacing_s: float = 0.1,
    speaking_rate: float = 1.25,
) -> list[dict]:
    """Sintetiza audios por secuencia y calcula su timeline real en el video."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    synthesized_sequences: list[dict] = []
    last_end_time = 0.0

    for sequence in sorted(narrated_sequences, key=lambda item: item.get("start_time_s", 0.0)):
        updated = dict(sequence)
        updated["audio_path"] = None
        updated["audio_duration_s"] = 0.0
        updated["audio_error"] = None
        try:
            sequence_id = int(sequence.get("sequence_id", len(synthesized_sequences)))
            output_path = f"{output_dir}/sequence_{sequence_id:03d}.wav"
            updated["audio_path"] = model.synthesize(
                sequence["comment"],
                output_path,
                speaking_rate=speaking_rate,
            )
            updated["audio_duration_s"] = get_wav_duration_s(updated["audio_path"])
        except Exception as exc:
            updated["audio_error"] = str(exc)

        desired_start = float(sequence.get("start_time_s", 0.0))
        actual_start = max(desired_start, last_end_time + min_spacing_s)
        updated["narration_time_s"] = actual_start
        updated["narration_end_s"] = actual_start + float(updated["audio_duration_s"])
        updated["narration_delay_s"] = max(0.0, actual_start - desired_start)

        last_end_time = updated["narration_end_s"]
        synthesized_sequences.append(updated)

    return synthesized_sequences
