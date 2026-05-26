from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


class VisionModel:
    """Modelo de vision para describir frames con BLIP."""

    def __init__(self):
        """Carga procesador y modelo BLIP en GPU si existe; si no, usa CPU."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        model_name = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(model_name)

        if self.device.type == "cuda":
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            )
        else:
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()
        self.caption_prompt = (
            "minecraft skywars first person gameplay showing pvp action, "
            "items, movement and map context:"
        )

    def describe_frame(self, frame: np.ndarray) -> str:
        """Genera una descripcion textual de un frame RGB."""
        image = Image.fromarray(frame)
        inputs = self.processor(image, text=self.caption_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )

        with torch.inference_mode():
            with autocast_ctx:
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=34,
                    num_beams=4,
                    repetition_penalty=1.2,
                    length_penalty=0.95,
                    no_repeat_ngram_size=3,
                )
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption


def analyze_frames(frames: list, model: VisionModel) -> list[dict]:
    """Analiza frames y retorna captions por indice.

    Acepta:
    - list[np.ndarray]
    - list[dict] con campos {"sample_idx", "source_frame_idx", "time_s", "frame"}
    """
    results: list[dict[str, Any]] = []

    for i, item in enumerate(frames):
        if isinstance(item, dict):
            frame = item.get("frame")
            frame_idx = int(item.get("sample_idx", i))
            source_frame_idx = item.get("source_frame_idx")
            time_s = item.get("time_s")
        else:
            frame = item
            frame_idx = i
            source_frame_idx = None
            time_s = None

        base_result: dict[str, Any] = {"frame_idx": frame_idx}
        if source_frame_idx is not None:
            base_result["source_frame_idx"] = int(source_frame_idx)
        if time_s is not None:
            base_result["time_s"] = float(time_s)

        try:
            caption = model.describe_frame(frame)
            base_result["caption"] = caption
        except Exception:
            base_result["caption"] = "error al procesar frame"

        results.append(base_result)

    return results
