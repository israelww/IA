from __future__ import annotations

from PIL import Image
from transformers import pipeline

ACTION_LABELS = [
    "minecraft player fighting another player with sword",
    "minecraft player close melee pvp combat",
    "minecraft player throwing snowballs at enemy",
    "minecraft player throwing eggs at enemy",
    "minecraft player eating golden apple",
    "minecraft player drinking potion to heal",
    "minecraft player opening chest and looting",
    "minecraft player placing blocks to bridge over void",
    "minecraft player near the void edge in danger",
    "minecraft player falling into the void",
    "minecraft player in inventory menu",
    "minecraft player running across islands",
    "minecraft player holding defensive position behind blocks",
    "minecraft skywars final duel 1v1",
    "minecraft skywars victory screen",
    "minecraft player opening inventory and swapping gear quickly",
    "minecraft player building upward tower for height advantage",
    "minecraft player chasing enemy on bridge",
    "minecraft player taking knockback near edge",
]

LABEL_ALIAS = {
    "minecraft player fighting another player with sword": "fighting with sword",
    "minecraft player close melee pvp combat": "close pvp combat",
    "minecraft player throwing snowballs at enemy": "throwing snowballs",
    "minecraft player throwing eggs at enemy": "throwing eggs",
    "minecraft player eating golden apple": "eating golden apple",
    "minecraft player drinking potion to heal": "drinking healing potion",
    "minecraft player opening chest and looting": "opening chest loot",
    "minecraft player placing blocks to bridge over void": "bridging over void",
    "minecraft player near the void edge in danger": "danger near void edge",
    "minecraft player falling into the void": "falling into void",
    "minecraft player in inventory menu": "inventory menu",
    "minecraft player running across islands": "rotating across islands",
    "minecraft player holding defensive position behind blocks": "defensive holding position",
    "minecraft skywars final duel 1v1": "final duel 1v1",
    "minecraft skywars victory screen": "victory screen",
    "minecraft player opening inventory and swapping gear quickly": "inventory swap",
    "minecraft player building upward tower for height advantage": "towering for height",
    "minecraft player chasing enemy on bridge": "bridge chase pressure",
    "minecraft player taking knockback near edge": "knockback near edge",
}


class ActionVisionModel:
    """Detector de acciones visuales con CLIP zero-shot."""

    def __init__(self):
        device = 0
        try:
            import torch

            if not torch.cuda.is_available():
                device = -1
        except Exception:
            device = -1

        self.classifier = pipeline(
            task="zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            device=device,
        )

    def classify_frame(
        self,
        frame,
        top_k: int = 4,
        min_score: float = 0.12,
    ) -> list[dict]:
        """Retorna top etiquetas de accion para un frame RGB."""
        image = Image.fromarray(frame)
        results = self.classifier(
            image,
            candidate_labels=ACTION_LABELS,
            hypothesis_template="This is a screenshot of {}.",
        )

        selected: list[dict] = []
        for item in results[: max(1, top_k)]:
            label = str(item.get("label", ""))
            score = float(item.get("score", 0.0))
            if score < min_score:
                continue
            selected.append(
                {
                    "label": label,
                    "alias": LABEL_ALIAS.get(label, label),
                    "score": score,
                }
            )
        return selected
