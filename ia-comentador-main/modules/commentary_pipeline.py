from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher

_NOISE_PATTERNS = [
    "youtube",
    "minecraft mod",
    "a black and white photo",
    "a photo of a man in a suit",
    "man in a suit",
    "screenshot of a person walking",
    "how to build",
    "a screenshot of a building in minecraft",
    "a screenshot of a castle in minecraft",
    "a screenshot of a minecraft video game",
]

_MIN_WORD_LEN = 2

_VISUAL_SIGNALS: dict[str, list[str]] = {
    "combate": [
        "fighting",
        "pvp",
        "duel",
        "sword",
        "melee",
        "hits",
        "critical",
        "enemy",
        "rival",
    ],
    "proyectiles": ["snowball", "snowballs", "egg", "eggs"],
    "loot": ["chest", "inventory", "looting", "armor", "helmet", "sword", "potion"],
    "movilidad": ["bridge", "bridging", "running", "rotating", "islands", "jumping"],
    "riesgo_vacio": ["void", "falling", "edge", "knockback near edge", "fell"],
    "curacion": ["golden apple", "gapple", "drinking potion", "heal", "healing"],
    "cierre": ["top 3", "top 2", "1v1", "final duel", "endgame"],
    "victoria": ["victory", "winner", "you win", "win screen"],
}

_ACTION_ALIAS_ES: dict[str, str] = {
    "fighting with sword": "duelo melee",
    "close pvp combat": "intercambio corto",
    "throwing snowballs": "presion con snowballs",
    "throwing eggs": "presion con huevos",
    "eating golden apple": "curacion con manzana dorada",
    "drinking healing potion": "curacion con pocion",
    "opening chest loot": "loot de cofre",
    "bridging over void": "puenteo sobre vacio",
    "danger near void edge": "riesgo al borde del vacio",
    "falling into void": "caida al vacio",
    "inventory menu": "gestion de inventario",
    "rotating across islands": "rotacion entre islas",
    "defensive holding position": "posicion defensiva",
    "final duel 1v1": "duelo final 1v1",
    "victory screen": "pantalla de victoria",
    "inventory swap": "cambio rapido de equipo",
    "towering for height": "busca altura con torre",
    "bridge chase pressure": "persecucion en puente",
    "knockback near edge": "knockback cerca del borde",
}

_BLOCKED_ACTION_ALIASES = {"shooting bow"}
_SINGLE_FRAME_ACTIONS = {
    "victory screen",
    "falling into void",
    "danger near void edge",
    "knockback near edge",
}
_MIN_STABLE_ACTION_SCORE = 0.18
_LATE_GAME_ACTION_ALIASES = {"final duel 1v1", "victory screen"}
_MIN_LATE_GAME_ACTION_SCORE = 0.30

_SIGNAL_ES: dict[str, str] = {
    "combate": "combate visible",
    "proyectiles": "uso de proyectiles",
    "loot": "loot o inventario",
    "movilidad": "rotacion o cruce",
    "riesgo_vacio": "riesgo de vacio",
    "curacion": "curacion",
    "cierre": "posible tramo final",
    "victoria": "victoria",
}


def _clean_caption(caption: str) -> str:
    text = (caption or "").strip()
    if not text:
        return ""
    lower = text.lower()
    if lower in {"error al procesar frame", "none", "null", "n/a"}:
        return ""
    return text


def _dedupe_captions(captions: list[str], similarity_threshold: float = 0.9) -> list[str]:
    unique: list[str] = []
    for caption in captions:
        if not caption:
            continue
        if not unique:
            unique.append(caption)
            continue
        if all(
            SequenceMatcher(None, caption.lower(), prev.lower()).ratio() < similarity_threshold
            for prev in unique
        ):
            unique.append(caption)
    return unique


def _is_noise_caption(caption: str) -> bool:
    lower = caption.lower().strip()
    if not lower:
        return True

    if any(pattern in lower for pattern in _NOISE_PATTERNS):
        return True

    if lower.startswith("a screenshot of") and not any(
        token in lower
        for token in [
            "fighting",
            "throwing",
            "snowball",
            "egg",
            "sword",
            "void",
            "bridge",
            "chest",
            "inventory",
            "eating",
            "potion",
            "golden apple",
        ]
    ):
        return True

    words = [word for word in lower.replace("-", " ").split() if len(word) >= _MIN_WORD_LEN]
    if len(words) < 3:
        return True

    unique_ratio = len(set(words)) / max(1, len(words))
    if unique_ratio < 0.35:
        return True

    minecraft_count = sum(1 for word in words if word == "minecraft")
    if minecraft_count >= max(4, len(words) // 2):
        return True

    return False


def _reason_fallback_context(reason: str) -> str:
    lower = (reason or "").lower()
    if "anclaje temporal" in lower:
        return (
            "Secuencia de seguimiento continuo en skywars: lectura de posicionamiento, "
            "rotaciones y posibles ventanas de combate."
        )
    if "evento clave" in lower:
        return (
            "Secuencia de skywars con evento clave: posible eliminacion, riesgo de vacio "
            "o jugada de cierre."
        )
    if "cambio de accion" in lower:
        return (
            "Secuencia de skywars con cambio de accion: rotacion, puenteo y entrada a combate."
        )
    if "inicio" in lower:
        return "Secuencia de apertura de skywars: loot inicial y preparacion de ruta."
    return "Secuencia de skywars en desarrollo con cambios tacticos."


def _extract_visual_signals(captions: list[str]) -> list[str]:
    if not captions:
        return []
    lower_blob = " ".join(captions).lower()
    signals: list[str] = []
    for signal_name, keywords in _VISUAL_SIGNALS.items():
        if any(keyword in lower_blob for keyword in keywords):
                signals.append(signal_name)
    return signals


def _progress_ratio(start_time_s: float, video_duration_s: float) -> float:
    if video_duration_s <= 0:
        return 0.0
    return max(0.0, min(1.0, start_time_s / video_duration_s))


def _detect_phase_hint(
    start_time_s: float,
    video_duration_s: float,
    reason: str,
    captions: list[str],
) -> str:
    lower_reason = (reason or "").lower()
    caption_blob = " ".join(captions).lower()
    progress = _progress_ratio(start_time_s, video_duration_s)

    if "victory" in caption_blob or "you win" in caption_blob or "win screen" in caption_blob:
        return "cierre"

    is_near_opening = start_time_s <= 8.0 or progress <= 0.12
    if ("inicio" in lower_reason and (start_time_s <= 12.0 or progress <= 0.15)) or is_near_opening:
        return "inicio"

    late_text_signals = ["endgame", "deathmatch", "top 2", "top 3", "last players"]
    if progress >= 0.55 and any(signal in caption_blob for signal in late_text_signals):
        return "cierre"

    if progress >= 0.82 and video_duration_s >= 20.0:
        return "cierre"

    return "medio"


def _phase_label_es(phase_hint: str) -> str:
    if phase_hint == "inicio":
        return "inicio de partida"
    if phase_hint == "cierre":
        return "tramo final"
    return "juego medio"


def _build_context_summary_es(
    phase_hint: str,
    progress_pct: int,
    visual_signals: list[str],
    trigger_reasons: list[str],
) -> str:
    phase_text = f"Fase: {_phase_label_es(phase_hint)} ({progress_pct}% del video)"
    signal_text = ""
    if visual_signals:
        translated = [_SIGNAL_ES.get(signal, signal) for signal in visual_signals]
        signal_text = f"Senales: {', '.join(translated)}"

    reason_text = ""
    if trigger_reasons:
        reason_text = f"Evento: {', '.join(sorted(set(trigger_reasons)))}"

    parts = [part for part in [phase_text, signal_text, reason_text] if part]
    return ". ".join(parts) + "."


def _remove_unreliable_item_claims(text: str) -> str:
    lower = text.lower()
    unreliable_terms = ["bow", "arrows", "shooting"]
    if not any(term in lower for term in unreliable_terms):
        return text

    cleaned = text
    for phrase in [
        "shooting bow and arrows",
        "shooting a bow and arrow",
        "shooting a bow",
        "bow and arrows",
        "with a bow",
        "holding a bow",
        "a bow",
        "bow",
        "arrows",
    ]:
        cleaned = cleaned.replace(phrase, "")
        cleaned = cleaned.replace(phrase.title(), "")
    return " ".join(cleaned.split()).strip(" ,;:.")


def _build_sequence_context(
    captions: list[str],
    trigger_reasons: list[str],
    fallback_reason: str,
    phase_hint: str,
    progress_pct: int,
    context_summary_es: str,
) -> str:
    phase_text = f"Fase temporal: {phase_hint}. Progreso video: {progress_pct}%."

    if not captions:
        fallback = _reason_fallback_context(fallback_reason)
        if trigger_reasons:
            return (
                f"{phase_text} Resumen visual: {context_summary_es} {fallback} "
                f"Razones detectadas: {', '.join(sorted(set(trigger_reasons)))}."
            )
        return f"{phase_text} Resumen visual: {context_summary_es} {fallback}"

    signals = _extract_visual_signals(captions)
    if phase_hint != "cierre":
        signals = [signal for signal in signals if signal not in {"cierre"}]
    compact_captions = " ; ".join(captions[:4])
    signal_text = f"Senales tacticas: {', '.join(signals)}." if signals else ""
    reason_text = (
        f"Razones detectadas: {', '.join(sorted(set(trigger_reasons)))}."
        if trigger_reasons
        else ""
    )
    return (
        f"Secuencia de skywars. {phase_text} Resumen visual: {context_summary_es} "
        f"Captions utiles: {compact_captions}. {signal_text} {reason_text}"
    ).strip()


def build_commentary_sequences(
    analyzed_frames: list[dict],
    selected_frames: list[dict],
    context_before: int = 2,
    context_after: int = 3,
    max_captions_per_sequence: int = 6,
    max_sequences: int = 24,
    target_commentary_count: int | None = None,
) -> list[dict]:
    """Agrupa frames seleccionados en secuencias para narrar con mayor contexto."""
    if not analyzed_frames:
        return []

    analyzed_by_idx = {int(frame["frame_idx"]): frame for frame in analyzed_frames}
    analyzed_indices = sorted(analyzed_by_idx.keys())
    min_idx = analyzed_indices[0]
    max_idx = analyzed_indices[-1]

    selected_idx_with_reason: list[tuple[int, str]] = []
    for frame in selected_frames:
        if "frame_idx" not in frame:
            continue
        selected_idx_with_reason.append((int(frame["frame_idx"]), str(frame.get("reason", ""))))

    if not selected_idx_with_reason:
        selected_idx_with_reason = [(min_idx, "inicio de la partida")]

    selected_idx_with_reason.sort(key=lambda item: item[0])

    # Fallback temporal: si hay pocos eventos detectados, insertar anclajes repartidos
    # para asegurar narracion continua en videos largos aunque vision falle.
    duration_s = float(analyzed_by_idx[max_idx].get("time_s", 0.0)) - float(
        analyzed_by_idx[min_idx].get("time_s", 0.0)
    )
    video_duration_s = max(
        duration_s,
        float(analyzed_by_idx[max_idx].get("time_s", 0.0)),
        0.0,
    )
    auto_target = max(3, min(max_sequences, int(duration_s // 12) + 1))
    desired_count = (
        max(1, min(max_sequences, target_commentary_count))
        if target_commentary_count is not None
        else auto_target
    )

    if len(selected_idx_with_reason) < desired_count:
        existing_indices = [idx for idx, _ in selected_idx_with_reason]
        span = max(1, max_idx - min_idx)
        for k in range(desired_count):
            ratio = k / max(1, desired_count - 1)
            anchor_idx = min_idx + int(round(span * ratio))
            too_close = any(abs(anchor_idx - existing) <= (context_before + context_after + 1) for existing in existing_indices)
            if too_close:
                continue
            selected_idx_with_reason.append((anchor_idx, "anclaje temporal"))
            existing_indices.append(anchor_idx)

    selected_idx_with_reason.sort(key=lambda item: item[0])

    raw_spans: list[dict] = []
    for idx, reason in selected_idx_with_reason:
        start = max(min_idx, idx - context_before)
        end = min(max_idx, idx + context_after)
        raw_spans.append({"start": start, "end": end, "triggers": [(idx, reason)]})

    merged_spans: list[dict] = []
    for span in raw_spans:
        if not merged_spans:
            merged_spans.append(span)
            continue
        last = merged_spans[-1]
        if span["start"] <= last["end"] + 1:
            last["end"] = max(last["end"], span["end"])
            last["triggers"].extend(span["triggers"])
        else:
            merged_spans.append(span)

    sequences: list[dict] = []
    for seq_id, span in enumerate(merged_spans[:max_sequences]):
        entries: list[dict] = []
        for idx in range(span["start"], span["end"] + 1):
            entry = analyzed_by_idx.get(idx)
            if entry is not None:
                entries.append(entry)

        if not entries:
            continue

        captions = [
            _remove_unreliable_item_claims(_clean_caption(str(entry.get("caption", ""))))
            for entry in entries
        ]
        captions = [caption for caption in captions if caption]
        captions = [caption for caption in captions if not _is_noise_caption(caption)]
        captions = _dedupe_captions(captions)
        if max_captions_per_sequence > 0:
            captions = captions[:max_captions_per_sequence]

        trigger_indices = [trigger_idx for trigger_idx, _ in span["triggers"]]
        trigger_reasons = [reason for _, reason in span["triggers"] if reason]
        reason_counter = Counter(trigger_reasons)
        dominant_reason = reason_counter.most_common(1)[0][0] if reason_counter else "momento detectado"
        key_frame_idx = trigger_indices[len(trigger_indices) // 2]

        start_time_s = float(entries[0].get("time_s", 0.0))
        end_time_s = float(entries[-1].get("time_s", start_time_s))
        if end_time_s < start_time_s:
            end_time_s = start_time_s
        key_entry = analyzed_by_idx.get(key_frame_idx, entries[0])
        key_time_s = float(key_entry.get("time_s", start_time_s))

        visual_signals = _extract_visual_signals(captions)
        phase_hint = _detect_phase_hint(
            start_time_s=key_time_s,
            video_duration_s=video_duration_s,
            reason=dominant_reason,
            captions=captions,
        )
        if phase_hint != "cierre":
            visual_signals = [signal for signal in visual_signals if signal != "cierre"]
        progress_pct = int(round(_progress_ratio(key_time_s, video_duration_s) * 100))
        context_summary_es = _build_context_summary_es(
            phase_hint=phase_hint,
            progress_pct=progress_pct,
            visual_signals=visual_signals,
            trigger_reasons=trigger_reasons,
        )
        context_caption = _build_sequence_context(
            captions=captions,
            trigger_reasons=trigger_reasons,
            fallback_reason=dominant_reason,
            phase_hint=phase_hint,
            progress_pct=progress_pct,
            context_summary_es=context_summary_es,
        )

        sequences.append(
            {
                "sequence_id": seq_id,
                "start_frame_idx": int(entries[0]["frame_idx"]),
                "end_frame_idx": int(entries[-1]["frame_idx"]),
                "key_frame_idx": int(key_frame_idx),
                "key_time_s": key_time_s,
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "captions": captions,
                "context_caption": context_caption,
                "reason": dominant_reason,
                "trigger_count": len(trigger_indices),
                "has_visual_context": bool(captions),
                "visual_signals": visual_signals,
                "phase_hint": phase_hint,
                "progress_pct": progress_pct,
                "context_summary_es": context_summary_es,
            }
        )

    return sequences


def narrate_sequences(sequences: list[dict], model) -> list[dict]:
    """Genera comentarios por secuencia (no por frame individual)."""
    narrated: list[dict] = []
    for sequence in sequences:
        updated = dict(sequence)
        try:
            comment, category = model.generate_comment(sequence["context_caption"])
            updated["comment"] = comment
            updated["category"] = category
        except Exception:
            updated["comment"] = "Momento importante de Skywars en desarrollo."
            updated["category"] = "general"
        narrated.append(updated)
    return narrated


def enrich_sequences_with_action_tags(
    sequences: list[dict],
    frames: list,
    action_model,
    top_k: int = 4,
    min_score: float = 0.12,
    window_radius: int = 2,
) -> list[dict]:
    """Inyecta acciones visuales detectadas por frame clave en cada secuencia."""
    if action_model is None:
        return [dict(sequence, action_tags=[], has_action_context=False) for sequence in sequences]

    enriched: list[dict] = []
    for sequence in sequences:
        updated = dict(sequence)
        key_idx = int(sequence.get("key_frame_idx", 0))
        phase_hint = str(sequence.get("phase_hint", "medio"))
        action_tags: list[dict] = []
        try:
            frame_candidates = []
            start_idx = max(0, key_idx - window_radius)
            end_idx = min(len(frames) - 1, key_idx + window_radius)
            for idx in range(start_idx, end_idx + 1):
                distance = abs(idx - key_idx)
                temporal_weight = 1.0 / (1.0 + distance)
                frame_candidates.append((frames[idx], temporal_weight))

            weighted_score_by_alias: dict[str, float] = {}
            weight_total_by_alias: dict[str, float] = {}
            peak_score_by_alias: dict[str, float] = {}
            count_by_alias: dict[str, int] = {}
            label_by_alias: dict[str, str] = {}
            for frame, temporal_weight in frame_candidates:
                per_frame = action_model.classify_frame(
                    frame,
                    top_k=max(3, top_k),
                    min_score=min_score,
                )
                for tag in per_frame:
                    alias = str(tag.get("alias", ""))
                    score = float(tag.get("score", 0.0))
                    if not alias:
                        continue
                    if alias in _BLOCKED_ACTION_ALIASES:
                        continue
                    weighted_score_by_alias[alias] = (
                        weighted_score_by_alias.get(alias, 0.0) + (score * temporal_weight)
                    )
                    weight_total_by_alias[alias] = (
                        weight_total_by_alias.get(alias, 0.0) + temporal_weight
                    )
                    peak_score_by_alias[alias] = max(peak_score_by_alias.get(alias, 0.0), score)
                    count_by_alias[alias] = count_by_alias.get(alias, 0) + 1
                    label_by_alias[alias] = str(tag.get("label", alias))

            scored_aliases: list[tuple[str, float]] = []
            for alias, weighted_sum in weighted_score_by_alias.items():
                avg_score = weighted_sum / max(1e-6, weight_total_by_alias.get(alias, 0.0))
                boosted_score = 0.65 * avg_score + 0.35 * peak_score_by_alias.get(alias, 0.0)
                if boosted_score < _MIN_STABLE_ACTION_SCORE:
                    continue
                if alias in _LATE_GAME_ACTION_ALIASES:
                    if phase_hint != "cierre" or boosted_score < _MIN_LATE_GAME_ACTION_SCORE:
                        continue
                if count_by_alias.get(alias, 0) < 2 and alias not in _SINGLE_FRAME_ACTIONS:
                    continue
                scored_aliases.append((alias, boosted_score))

            ranked = sorted(scored_aliases, key=lambda item: item[1], reverse=True)
            for alias, score in ranked[:top_k]:
                action_tags.append(
                    {
                        "label": label_by_alias.get(alias, alias),
                        "alias": alias,
                        "score": score,
                    }
                )
        except Exception as exc:
            updated["action_error"] = str(exc)

        updated["action_tags"] = action_tags
        updated["has_action_context"] = bool(action_tags)

        if action_tags:
            action_aliases = [tag["alias"] for tag in action_tags]
            action_text = ", ".join(action_aliases)
            action_text_es = ", ".join(_ACTION_ALIAS_ES.get(alias, alias) for alias in action_aliases)
            updated["context_caption"] = (
                f"{updated.get('context_caption', '').strip()} "
                f"Acciones visuales probables: {action_text}. "
                f"Lectura en espanol: {action_text_es}."
            ).strip()
            updated["action_summary_es"] = action_text_es
        enriched.append(updated)
    return enriched
