"""Narrador inteligente para Minecraft Skywars con reglas robustas."""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuracion semantica para detectar escenas y contexto
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "inicio": [
        "spawn",
        "starting island",
        "island spawn",
        "inicio",
        "apertura",
        "fase temporal: inicio",
        "inicio de partida",
        "primeros segundos",
        "loot inicial",
        "preparacion de ruta",
        "jaula",
        "cage",
        "countdown",
        "start",
        "begin",
        "pre game",
        "lobby",
        "skywars begins",
    ],
    "loot": [
        "chest",
        "cofre",
        "loot",
        "inventory",
        "inventario",
        "item",
        "armor",
        "armadura",
        "helmet",
        "casco",
        "chestplate",
        "leggings",
        "boots",
        "sword",
        "espada",
        "pickaxe",
        "potion",
        "flechas",
        "enchanted",
        "diamond",
        "iron",
        "gold",
    ],
    "puenteo": [
        "bridge",
        "bridging",
        "speed bridge",
        "ninja bridge",
        "placing blocks",
        "building over void",
        "crossing to island",
        "bloques",
        "puente",
        "island to island",
    ],
    "centro": [
        "mid",
        "center",
        "centro",
        "middle",
        "middle island",
        "control mid",
        "rotating to mid",
    ],
    "combate": [
        "fight",
        "fighting",
        "combat",
        "duel",
        "pvp",
        "attacking",
        "hits",
        "trading hits",
        "enemy",
        "rival",
        "crosshair",
        "aim",
        "critical hit",
        "knockback",
        "rod",
        "sword",
        "melee",
        "projectile",
        "snowball",
        "snowballs",
        "egg",
        "eggs",
        "throwing",
        "duelo melee",
        "intercambio corto",
        "presion con snowballs",
        "presion con huevos",
    ],
    "eliminacion": [
        "killed",
        "kill",
        "eliminated",
        "slain",
        "dead",
        "death",
        "final kill",
        "void kill",
        "wins the fight",
    ],
    "peligro_vacio": [
        "void",
        "falling",
        "fell",
        "fall off",
        "edge",
        "borde",
        "knocked",
        "clutch",
        "pearled",
        "mlg water",
        "riesgo al borde del vacio",
        "knockback cerca del borde",
        "caida al vacio",
    ],
    "cierre": [
        "endgame",
        "late game",
        "fase temporal: cierre",
        "tramo final",
        "deathmatch",
        "final phase",
        "border closing",
        "last players",
        "top 4",
        "top 3",
        "top 2",
        "1v1",
        "duelo final 1v1",
    ],
    "victoria": [
        "victory",
        "winner",
        "won",
        "you win",
        "win screen",
        "game over winner",
        "champion",
    ],
}

_CATEGORY_WEIGHTS: dict[str, int] = {
    "inicio": 2,
    "loot": 2,
    "puenteo": 3,
    "centro": 2,
    "combate": 3,
    "eliminacion": 5,
    "peligro_vacio": 4,
    "cierre": 3,
    "victoria": 6,
}

_INVALID_CAPTION_SIGNALS = {
    "",
    "none",
    "null",
    "n/a",
    "error al procesar frame",
    "error processing frame",
    "unable to generate caption",
}

_ITEM_KEYWORDS: dict[str, str] = {
    "diamond sword": "espada de diamante",
    "iron sword": "espada de hierro",
    "stone sword": "espada de piedra",
    "gold sword": "espada de oro",
    "netherite sword": "espada de netherite",
    "sword": "espada",
    "rod": "cana",
    "axe": "hacha",
    "pickaxe": "pico",
    "ender pearl": "ender pearl",
    "pearl": "ender pearl",
    "snowball": "snowballs",
    "egg": "huevos",
    "golden apple": "manzana dorada",
    "gapple": "manzana dorada",
    "apple": "manzana",
    "potion": "pocion",
    "lava bucket": "cubo de lava",
    "water bucket": "cubo de agua",
    "tnt": "tnt",
    "diamond armor": "armadura de diamante",
    "iron armor": "armadura de hierro",
    "chain armor": "armadura de cota",
    "armor": "armadura",
}

_PROJECTILE_ITEMS = {"snowballs", "huevos"}

_CONTEXT_KEYWORDS: dict[str, list[str]] = {
    "mid": ["mid", "center", "centro", "middle island", "middle"],
    "bridge": ["bridge", "bridging", "placing blocks", "speed bridge", "ninja bridge", "puente"],
    "void": ["void", "vacio", "falling", "fell", "edge", "borde", "filo", "knocked", "clutch", "mlg water"],
    "height": ["high ground", "tower", "top", "elevated", "above", "altura", "torre"],
    "pressure": ["rush", "pushing", "chasing", "sprinting", "aggressive", "presion", "persecucion"],
    "defense": ["hiding", "cover", "wall", "holding angle", "camping", "sneak", "defensiva"],
    "low_health": ["low health", "one heart", "half heart", "critical health", "hearts low"],
}

_PLAYERS_HINTS: dict[str, list[str]] = {
    "top2": ["top 2", "two players left", "2 players left", "1v1", "last two"],
    "top3": ["top 3", "three players left", "3 players left", "last three"],
    "top4": ["top 4", "four players left", "4 players left", "last four"],
}

_PLAYERS_HINT_TEXT: dict[str, str] = {
    "top2": "1v1: cada trade puede cerrar la partida.",
    "top3": "Top 3, rotar mal cuesta partida.",
    "top4": "Top 4, recursos y posicion pesan mas.",
}

_TEMPLATES: dict[str, list[str]] = {
    "inicio": [
        "Sale de spawn y define la ruta inicial.",
        "Primeros segundos: loot rapido o rush directo.",
        "Arranque limpio; necesita velocidad antes del cruce.",
        "Inicio de ronda, el timing ya empieza a contar.",
    ],
    "loot": [
        "Abre cofre y acelera la salida.",
        "Ordena recursos para pelear sin perder tempo.",
        "Buen loot; ahora importa salir antes del rival.",
        "Se equipa con {item} y prepara presion.",
        "Inventario rapido para entrar mejor al duelo.",
        "Consigue recursos y busca la siguiente rotacion.",
    ],
    "puenteo": [
        "Cruce expuesto; un golpe puede mandarlo al vacio.",
        "Puenteo con presion, cada bloque cuenta.",
        "Se abre camino entre islas buscando iniciativa.",
        "Rotacion arriesgada sobre vacio.",
    ],
    "centro": [
        "Gana espacio en mid y controla rutas.",
        "Pisa centro para castigar rotaciones tardias.",
        "Mid le da recursos y angulos de presion.",
        "Buena rotacion al centro; llega con iniciativa.",
    ],
    "combate": [
        "Duelo abierto; el primer combo pesa mucho.",
        "Trade cerrado, el knockback decide.",
        "Entra al intercambio buscando ventaja.",
        "Pelea directa; posicion antes que ego.",
        "Presiona con {item}; busca cortar el avance.",
        "Choque rapido, no puede regalar rango.",
    ],
    "eliminacion": [
        "Baja confirmada; se abre el mapa.",
        "Convierte la presion en eliminacion.",
        "El rival cae y cambia el cierre.",
        "Kill importante; gana espacio inmediato.",
    ],
    "peligro_vacio": [
        "Al borde del vacio; un hit decide.",
        "Zona peligrosa, no puede regalar knockback.",
        "Se juega la vida en el borde.",
        "Momento delicado sobre vacio.",
    ],
    "cierre": [
        "Tramo final; cada rotacion vale oro.",
        "Endgame tenso, posicion antes que ego.",
        "Final cerrado; paciencia y altura deciden.",
        "Quedan pocos, no puede regalar espacio.",
    ],
    "victoria": [
        "Victoria cerrada con buena lectura.",
        "Se lleva el Skywars con autoridad.",
        "Win confirmado; ejecucion limpia.",
        "Cierra la partida sin dar opciones.",
    ],
    "general": [
        "Ronda abierta; toca leer la siguiente rotacion.",
        "Momento de pausa, pero el mapa sigue vivo.",
        "Busca informacion antes de comprometerse.",
        "Se mantiene activo, esperando ventana clara.",
    ],
    "fallback_error": [
        "Escena confusa, pero la partida sigue intensa y abierta.",
        "No se ve claro el detalle, mantenemos foco en el ritmo del juego.",
        "Frame ambiguo, seguimos leyendo la siguiente accion clave.",
    ],
}

_ACTION_TEMPLATES: list[tuple[str, list[str]]] = [
    (
        "pantalla de victoria",
        [
            "Pantalla de victoria; la ronda queda cerrada.",
            "Win en pantalla, cierre limpio.",
        ],
    ),
    (
        "caida al vacio",
        [
            "Caida al vacio; castigo total al posicionamiento.",
            "Se va al vacio y la pelea cambia al instante.",
        ],
    ),
    (
        "knockback cerca del borde",
        [
            "Knockback cerca del borde; un hit mas decide.",
            "Lo empuja al filo y fuerza panico defensivo.",
        ],
    ),
    (
        "riesgo al borde del vacio",
        [
            "Esta al borde; no puede aceptar otro trade.",
            "Zona de vacio, cualquier golpe lo sentencia.",
        ],
    ),
    (
        "puenteo sobre vacio",
        [
            "Puenteo sobre vacio; cruce muy castigable.",
            "Cruza expuesto, necesita terminar rapido.",
        ],
    ),
    (
        "persecucion en puente",
        [
            "Persecucion en puente; el knockback manda.",
            "Lo corre en linea recta, buscando el golpe final.",
        ],
    ),
    (
        "busca altura con torre",
        [
            "Sube por altura y obliga al rival a mirar arriba.",
            "Toma verticalidad para pelear con ventaja.",
        ],
    ),
    (
        "duelo final 1v1",
        [
            "1v1 final; aqui no hay margen.",
            "Duelo final, cada hit vale partida.",
        ],
    ),
    (
        "curacion con manzana dorada",
        [
            "Se cura antes de reentrar al trade.",
            "Manzana dorada para comprar segundos clave.",
        ],
    ),
    (
        "curacion con pocion",
        [
            "Pocion usada para reiniciar la pelea.",
            "Recupera vida y busca volver con tempo.",
        ],
    ),
    (
        "presion con huevos",
        [
            "Huevos para cortar avance y buscar knockback.",
            "Tira huevos para romper el timing rival.",
        ],
    ),
    (
        "presion con snowballs",
        [
            "Snowballs para frenar entrada y abrir combo.",
            "Presiona con snowballs, buscando descolocar.",
        ],
    ),
    (
        "loot de cofre",
        [
            "Cofre rapido; quiere salir antes del rush.",
            "Loot directo, sin regalar segundos.",
        ],
    ),
    (
        "gestion de inventario",
        [
            "Ajusta inventario para entrar mas limpio.",
            "Ordena la hotbar antes del siguiente choque.",
        ],
    ),
    (
        "cambio rapido de equipo",
        [
            "Cambia equipo rapido y prepara reentrada.",
            "Swap veloz; busca llegar mejor al duelo.",
        ],
    ),
    (
        "rotacion entre islas",
        [
            "Rota entre islas buscando mejor angulo.",
            "Cambia de isla para no quedar encerrado.",
        ],
    ),
    (
        "posicion defensiva",
        [
            "Se cubre y fuerza al rival a iniciar.",
            "Defiende posicion, esperando error rival.",
        ],
    ),
    (
        "duelo melee",
        [
            "Melee directo; el rango decide el trade.",
            "Entra al cuerpo a cuerpo buscando combo.",
        ],
    ),
    (
        "intercambio corto",
        [
            "Trade corto; cada golpe mueve la ventaja.",
            "Intercambio rapido, no hay espacio gratis.",
        ],
    ),
]

CATEGORY_EMOJI: dict[str, str] = {
    "inicio": "🚀",
    "loot": "📦",
    "puenteo": "🧱",
    "centro": "🎯",
    "combate": "⚔️",
    "eliminacion": "💀",
    "peligro_vacio": "🌌",
    "cierre": "🔥",
    "victoria": "🏆",
    "general": "🎮",
}

_PRIORITY_CATEGORIES = ["victoria", "eliminacion", "peligro_vacio"]
_MAX_COMMENT_WORDS = 22


@dataclass
class SceneContext:
    item: str = "su equipo"
    action_tags: tuple[str, ...] = ()
    visual_signals: tuple[str, ...] = ()
    phase_hint: str = "medio"
    progress_pct: int = 0
    is_ranged: bool = False
    in_mid: bool = False
    bridging: bool = False
    void_risk: bool = False
    has_height: bool = False
    is_pressuring: bool = False
    is_defensive: bool = False
    low_health: bool = False
    players_hint: str = ""
    is_caption_invalid: bool = False


class NarratorModel:
    """Narrador de Skywars basado en plantillas con heuristicas robustas."""

    def __init__(self):
        self._used_recently: list[str] = []
        self._max_memory = 10
        self._last_category = "general"
        self._last_comment = ""
        self._category_streak = 0
        self.max_comment_words = _MAX_COMMENT_WORDS

    def _normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower().strip()

    def _contains(self, text: str, keyword: str) -> bool:
        if " " in keyword:
            return keyword in text
        return bool(re.search(rf"\b{re.escape(keyword)}\b", text))

    def _is_invalid_caption(self, caption: str) -> bool:
        lower = self._normalize(caption)
        if lower in _INVALID_CAPTION_SIGNALS:
            return True
        return lower.startswith("error ") or "traceback" in lower

    def _extract_item(self, caption: str) -> str:
        lower = self._normalize(caption)
        has_stable_action_read = "lectura en espanol" in lower
        if not has_stable_action_read:
            return "su equipo"

        if "curacion con manzana dorada" in lower or "eating golden apple" in lower:
            return "manzana dorada"
        if "curacion con pocion" in lower or "drinking healing potion" in lower:
            return "pocion"
        if "presion con snowballs" in lower or "throwing snowballs" in lower:
            return "snowballs"
        if "presion con huevos" in lower or "throwing eggs" in lower:
            return "huevos"
        return "su equipo"

    def _extract_players_hint(self, lower: str) -> str:
        if any(self._contains(lower, token) for token in _PLAYERS_HINTS["top2"]):
            return _PLAYERS_HINT_TEXT["top2"]
        if any(self._contains(lower, token) for token in _PLAYERS_HINTS["top3"]):
            return _PLAYERS_HINT_TEXT["top3"]
        if any(self._contains(lower, token) for token in _PLAYERS_HINTS["top4"]):
            return _PLAYERS_HINT_TEXT["top4"]
        return ""

    def _extract_marker_values(self, lower: str, marker: str) -> tuple[str, ...]:
        match = re.search(rf"{re.escape(marker)}\s*:\s*([^.]*)", lower)
        if not match:
            return ()
        values = []
        for value in match.group(1).split(","):
            clean = value.strip(" ;")
            if clean:
                values.append(clean)
        return tuple(values)

    def _extract_action_tags(self, lower: str) -> tuple[str, ...]:
        tags = list(self._extract_marker_values(lower, "lectura en espanol"))
        for phrase, _ in _ACTION_TEMPLATES:
            if phrase in lower and phrase not in tags:
                tags.append(phrase)
        return tuple(tags)

    def _extract_visual_signals(self, lower: str) -> tuple[str, ...]:
        return self._extract_marker_values(lower, "senales tacticas")

    def _extract_phase_hint(self, lower: str) -> str:
        match = re.search(r"fase temporal\s*:\s*(inicio|medio|cierre)", lower)
        if match:
            return match.group(1)
        if "inicio de partida" in lower or "primeros segundos" in lower:
            return "inicio"
        if "tramo final" in lower or "endgame" in lower:
            return "cierre"
        return "medio"

    def _extract_progress_pct(self, lower: str) -> int:
        match = re.search(r"progreso video\s*:\s*(\d+)", lower)
        if not match:
            return 0
        return max(0, min(100, int(match.group(1))))

    def _extract_context(self, caption: str) -> SceneContext:
        lower = self._normalize(caption)
        item = self._extract_item(caption)
        action_tags = self._extract_action_tags(lower)
        visual_signals = self._extract_visual_signals(lower)
        phase_hint = self._extract_phase_hint(lower)
        progress_pct = self._extract_progress_pct(lower)
        return SceneContext(
            item=item,
            action_tags=action_tags,
            visual_signals=visual_signals,
            phase_hint=phase_hint,
            progress_pct=progress_pct,
            is_ranged=item in _PROJECTILE_ITEMS,
            in_mid=any(self._contains(lower, k) for k in _CONTEXT_KEYWORDS["mid"]),
            bridging=any(self._contains(lower, k) for k in _CONTEXT_KEYWORDS["bridge"]),
            void_risk=any(self._contains(lower, k) for k in _CONTEXT_KEYWORDS["void"]),
            has_height=any(self._contains(lower, k) for k in _CONTEXT_KEYWORDS["height"]),
            is_pressuring=any(self._contains(lower, k) for k in _CONTEXT_KEYWORDS["pressure"]),
            is_defensive=any(self._contains(lower, k) for k in _CONTEXT_KEYWORDS["defense"]),
            low_health=any(self._contains(lower, k) for k in _CONTEXT_KEYWORDS["low_health"]),
            players_hint=self._extract_players_hint(lower),
            is_caption_invalid=self._is_invalid_caption(caption),
        )

    def _score_categories(self, caption: str, context: SceneContext) -> dict[str, int]:
        lower = self._normalize(caption)
        scores: dict[str, int] = {}

        for category, keywords in _CATEGORY_KEYWORDS.items():
            hits = sum(1 for kw in keywords if self._contains(lower, kw))
            if hits:
                scores[category] = hits * _CATEGORY_WEIGHTS[category]

        if context.in_mid:
            scores["centro"] = scores.get("centro", 0) + 2
        if context.bridging:
            scores["puenteo"] = scores.get("puenteo", 0) + 3
        if context.void_risk:
            scores["peligro_vacio"] = scores.get("peligro_vacio", 0) + 3
        if context.low_health:
            scores["peligro_vacio"] = scores.get("peligro_vacio", 0) + 1
        if context.is_pressuring and context.item != "su equipo":
            scores["combate"] = scores.get("combate", 0) + 2
        if context.is_defensive and context.in_mid:
            scores["centro"] = scores.get("centro", 0) + 1
        if context.item != "su equipo":
            scores["loot"] = scores.get("loot", 0) + 1

        action_blob = " ".join(context.action_tags)
        if context.phase_hint == "inicio":
            scores["inicio"] = scores.get("inicio", 0) + 28
            if "victoria" not in scores:
                scores.pop("cierre", None)
        elif context.phase_hint == "medio":
            if "victoria" not in scores:
                scores.pop("cierre", None)
        elif context.phase_hint == "cierre":
            scores["cierre"] = scores.get("cierre", 0) + 6

        if any(token in action_blob for token in ["pantalla de victoria"]):
            scores["victoria"] = scores.get("victoria", 0) + 30
        if any(token in action_blob for token in ["caida al vacio", "knockback cerca del borde", "riesgo al borde del vacio"]):
            scores["peligro_vacio"] = scores.get("peligro_vacio", 0) + 12
        if any(token in action_blob for token in ["duelo melee", "intercambio corto", "presion con huevos", "presion con snowballs"]):
            scores["combate"] = scores.get("combate", 0) + 10
        if any(token in action_blob for token in ["puenteo sobre vacio", "persecucion en puente"]):
            scores["puenteo"] = scores.get("puenteo", 0) + 10
        if any(token in action_blob for token in ["loot de cofre", "gestion de inventario", "cambio rapido de equipo"]):
            scores["loot"] = scores.get("loot", 0) + 8
        if "rotacion entre islas" in action_blob:
            scores["centro"] = scores.get("centro", 0) + 5
        if "duelo final 1v1" in action_blob and context.phase_hint == "cierre":
            scores["cierre"] = scores.get("cierre", 0) + 12

        # Si hay victoria o kill explicita, forzar alta prioridad.
        for category in _PRIORITY_CATEGORIES:
            if any(self._contains(lower, kw) for kw in _CATEGORY_KEYWORDS[category]):
                scores[category] = scores.get(category, 0) + 20

        # Evitar que "loot" tape peleas reales.
        has_combat_signal = any(
            self._contains(lower, kw) for kw in _CATEGORY_KEYWORDS["combate"]
        )
        if has_combat_signal and context.item != "su equipo":
            scores["combate"] = scores.get("combate", 0) + 4

        # Si hay "last players" sin kill/victoria, reforzar cierre.
        if context.players_hint and context.phase_hint == "cierre" and "victoria" not in scores:
            scores["cierre"] = scores.get("cierre", 0) + 3

        return scores

    def _classify_scene(self, caption: str, context: SceneContext) -> str:
        if context.is_caption_invalid:
            return "general"

        scores = self._score_categories(caption, context)
        if not scores:
            return "general"

        if context.phase_hint == "inicio" and "victoria" not in scores:
            return "inicio"

        sorted_candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_category, best_score = sorted_candidates[0]

        # Estabilizador: si la diferencia es minima, conservar la categoria anterior.
        if len(sorted_candidates) > 1:
            second_score = sorted_candidates[1][1]
            if self._last_category in scores and (best_score - second_score) <= 2:
                if scores[self._last_category] >= second_score:
                    best_category = self._last_category

        return best_category

    def _pick_template(self, category: str, allow_item_template: bool = True) -> str:
        templates = _TEMPLATES.get(category, _TEMPLATES["general"])
        if not allow_item_template:
            templates = [template for template in templates if "{item}" not in template]
            if not templates:
                templates = _TEMPLATES.get(category, _TEMPLATES["general"])
        available = [template for template in templates if template not in self._used_recently]

        if not available:
            self._used_recently.clear()
            available = templates

        chosen = random.choice(available)
        self._used_recently.append(chosen)
        if len(self._used_recently) > self._max_memory:
            self._used_recently = self._used_recently[-self._max_memory :]
        return chosen

    def _pick_from_options(self, options: list[str]) -> str:
        available = [option for option in options if option not in self._used_recently]
        if not available:
            available = options
        chosen = random.choice(available)
        self._used_recently.append(chosen)
        if len(self._used_recently) > self._max_memory:
            self._used_recently = self._used_recently[-self._max_memory :]
        return chosen

    def _build_specific_comment(self, category: str, ctx: SceneContext) -> str:
        action_blob = " ".join(ctx.action_tags)

        if ctx.void_risk and any(item in action_blob for item in ["presion con huevos", "presion con snowballs"]):
            return self._pick_from_options(
                [
                    "Proyectil al borde; busca knockback inmediato.",
                    "Presiona al filo para forzar caida.",
                ]
            )

        if ctx.phase_hint == "cierre" and "duelo final 1v1" in action_blob and any(
            item in action_blob for item in ["duelo melee", "intercambio corto"]
        ):
            return self._pick_from_options(
                [
                    "1v1 final; cada hit vale partida.",
                    "Ultimo duelo, no puede fallar el trade.",
                ]
            )

        for action_name, options in _ACTION_TEMPLATES:
            if action_name in {"duelo final 1v1", "pantalla de victoria"} and ctx.phase_hint != "cierre":
                continue
            if action_name in action_blob:
                return self._pick_from_options(options)

        if category == "centro" and "movilidad" in ctx.visual_signals:
            return self._pick_from_options(
                [
                    "Rota para ganar angulo antes del choque.",
                    "Busca nueva ruta sin quedarse encerrado.",
                ]
            )
        if category == "combate" and "combate" in ctx.visual_signals:
            return self._pick_from_options(
                [
                    "Pelea activa; necesita mantener posicion.",
                    "Se arma el trade, el spacing importa.",
                ]
            )
        if category == "puenteo" and "riesgo_vacio" in ctx.visual_signals:
            return self._pick_from_options(
                [
                    "Cruce peligroso, cualquier hit castiga.",
                    "Esta expuesto sobre vacio.",
                ]
            )
        return ""

    def _build_detail_line(self, category: str, ctx: SceneContext) -> str:
        options: list[str] = []

        if category == "loot":
            if ctx.item != "su equipo":
                options.append(f"Item clave: {ctx.item}.")
            if ctx.in_mid:
                options.append("Puede disputar centro rapido.")
            if ctx.low_health:
                options.append("Necesita curarse antes del choque.")
        elif category == "puenteo":
            if ctx.void_risk:
                options.append("Un paso mal y cae.")
            if ctx.is_pressuring:
                options.append("Cruza para tomar iniciativa.")
            if ctx.players_hint:
                options.append(ctx.players_hint)
        elif category == "centro":
            if ctx.has_height:
                options.append("La altura le favorece.")
            if ctx.is_defensive:
                options.append("No regala dano gratis.")
            if ctx.players_hint:
                options.append(ctx.players_hint)
        elif category == "combate":
            if ctx.item != "su equipo":
                if "manzana" in ctx.item:
                    options.append("Se cura antes de reentrar.")
                elif "pocion" in ctx.item:
                    options.append("Reentra con mas vida.")
                elif "snowballs" in ctx.item or "huevos" in ctx.item:
                    options.append(f"Presiona con {ctx.item} para buscar knockback.")
                    options.append(f"Usa {ctx.item} para cortar el avance rival.")
                elif not ctx.is_ranged:
                    options.append(f"Juega melee con {ctx.item} buscando knockback.")
            if ctx.void_risk:
                options.append("El borde vuelve peligroso el trade.")
            if ctx.low_health:
                options.append("Vida baja; un hit decide.")
            if ctx.has_height:
                options.append("La altura pesa en el trade.")
        elif category == "eliminacion":
            if ctx.void_risk:
                options.append("La baja abre el mapa.")
            if ctx.in_mid:
                options.append("Kill en mid, control inmediato.")
            if ctx.players_hint:
                options.append(ctx.players_hint)
        elif category == "peligro_vacio":
            if ctx.bridging:
                options.append("El puente lo deja vendido.")
            if ctx.low_health:
                options.append("Tocado, necesita defender perfecto.")
            if ctx.players_hint:
                options.append(ctx.players_hint)
        elif category == "cierre":
            if ctx.has_height:
                options.append("La altura puede decidir.")
            if ctx.in_mid:
                options.append("Centro vale media partida.")
            if ctx.players_hint:
                options.append(ctx.players_hint)
        elif category == "victoria":
            if ctx.item != "su equipo":
                options.append(f"Cierra con gran uso de {ctx.item}.")
            if ctx.players_hint:
                options.append("Lo logra en el tramo mas tenso.")
        elif category == "general":
            if ctx.players_hint:
                options.append(ctx.players_hint)
            if ctx.low_health:
                options.append("Vida baja, toca jugar fino.")

        return random.choice(options) if options else ""

    def _transition_line(self, category: str) -> str:
        if self._last_category == category:
            self._category_streak += 1
        else:
            self._category_streak = 1

        if self._last_category != category:
            if category == "combate" and self._last_category in {"loot", "puenteo", "centro"}:
                return " Entra pelea inmediata."
            if category == "eliminacion":
                return " Presion convertida."
            if category == "cierre":
                return " Fase decisiva."
            if category == "victoria":
                return " Partida cerrada."

        if self._category_streak >= 3 and category in {"combate", "peligro_vacio"}:
            return " Sigue el peligro."
        return ""

    def _finalize_comment(self, base_comment: str, detail_line: str, transition: str) -> str:
        comment = base_comment
        if detail_line:
            comment = f"{comment} {detail_line}"
        if transition:
            comment = f"{comment}{transition}"
        if comment == self._last_comment:
            comment = f"{comment} Sigue siendo un punto critico."
        comment = self._compact_comment(comment)
        self._last_comment = comment
        return comment

    def _compact_comment(self, text: str) -> str:
        clean = re.sub(r"\s+", " ", text.strip())
        words = clean.split()
        max_words = max(8, min(40, int(self.max_comment_words)))
        if len(words) <= max_words:
            return clean
        trimmed = " ".join(words[:max_words]).rstrip(",;:")
        if not trimmed.endswith((".", "!", "?")):
            trimmed += "."
        return trimmed

    def generate_comment(self, caption: str) -> tuple[str, str]:
        context = self._extract_context(caption)
        if context.is_caption_invalid:
            template = self._pick_template("fallback_error")
            self._last_category = "general"
            return template, "general"

        category = self._classify_scene(caption, context)
        specific_comment = self._build_specific_comment(category, context)
        if specific_comment:
            base_comment = specific_comment
            detail_line = ""
            transition = ""
        else:
            template = self._pick_template(category, allow_item_template=context.item != "su equipo")
            base_comment = template.format(item=context.item)
            detail_line = self._build_detail_line(category, context)
            transition = self._transition_line(category)
        comment = self._finalize_comment(base_comment, detail_line, transition)
        self._last_category = category
        return comment, category


def narrate_frames(filtered_frames: list[dict], model: NarratorModel) -> list[dict]:
    """Agrega comentario narrado y categoria a cada frame filtrado."""
    narrated_frames: list[dict] = []

    for frame in filtered_frames:
        updated = dict(frame)
        try:
            caption = str(frame.get("caption", ""))
            comment, category = model.generate_comment(caption)
            updated["comment"] = comment
            updated["category"] = category
        except Exception:
            updated["comment"] = "Partida intensa de Skywars en este momento."
            updated["category"] = "general"
        narrated_frames.append(updated)

    return narrated_frames
