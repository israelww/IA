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
    "mid": ["mid", "center", "centro", "middle island", "middle", "zona central"],
    "bridge": ["bridge", "bridging", "placing blocks", "speed bridge", "ninja bridge", "puente", "puenteo", "cruce entre islas"],
    "void": ["void", "vacio", "falling", "fell", "edge", "borde", "filo", "knocked", "clutch", "mlg water", "cerca del vacio"],
    "height": ["high ground", "tower", "top", "elevated", "above", "altura", "torre"],
    "pressure": ["rush", "pushing", "chasing", "sprinting", "aggressive", "presion", "persecucion", "amenaza directa", "proyectiles"],
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
        "Arranca la ronda; toca leer cofres, ruta y primer rival.",
        "Primer movimiento: saqueo rapido y salida con decision.",
        "Empieza Skywars; cada segundo perdido abre la puerta al rush.",
        "Inicio con tension: necesita loot y ruta sin quedarse quieto.",
        "Sale la partida y ya importa quien toma la iniciativa.",
        "Primeros compases; mira recursos y prepara el cruce.",
        "Apertura activa, si encuentra equipo puede acelerar.",
        "Se despierta el mapa; hay que decidir entre loot o presion.",
        "Comienza el round, la ventaja nace desde el primer cofre.",
        "Apenas inicia y ya busca tempo para no quedar encerrado.",
    ],
    "loot": [
        "Abre cofre y acelera la salida.",
        "Ordena recursos para pelear sin perder tempo.",
        "Buen loot; ahora importa salir antes del rival.",
        "Se equipa con {item} y prepara presion.",
        "Inventario rapido para entrar mejor al duelo.",
        "Consigue recursos y busca la siguiente rotacion.",
        "Loot veloz; si no duda, puede tomar mapa.",
        "Revisa inventario y arma la siguiente jugada.",
        "Cada item cuenta, pero quedarse mucho aqui es peligroso.",
        "Se abastece rapido y ya piensa en la salida.",
        "Recolecta recursos; la pelea se prepara desde la hotbar.",
        "Buen momento para ordenar equipo antes de cruzar.",
        "Encuentra utilidad y puede convertirla en presion.",
        "Saca valor del cofre y no quiere perder ritmo.",
        "Gestion de items clave, ahora toca moverse.",
        "Esta preparando el kit para el primer choque serio.",
    ],
    "puenteo": [
        "Cruce expuesto; un golpe puede mandarlo al vacio.",
        "Puenteo con presion, cada bloque cuenta.",
        "Se abre camino entre islas buscando iniciativa.",
        "Rotacion arriesgada sobre vacio.",
        "Cruza con riesgo, aqui el knockback castiga durisimo.",
        "Va por el puente; si lo leen, queda vendido.",
        "Movimiento valiente entre islas, pero no puede fallar.",
        "Se lanza a rotar y el vacio mete presion.",
        "Cada bloque lo acerca a mid, cada segundo lo expone.",
        "Puenteo tenso; necesita terminar antes de recibir presion.",
        "El cruce abre mapa, pero tambien abre peligro.",
        "Busca nueva isla sin perder la iniciativa.",
        "Rotacion fina: un mal paso cambia la ronda.",
        "Avanza sobre vacio con el timing encima.",
    ],
    "centro": [
        "Gana espacio en mid y controla rutas.",
        "Pisa centro para castigar rotaciones tardias.",
        "Mid le da recursos y angulos de presion.",
        "Buena rotacion al centro; llega con iniciativa.",
        "Toma zona central y empieza a dictar el ritmo.",
        "Centro en disputa; quien controle rutas controla pelea.",
        "Llega a mid y fuerza al resto a reaccionar.",
        "Buena posicion, desde aqui puede leer todo el mapa.",
        "Gana terreno importante para cortar rotaciones.",
        "Mid se vuelve clave: recursos, altura y presion.",
        "Se instala en una zona que puede decidir la partida.",
        "Control central, pero tiene que vigilar los flancos.",
        "Rota a la zona fuerte y amenaza cualquier cruce.",
        "El mapa se abre desde mid; buena lectura de posicion.",
    ],
    "combate": [
        "Duelo abierto; el primer combo pesa mucho.",
        "Trade cerrado, el knockback decide.",
        "Entra al intercambio buscando ventaja.",
        "Pelea directa; posicion antes que ego.",
        "Presiona con {item}; busca cortar el avance.",
        "Choque rapido, no puede regalar rango.",
        "Ojo al trade, se puede romper en un combo.",
        "Se prende el duelo; spacing y knockback mandan.",
        "Choque abierto, no hay margen para regalar el primer hit.",
        "La pelea sube de ritmo; quien controle distancia gana.",
        "Intercambio caliente, cada golpe cambia la ventaja.",
        "Se va encima buscando forzar error rival.",
        "Duelo peligroso, necesita mantener la mira estable.",
        "Presion directa; si conecta combo, abre la ronda.",
        "Esto ya es pelea real, posicion y timing pesan todo.",
        "Se cruzan golpes y el mapa empieza a decidir.",
        "Momento de manos: no puede perder el control del rango.",
        "Aprieta el ritmo y obliga al rival a defender.",
    ],
    "eliminacion": [
        "Baja confirmada; se abre el mapa.",
        "Convierte la presion en eliminacion.",
        "El rival cae y cambia el cierre.",
        "Kill importante; gana espacio inmediato.",
        "Eliminacion clave, ahora tiene mucho mas aire.",
        "Saca la baja y el mapa se inclina a su favor.",
        "Presion bien cobrada; un rival menos en la ronda.",
        "Consigue la kill y puede tomar recursos sin pausa.",
        "Baja que pesa, especialmente si habia presion encima.",
        "Lo resuelve y transforma el caos en ventaja.",
        "Castiga el error rival y gana control.",
        "Kill limpia, ahora toca convertirla en posicion.",
    ],
    "peligro_vacio": [
        "Al borde del vacio; un hit decide.",
        "Zona peligrosa, no puede regalar knockback.",
        "Se juega la vida en el borde.",
        "Momento delicado sobre vacio.",
        "Mucho cuidado, el vacio esta demasiado cerca.",
        "Aqui no hay segunda oportunidad: un empujon termina todo.",
        "La posicion es critica, cualquier trade puede ser fatal.",
        "Esta caminando sobre una linea muy fina.",
        "El borde mete panico, necesita salir de ahi ya.",
        "Situacion incomoda: defender mal significa caer.",
        "El mapa aprieta y el vacio castiga cualquier error.",
        "Zona roja, tiene que elegir entre pelear o escapar.",
        "Se complica la posicion, el knockback manda.",
        "Vacio al lado, decision rapida o ronda perdida.",
    ],
    "cierre": [
        "Tramo final; cada rotacion vale oro.",
        "Endgame tenso, posicion antes que ego.",
        "Final cerrado; paciencia y altura deciden.",
        "Quedan pocos, no puede regalar espacio.",
        "La partida entra en zona caliente, cada error cuesta.",
        "Cierre de ronda, recursos y calma pesan demasiado.",
        "Momento decisivo: mejor posicion puede valer la win.",
        "Ya no hay jugadas gratis, todo se castiga.",
        "Endgame apretado; necesita leer antes de entrar.",
        "El mapa se achica mentalmente, cada paso importa.",
        "Fase final, ahora el tempo vale partida.",
        "Pocos jugadores, mucha tension y cero margen.",
        "Tiene que convertir posicion en cierre.",
        "Se viene el momento donde una rotacion gana o pierde.",
    ],
    "victoria": [
        "Victoria cerrada con buena lectura.",
        "Se lleva el Skywars con autoridad.",
        "Win confirmado; ejecucion limpia.",
        "Cierra la partida sin dar opciones.",
        "Partida cerrada, gran remate para asegurar la win.",
        "Victoria en pantalla, lectura solida de principio a fin.",
        "Lo termina bien y firma una ronda muy seria.",
        "Win merecida, con control en los momentos clave.",
        "Cierre perfecto, no deja escapar la ventaja.",
        "Se queda con la ronda y apaga cualquier respuesta.",
        "Final contundente, Skywars resuelto.",
        "La victoria cae con buen manejo del cierre.",
    ],
    "general": [
        "Ronda abierta; toca leer la siguiente rotacion.",
        "Momento de pausa, pero el mapa sigue vivo.",
        "Busca informacion antes de comprometerse.",
        "Se mantiene activo, esperando ventana clara.",
        "La ronda respira, pero cualquier cruce puede activar pelea.",
        "Se reposiciona y busca una lectura mas clara.",
        "No todo es pelea: tambien importa elegir bien la ruta.",
        "Momento de lectura, el siguiente movimiento pesa.",
        "Mantiene ritmo mientras revisa opciones del mapa.",
        "Aun no se define, pero la presion esta creciendo.",
        "Busca informacion sin regalar posicion.",
        "La escena esta abierta; falta encontrar la ventana buena.",
        "Se mueve con cautela, esperando el error rival.",
        "Hay tension latente, cualquier contacto puede explotar.",
    ],
    "fallback_error": [
        "Escena confusa, pero la partida sigue intensa y abierta.",
        "No se ve claro el detalle, mantenemos foco en el ritmo del juego.",
        "Frame ambiguo, seguimos leyendo la siguiente accion clave.",
        "La vision no lo deja perfecto, pero el ritmo sigue vivo.",
        "Lectura dificil; mejor narrar posicion y tempo sin inventar.",
        "No hay detalle fiable, seguimos con la lectura general del mapa.",
    ],
}

_ACTION_TEMPLATES: list[tuple[str, list[str]]] = [
    (
        "pantalla de victoria",
        [
            "Pantalla de victoria; la ronda queda cerrada.",
            "Win en pantalla, cierre limpio.",
            "Victoria visible, ya no hay vuelta atras.",
            "La pantalla confirma el cierre de la partida.",
            "Ronda terminada, remate firme.",
            "Aparece la win y se acaba la tension.",
        ],
    ),
    (
        "caida al vacio",
        [
            "Caida al vacio; castigo total al posicionamiento.",
            "Se va al vacio y la pelea cambia al instante.",
            "Pierde el suelo y el mapa cobra la factura.",
            "El vacio decide la jugada de golpe.",
            "No alcanza a salvarse, caida durisima.",
            "Un mal trade termina directo abajo.",
        ],
    ),
    (
        "knockback cerca del borde",
        [
            "Knockback cerca del borde; un hit mas decide.",
            "Lo empuja al filo y fuerza panico defensivo.",
            "Lo manda hacia el borde y sube la tension.",
            "Ese empujon casi rompe la ronda.",
            "El knockback lo deja en zona critica.",
            "Empuje peligroso, ahora tiene que estabilizarse.",
        ],
    ),
    (
        "riesgo al borde del vacio",
        [
            "Esta al borde; no puede aceptar otro trade.",
            "Zona de vacio, cualquier golpe lo sentencia.",
            "Muy cerca del vacio, toca jugar perfecto.",
            "El borde esta metiendo toda la presion.",
            "No puede retroceder mas, el mapa lo encierra.",
            "Peligro maximo, un error y desaparece.",
        ],
    ),
    (
        "puenteo sobre vacio",
        [
            "Puenteo sobre vacio; cruce muy castigable.",
            "Cruza expuesto, necesita terminar rapido.",
            "Va sobre el vacio, cada bloque pesa.",
            "Rotacion peligrosa, si lo ven queda vendido.",
            "Puentea con riesgo y busca no perder tempo.",
            "Cruce tenso, tiene que mirar rival y camino a la vez.",
        ],
    ),
    (
        "persecucion en puente",
        [
            "Persecucion en puente; el knockback manda.",
            "Lo corre en linea recta, buscando el golpe final.",
            "Presiona en puente, terreno perfecto para empujar.",
            "La persecucion se vuelve peligrosa sobre el cruce.",
            "Lo sigue de cerca y amenaza caida inmediata.",
            "Puente bajo presion, aqui no hay espacio para fallar.",
        ],
    ),
    (
        "busca altura con torre",
        [
            "Sube por altura y obliga al rival a mirar arriba.",
            "Toma verticalidad para pelear con ventaja.",
            "Busca altura para mandar el ritmo del duelo.",
            "Se eleva y cambia el angulo de la pelea.",
            "Torre rapida, quiere pelear desde ventaja.",
            "Gana verticalidad y fuerza al rival a incomodarse.",
        ],
    ),
    (
        "duelo final 1v1",
        [
            "1v1 final; aqui no hay margen.",
            "Duelo final, cada hit vale partida.",
            "Ultimo cruce, una mala entrada define todo.",
            "Mano a mano total, el siguiente combo puede cerrar.",
            "Final de nervios, cada pixel de rango importa.",
            "La ronda se reduce a un duelo directo.",
        ],
    ),
    (
        "curacion con manzana dorada",
        [
            "Se cura antes de reentrar al trade.",
            "Manzana dorada para comprar segundos clave.",
            "Come manzana y prepara una reentrada mas fuerte.",
            "Buena pausa de curacion antes del siguiente choque.",
            "Compra vida extra para no regalar el trade.",
            "Se blinda con manzana, quiere volver a pelear.",
        ],
    ),
    (
        "curacion con pocion",
        [
            "Pocion usada para reiniciar la pelea.",
            "Recupera vida y busca volver con tempo.",
            "Pocion a tiempo para sostener la presion.",
            "Se cura rapido y evita quedar vendido.",
            "Reset defensivo, ahora puede volver al intercambio.",
            "Toma aire con la pocion antes de reentrar.",
        ],
    ),
    (
        "presion con huevos",
        [
            "Huevos para cortar avance y buscar knockback.",
            "Tira huevos para romper el timing rival.",
            "Proyectiles rapidos para negar el cruce.",
            "Usa huevos para incomodar y abrir distancia.",
            "Presion barata pero peligrosa, busca empujar.",
            "Corta el avance con huevos y cambia el ritmo.",
        ],
    ),
    (
        "presion con snowballs",
        [
            "Snowballs para frenar entrada y abrir combo.",
            "Presiona con snowballs, buscando descolocar.",
            "Snowballs al frente para romper el spacing rival.",
            "Usa snowballs y busca convertirlas en combo.",
            "Proyectiles de presion, no deja entrar comodo.",
            "Frena el avance con snowballs y gana segundos.",
        ],
    ),
    (
        "loot de cofre",
        [
            "Cofre rapido; quiere salir antes del rush.",
            "Loot directo, sin regalar segundos.",
            "Abre cofre y busca el item que cambie el inicio.",
            "Saqueo veloz, necesita convertirlo en ruta.",
            "Toma recursos y ya tiene que pensar en moverse.",
            "Loot en pantalla, cada segundo aqui cuenta.",
        ],
    ),
    (
        "gestion de inventario",
        [
            "Ajusta inventario para entrar mas limpio.",
            "Ordena la hotbar antes del siguiente choque.",
            "Gestiona items para no fallar en la pelea.",
            "Hotbar en orden, preparando la siguiente entrada.",
            "Se toma un segundo para armar bien la mano.",
            "Acomoda recursos, decision pequena pero clave.",
        ],
    ),
    (
        "cambio rapido de equipo",
        [
            "Cambia equipo rapido y prepara reentrada.",
            "Swap veloz; busca llegar mejor al duelo.",
            "Cambia piezas y mejora antes de exponerse.",
            "Ajuste rapido de equipo para pelear con ventaja.",
            "Swap oportuno, no quiere entrar mal armado.",
            "Se reequipa y sube sus opciones en el choque.",
        ],
    ),
    (
        "rotacion entre islas",
        [
            "Rota entre islas buscando mejor angulo.",
            "Cambia de isla para no quedar encerrado.",
            "Busca otra ruta y cambia el angulo de presion.",
            "Rotacion inteligente para evitar quedar atrapado.",
            "Se mueve entre islas y abre nuevas opciones.",
            "Cambio de posicion, quiere leer mejor el mapa.",
        ],
    ),
    (
        "posicion defensiva",
        [
            "Se cubre y fuerza al rival a iniciar.",
            "Defiende posicion, esperando error rival.",
            "Aguanta el angulo y no regala entrada.",
            "Se planta defensivo, obligando al rival a exponerse.",
            "Juega paciente y busca castigar la prisa rival.",
            "Posicion cerrada, quiere que el otro cometa el error.",
        ],
    ),
    (
        "duelo melee",
        [
            "Melee directo; el rango decide el trade.",
            "Entra al cuerpo a cuerpo buscando combo.",
            "Espada contra espada, aqui mandan los hits limpios.",
            "Duelo cerrado, un combo cambia toda la ronda.",
            "Se mete al melee y fuerza respuesta inmediata.",
            "Cuerpo a cuerpo intenso, no puede perder distancia.",
        ],
    ),
    (
        "intercambio corto",
        [
            "Trade corto; cada golpe mueve la ventaja.",
            "Intercambio rapido, no hay espacio gratis.",
            "Golpes rapidos, la ventaja cambia en segundos.",
            "Trade breve pero peligroso, ojo al knockback.",
            "Choque corto, perfecto para sacar combo.",
            "Intercambio de reflejos, nada esta seguro.",
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
_MAX_COMMENT_WORDS = 26


@dataclass
class SceneContext:
    item: str = "su equipo"
    scene_read: str = ""
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
        self._max_memory = 28
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

    def _extract_item(self, caption: str, action_tags: tuple[str, ...] = ()) -> str:
        lower = self._normalize(caption)
        has_stable_action_read = "lectura en espanol" in lower
        if not has_stable_action_read:
            return "su equipo"

        for action_tag in action_tags:
            if "manzana dorada" in action_tag:
                return "manzana dorada"
            if "pocion" in action_tag:
                return "pocion"
            if "snowballs" in action_tag:
                return "snowballs"
            if "huevos" in action_tag:
                return "huevos"
            if action_tag in {"loot de cofre", "gestion de inventario", "cambio rapido de equipo"}:
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
        return self._extract_marker_values(lower, "señales tacticas")

    def _extract_scene_read(self, lower: str) -> str:
        for marker in ["escena percibida", "escena"]:
            match = re.search(rf"{re.escape(marker)}\s*:\s*([^.]*)", lower)
            if match:
                return match.group(1).strip(" ;")
        return ""

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
        scene_read = self._extract_scene_read(lower)
        action_tags = self._extract_action_tags(lower)
        item = self._extract_item(caption, action_tags)
        visual_signals = self._extract_visual_signals(lower)
        phase_hint = self._extract_phase_hint(lower)
        progress_pct = self._extract_progress_pct(lower)
        return SceneContext(
            item=item,
            scene_read=scene_read,
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
        action_options_by_name = dict(_ACTION_TEMPLATES)

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

        for action_name in ctx.action_tags:
            if action_name in {"duelo final 1v1", "pantalla de victoria"} and ctx.phase_hint != "cierre":
                continue
            options = action_options_by_name.get(action_name)
            if options:
                return self._pick_from_options(options)

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
                    "Cambia de carril y amenaza otra entrada.",
                    "Se mueve para llegar con mejor timing.",
                    "Rotacion viva, quiere encontrar el angulo bueno.",
                ]
            )
        if category == "combate" and "combate" in ctx.visual_signals:
            return self._pick_from_options(
                [
                    "Pelea activa; necesita mantener posicion.",
                    "Se arma el trade, el spacing importa.",
                    "El duelo se calienta y el primer combo pesa.",
                    "Hay contacto directo, toca cuidar el rango.",
                    "La pelea esta viva y cualquier hit cambia todo.",
                    "Se siente presion de combate, no puede dudar.",
                    "El intercambio amenaza con romper la ronda.",
                ]
            )
        if category == "puenteo" and "riesgo_vacio" in ctx.visual_signals:
            return self._pick_from_options(
                [
                    "Cruce peligroso, cualquier hit castiga.",
                    "Esta expuesto sobre vacio.",
                    "Puente y vacio, combinacion de mucho riesgo.",
                    "Rotacion castigable, necesita cruzar ya.",
                    "Cada bloque suma tension en este cruce.",
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
