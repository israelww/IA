"""
24-Puzzle Solver using IDA*

Requisitos:
- Implementa el 24-puzzle (5x5) con espacio vacío al final.
- Algoritmo: IDA* (Iterative Deepening A*)
- Permite alternar entre heurísticas: Manhattan y Conflicto Lineal (opcional).
- Justificación: IDA* es superior a A* en memoria para espacios de estados grandes, ya que combina la profundidad iterativa con la heurística de A*, evitando almacenar todos los nodos abiertos en memoria.
"""

import random
import heapq
import time
from typing import List, Tuple, Optional, Callable, Set
  
GOAL_STATE = tuple(list(range(1, 25)) + [0])  # 0 es el espacio vacío
SIZE = 5

class Puzzle24:
    def __init__(self, state: Tuple[int, ...]):
        self.state = state

    
    def random_state(moves=100) -> Tuple[int, ...]:
        state = list(GOAL_STATE)
        puzzle = Puzzle24(tuple(state))
        for _ in range(moves):
            moves_ = puzzle.possible_moves()
            move = random.choice(moves_)
            puzzle = puzzle.move(move)
        return puzzle.state

    def possible_moves(self) -> List[str]:
        idx = self.state.index(0)
        row, col = divmod(idx, SIZE)
        moves = []
        if row > 0: moves.append('up')
        if row < SIZE - 1: moves.append('down')
        if col > 0: moves.append('left')
        if col < SIZE - 1: moves.append('right')
        return moves

    def move(self, direction: str) -> 'Puzzle24':
        idx = self.state.index(0)
        row, col = divmod(idx, SIZE)
        swap_idx = None
        if direction == 'up': swap_idx = idx - SIZE
        elif direction == 'down': swap_idx = idx + SIZE
        elif direction == 'left': swap_idx = idx - 1
        elif direction == 'right': swap_idx = idx + 1
        else: raise ValueError('Invalid move')
        state = list(self.state)
        state[idx], state[swap_idx] = state[swap_idx], state[idx]
        return Puzzle24(tuple(state))

    def is_goal(self) -> bool:
        return self.state == GOAL_STATE

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __str__(self):
        s = ''
        for i in range(SIZE):
            s += ' '.join(f'{x:2d}' if x != 0 else '  ' for x in self.state[i*SIZE:(i+1)*SIZE]) + '\n'
        return s

# Heurísticas

def manhattan(state: Tuple[int, ...]) -> int:
    dist = 0
    for idx, val in enumerate(state):
        if val == 0:
            continue
        goal_idx = val - 1
        x1, y1 = divmod(idx, SIZE)
        x2, y2 = divmod(goal_idx, SIZE)
        dist += abs(x1 - x2) + abs(y1 - y2)
    return dist

# Opcional: conflicto lineal

def linear_conflict(state: Tuple[int, ...]) -> int:
    manh = manhattan(state)
    conflict = 0
    # Filas
    for row in range(SIZE):
        max_seen = -1
        for col in range(SIZE):
            idx = row * SIZE + col
            val = state[idx]
            if val != 0 and (val - 1) // SIZE == row:
                if val > max_seen:
                    max_seen = val
                else:
                    conflict += 2
    # Columnas
    for col in range(SIZE):
        max_seen = -1
        for row in range(SIZE):
            idx = row * SIZE + col
            val = state[idx]
            if val != 0 and (val - 1) % SIZE == col:
                if val > max_seen:
                    max_seen = val
                else:
                    conflict += 2
    return manh + conflict

# IDA*
def ida_star(start: Tuple[int, ...], heuristic: Callable[[Tuple[int, ...]], int]) -> Tuple[Optional[List[str]], int]:
    threshold = heuristic(start)
    path = [start]
    moves = []
    expanded_nodes = 0

    def search(g, threshold):
        nonlocal expanded_nodes
        node = path[-1]
        f = g + heuristic(node)
        if f > threshold:
            return f
        if node == GOAL_STATE:
            return 'FOUND'
        min_threshold = float('inf')
        puzzle = Puzzle24(node)
        for move in puzzle.possible_moves():
            child = puzzle.move(move).state
            if child in path:
                continue
            expanded_nodes += 1
            path.append(child)
            moves.append(move)
            t = search(g + 1, threshold)
            if t == 'FOUND':
                return 'FOUND'
            if t < min_threshold:
                min_threshold = t
            path.pop()
            moves.pop()
        return min_threshold

    while True:
        t = search(0, threshold)
        if t == 'FOUND':
            return moves.copy(), expanded_nodes
        if t == float('inf'):
            return None, expanded_nodes
        threshold = t

if __name__ == "__main__":
    print("24-Puzzle Solver (IDA*)\n")
    print("Generando estado inicial aleatorio...")
    print("")
    initial = Puzzle24.random_state(moves=50)
    puzzle = Puzzle24(initial)
    print("Estado inicial:")
    print(puzzle)
    print("     Seleccione heurística:")
    print("1. Manhattan\n2. Conflicto Lineal")
    h = input("Opción [1/2]: ")
    heuristic = manhattan if h.strip() == '1' else linear_conflict
    print("Buscando solución...")
    start_time = time.time()
    solution, expanded_nodes = ida_star(initial, heuristic)
    elapsed = time.time() - start_time
    if solution:
        print(f"Solución encontrada en {len(solution)} movimientos:")
        print(solution)
    else:
        print("No se encontró solución.")
    print("======================================")
    print(f"Nodos expandidos: {expanded_nodes}")
    
    print(f"Tiempo de ejecución: {elapsed:.3f} segundos")
