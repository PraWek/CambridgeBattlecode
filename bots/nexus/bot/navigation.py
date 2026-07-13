from __future__ import annotations
from heapq import heappop, heappush
from cambc import Controller, Position
from bot.constants import LARGE_NUMBER, ORTHOGONAL_DIRECTIONS
from bot.geometry import chebyshev


def a_star_to_any(ct: Controller, start: Position, goals: set[Position], traversable_fn,
                  preferred_tiles: set[Position] | None = None) -> list[Position]:
    if start in goals:
        return []

    queue = []
    came_from = {start: start}
    g_score = {start: 0}
    heappush(queue, (0, 0, start))

    if preferred_tiles is None:
        preferred_tiles = set()

    while queue:
        _, cost, current = heappop(queue)
        if current in goals:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        if cost != g_score[current]:
            continue

        for direction in ORTHOGONAL_DIRECTIONS:
            nxt = current.add(direction)
            if not traversable_fn(ct, nxt):
                continue

            # Сильно удешевляем стоимость шага, если клетка лежит на ветке дерева Штейнера
            step_cost = 1 if nxt in preferred_tiles else 4
            new_cost = cost + step_cost

            if new_cost >= g_score.get(nxt, LARGE_NUMBER):
                continue
            g_score[nxt] = new_cost
            came_from[nxt] = current

            # Эвристика должна быть согласована с минимальным step_cost = 1
            heuristic = min(chebyshev(nxt, goal) for goal in goals) * 1
            heappush(queue, (new_cost + heuristic, new_cost, nxt))

    return []