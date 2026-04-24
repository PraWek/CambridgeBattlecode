from __future__ import annotations

from heapq import heappop, heappush

from cambc import Controller, Position

from bot.constants import LARGE_NUMBER, ORTHOGONAL_DIRECTIONS
from bot.geometry import chebyshev


def a_star_to_any(ct: Controller, start: Position, goals: set[Position], traversable_fn) -> list[Position]:
    if start in goals:
        return []

    queue = []
    came_from = {start: start}
    g_score = {start: 0}
    heappush(queue, (0, 0, start))

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
            new_cost = cost + 1
            if new_cost >= g_score.get(nxt, LARGE_NUMBER):
                continue
            g_score[nxt] = new_cost
            came_from[nxt] = current
            heuristic = min(chebyshev(nxt, goal) for goal in goals)
            heappush(queue, (new_cost + heuristic, new_cost, nxt))

    return []
