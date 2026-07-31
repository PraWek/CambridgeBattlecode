from __future__ import annotations

from collections import deque
from heapq import heappop, heappush

from cambc import Controller, Environment, EntityType, Position

from bot.constants import LARGE_NUMBER, ORTHOGONAL_DIRECTIONS, PASSABLE_BUILDINGS, ORE_TYPES
from bot.geometry import chebyshev


def a_star_to_any(
    ct: Controller,
    start: Position,
    goals: set[Position],
    traversable_fn,
    preferred_tiles: set[Position] | None = None,
    conveyor_tiles: set[Position] | None = None,
) -> list[Position]:
    """
    A* от start до любой из целей
    - preferred_tiles (ветки Штейнера): шаг стоит 1, иначе 4
    - conveyor_tiles (существующие конвейеры чужих веток): шаг стоит 8 (nocross-пенальти)
    """
    if start in goals:
        return []

    if preferred_tiles is None:
        preferred_tiles = set()
    if conveyor_tiles is None:
        conveyor_tiles = set()

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

            if nxt in preferred_tiles:
                step_cost = 1
            elif nxt in conveyor_tiles:
                step_cost = 8
            else:
                step_cost = 4

            new_cost = cost + step_cost
            if new_cost >= g_score.get(nxt, LARGE_NUMBER):
                continue
            g_score[nxt] = new_cost
            came_from[nxt] = current

            heuristic = min(chebyshev(nxt, goal) for goal in goals)
            heappush(queue, (new_cost + heuristic, new_cost, nxt))

    return []


def score_reachable_tiles(
    start: Position,
    known_env: dict,
    known_buildings: dict,
    map_w: int,
    map_h: int,
    team,
) -> float:
    """
    BFS-оценка позиции: сколько доступной/неизвестной территории открывается из start
    """
    queue: deque[Position] = deque([start])
    visited: set[tuple[int, int]] = {(start.x, start.y)}
    score = 0.0
    decay = 0.93
    multi = 1.0

    while queue and multi >= 0.25:
        pos = queue.popleft()

        for d in ORTHOGONAL_DIRECTIONS:
            nxt = pos.add(d)
            key = (nxt.x, nxt.y)
            if key in visited:
                continue
            if not (0 <= nxt.x < map_w and 0 <= nxt.y < map_h):
                continue
            visited.add(key)

            env = known_env.get(nxt)
            if env is None:
                score += 3.0 * multi
                multi *= decay
                continue
            if env == Environment.WALL or env in ORE_TYPES:
                continue

            building_info = known_buildings.get(nxt)
            if building_info is not None:
                btype, bteam = building_info
                if bteam != team:
                    continue  # вражеская постройка блокирует
                if btype not in PASSABLE_BUILDINGS:
                    score -= 1.0 * multi
                    continue

            score += 1.0 * multi
            multi *= decay
            queue.append(nxt)

    return score


def find_existing_conveyor_tiles(
    known_buildings: dict,
    steiner_parent: dict[Position, Position],
    team,
) -> set[Position]:
    """
    Возвращает тайлы с конвейерами союзных веток, которые НЕ являются частью
    текущего дерева Штейнера (чтобы ввести nocross-пенальти в A*)
    """
    steiner_set = set(steiner_parent.keys())
    result: set[Position] = set()
    for pos, info in known_buildings.items():
        if info is None:
            continue
        btype, bteam = info
        if bteam != team:
            continue
        if btype not in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
            continue
        if pos not in steiner_set:
            result.add(pos)
    return result
