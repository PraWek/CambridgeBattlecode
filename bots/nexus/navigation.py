from collections import deque
from heapq import heappop, heappush

from cambc import Controller, Position

from constants import LARGE_NUMBER, ORTHOGONAL_DIRECTIONS
from geometry import chebyshev


def a_star_to_any(
        controller: Controller,
        start: Position,
        goals: set[Position],
        traversable_fn,
        neighbor_fn,
        preferred_tiles: set[Position] | None = None,
        movement_directions=ORTHOGONAL_DIRECTIONS,
        extra_step_cost_fn=None,
        max_expansions: int | None = None,
) -> list[Position]:
    """Find a lowest-cost A* path from ``start`` to any traversable goal tile.

    ``max_expansions`` bounds a search whose target is not reachable through
    the currently known map.  On exhaustion the function returns an empty
    path, just as it does when no route exists, so the caller can defer the
    job and continue scouting.
    """
    if start in goals:
        return []

    queue = [(0, 0, start)]
    came_from = {start: start}
    g_score = {start: 0}
    if preferred_tiles is None:
        preferred_tiles = set()
    minimum_step_cost = 1 if preferred_tiles else 4
    expansions = 0

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
        if max_expansions is not None and expansions >= max_expansions:
            return []
        expansions += 1

        for direction in movement_directions:
            next_pos = neighbor_fn(current, direction)
            if next_pos is None:
                continue
            if not traversable_fn(controller, next_pos):
                continue
            step_cost = 1 if next_pos in preferred_tiles else 4
            if extra_step_cost_fn is not None:
                step_cost += extra_step_cost_fn(next_pos)
            new_cost = cost + step_cost
            if new_cost >= g_score.get(next_pos, LARGE_NUMBER):
                continue
            g_score[next_pos] = new_cost
            came_from[next_pos] = current
            heuristic = minimum_step_cost * min(
                chebyshev(next_pos, goal) for goal in goals
            )
            heappush(queue, (new_cost + heuristic, new_cost, next_pos))

    return []


def breadth_first_sweep_path(
        start: Position,
        traversable_fn,
        neighbor_fn,
        visit_counts: dict[Position, int],
        movement_directions=ORTHOGONAL_DIRECTIONS,
        max_expansions: int | None = None,
) -> list[Position]:
    """Route to a far, least-visited tile in the known connected component.

    Patrol has no single semantic goal, so repeatedly running bounded A* at a
    handful of distant candidates is the wrong shape: hitting the bound does
    not prove those candidates unreachable.  One BFS discovers only genuinely
    connected tiles and selects a stable sweep endpoint from that set.
    """
    queue = deque([start])
    came_from = {start: start}
    distance = {start: 0}
    best: Position | None = None
    best_rank: tuple[int, int, int, int] | None = None
    expansions = 0

    while queue:
        if max_expansions is not None and expansions >= max_expansions:
            break
        current = queue.popleft()
        expansions += 1
        if current != start:
            rank = (
                -distance[current],
                visit_counts.get(current, 0),
                current.x,
                current.y,
            )
            if best_rank is None or rank < best_rank:
                best = current
                best_rank = rank
        for direction in movement_directions:
            next_pos = neighbor_fn(current, direction)
            if (
                next_pos is None
                or next_pos in came_from
                or not traversable_fn(next_pos)
            ):
                continue
            came_from[next_pos] = current
            distance[next_pos] = distance[current] + 1
            queue.append(next_pos)

    if best is None:
        return []
    path = []
    current = best
    while current != start:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path
