from heapq import heappop, heappush

from cambc import Controller, Direction, Position

from constants import LARGE_NUMBER, ORTHOGONAL_DIRECTIONS
from geometry import chebyshev


def a_star_to_any(
        controller: Controller,
        start: Position,
        goals: set[Position],
        traversable_fn,
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
            next_pos = current.add(direction)
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


def a_star_from_any(
        controller: Controller,
        starts: set[Position],
        goals: set[Position],
        traversable_fn,
        movement_directions=ORTHOGONAL_DIRECTIONS,
        max_expansions: int | None = None,
) -> list[Position]:
    """Find a shortest A* path from any start tile to any goal tile.

    The returned route includes its selected start tile, unlike
    :func:`a_star_to_any`, because callers such as conveyor planners must
    construct infrastructure on that first tile.  All steps have equal cost;
    the multi-source frontier lets nearby starts compete in a single bounded
    search instead of running one search for each possible source.
    """
    if not starts or not goals:
        return []

    queue: list[tuple[int, int, int, Position]] = []
    came_from: dict[Position, Position | None] = {}
    g_score: dict[Position, int] = {}
    sequence = 0
    for start in sorted(starts, key=lambda pos: (pos.y, pos.x)):
        came_from[start] = None
        g_score[start] = 0
        heuristic = min(chebyshev(start, goal) for goal in goals)
        heappush(queue, (heuristic, 0, sequence, start))
        sequence += 1

    expansions = 0
    while queue:
        _, cost, _, current = heappop(queue)
        if cost != g_score[current]:
            continue
        if current in goals:
            path: list[Position] = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        if max_expansions is not None and expansions >= max_expansions:
            return []
        expansions += 1

        for direction in movement_directions:
            next_pos = current.add(direction)
            if not traversable_fn(controller, next_pos):
                continue
            new_cost = cost + 1
            if new_cost >= g_score.get(next_pos, LARGE_NUMBER):
                continue
            g_score[next_pos] = new_cost
            came_from[next_pos] = current
            heuristic = min(chebyshev(next_pos, goal) for goal in goals)
            heappush(
                queue,
                (new_cost + heuristic, new_cost, sequence, next_pos),
            )
            sequence += 1

    return []


def a_star_from_any_with_bridges(
        controller: Controller,
        starts: set[Position],
        goals: set[Position],
        traversable_fn,
        normal_step_cost: int,
        bridge_step_cost: int,
        max_expansions: int | None = None,
        bridge_landing_fn=None,
) -> tuple[list[Position], dict[Position, Position], int] | None:
    """Find an A* route from ``starts`` to ``goals`` with short bridge jumps.

    A bridge jump is a cardinal leap of two or three tiles whose intermediate
    tiles contain at least one non-traversable obstacle.  When supplied,
    ``bridge_landing_fn`` additionally requires the far endpoint to be a
    currently walkable landing cell; a Launcher cannot throw a Builder onto
    bare ground.  The returned path includes both endpoints;
    ``bridge_targets[source]`` records every jump.  The caller can use the
    same route for walking from its first endpoint and for laying conveyors
    back in the reverse direction.
    """
    if not starts or not goals:
        return None

    queue: list[tuple[int, int, int, Position]] = []
    came_from: dict[Position, tuple[Position, bool]] = {}
    g_score: dict[Position, int] = {}
    sequence = 0
    for start in sorted(starts, key=lambda pos: (pos.y, pos.x)):
        g_score[start] = 0
        heuristic = normal_step_cost * min(
            abs(start.x - goal.x) + abs(start.y - goal.y)
            for goal in goals
        )
        heappush(queue, (heuristic, 0, sequence, start))
        sequence += 1

    expansions = 0
    while queue:
        _, cost, _, current = heappop(queue)
        if cost != g_score[current]:
            continue
        if current in goals:
            path = [current]
            bridge_targets: dict[Position, Position] = {}
            while current in came_from:
                previous, is_bridge = came_from[current]
                if is_bridge:
                    bridge_targets[previous] = current
                path.append(previous)
                current = previous
            path.reverse()
            return path, bridge_targets, cost
        if max_expansions is not None and expansions >= max_expansions:
            return None
        expansions += 1

        for direction in ORTHOGONAL_DIRECTIONS:
            next_pos = current.add(direction)
            if traversable_fn(controller, next_pos):
                new_cost = cost + normal_step_cost
                if new_cost < g_score.get(next_pos, LARGE_NUMBER):
                    g_score[next_pos] = new_cost
                    came_from[next_pos] = (current, False)
                    heuristic = normal_step_cost * min(
                        abs(next_pos.x - goal.x) + abs(next_pos.y - goal.y)
                        for goal in goals
                    )
                    heappush(queue, (new_cost + heuristic, new_cost, sequence, next_pos))
                    sequence += 1

            dx, dy = direction.delta()
            for distance in (2, 3):
                bridge_target = Position(
                    current.x + dx * distance,
                    current.y + dy * distance,
                )
                if not traversable_fn(controller, bridge_target):
                    continue
                if (
                    bridge_landing_fn is not None
                    and not bridge_landing_fn(controller, bridge_target)
                ):
                    continue
                intermediates = [
                    Position(current.x + dx * step, current.y + dy * step)
                    for step in range(1, distance)
                ]
                if not any(
                    not traversable_fn(controller, pos)
                    for pos in intermediates
                ):
                    continue
                new_cost = cost + bridge_step_cost
                if new_cost >= g_score.get(bridge_target, LARGE_NUMBER):
                    continue
                g_score[bridge_target] = new_cost
                came_from[bridge_target] = (current, True)
                heuristic = normal_step_cost * min(
                    abs(bridge_target.x - goal.x) + abs(bridge_target.y - goal.y)
                    for goal in goals
                )
                heappush(
                    queue,
                    (new_cost + heuristic, new_cost, sequence, bridge_target),
                )
                sequence += 1

    return None
