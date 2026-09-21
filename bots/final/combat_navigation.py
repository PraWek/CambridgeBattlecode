from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any

from cambc import Controller, Direction, Position

from constants import LARGE_NUMBER, ORTHOGONAL_DIRECTIONS
from geometry import chebyshev


@dataclass
class AStarSearchState:
    """Frontier retained when a bounded A* search needs another turn.

    A state belongs to one logical query at a time.  The navigation functions
    reset it if their start or target set changes, and leave it populated only
    after reaching ``max_expansions``.  Callers can therefore distinguish a
    pending search from a proven missing route through :attr:`pending`.
    """

    query: tuple[Any, ...] | None = None
    queue: list[Any] = field(default_factory=list)
    came_from: dict[Position, Any] = field(default_factory=dict)
    g_score: dict[Position, int] = field(default_factory=dict)
    sequence: int = 0
    pending: bool = False

    def begin(self, query: tuple[Any, ...]) -> bool:
        """Prepare ``query`` and return whether it needs initialisation."""
        if self.query == query:
            return False
        self.query = query
        self.queue.clear()
        self.came_from.clear()
        self.g_score.clear()
        self.sequence = 0
        self.pending = False
        return True

    def finish(self) -> None:
        """Discard a completed search so a future query starts fresh."""
        self.query = None
        self.queue.clear()
        self.came_from.clear()
        self.g_score.clear()
        self.sequence = 0
        self.pending = False


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
        state: AStarSearchState | None = None,
) -> list[Position]:
    """Find a lowest-cost A* path from ``start`` to any traversable goal tile.

    ``max_expansions`` limits work performed in this call.  When ``state`` is
    supplied, a limited search retains its frontier and resumes on the next
    call; ``state.pending`` distinguishes that case from a missing route.
    """
    search_state = state or AStarSearchState()
    if not goals or start in goals:
        search_state.finish()
        return []

    if preferred_tiles is None:
        preferred_tiles = set()
    query = (
        "to_any",
        start,
        frozenset(goals),
        frozenset(preferred_tiles),
        tuple(movement_directions),
    )
    if search_state.begin(query):
        search_state.queue.append((0, 0, start))
        search_state.came_from[start] = start
        search_state.g_score[start] = 0
    minimum_step_cost = 1 if preferred_tiles else 4
    expansions = 0

    while search_state.queue:
        _, cost, current = search_state.queue[0]
        if cost != search_state.g_score[current]:
            heappop(search_state.queue)
            continue
        if current in goals:
            heappop(search_state.queue)
            path = []
            while current != start:
                path.append(current)
                current = search_state.came_from[current]
            path.reverse()
            search_state.finish()
            return path
        if max_expansions is not None and expansions >= max_expansions:
            search_state.pending = True
            return []
        heappop(search_state.queue)
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
            if new_cost >= search_state.g_score.get(next_pos, LARGE_NUMBER):
                continue
            search_state.g_score[next_pos] = new_cost
            search_state.came_from[next_pos] = current
            heuristic = minimum_step_cost * min(
                chebyshev(next_pos, goal) for goal in goals
            )
            heappush(
                search_state.queue,
                (new_cost + heuristic, new_cost, next_pos),
            )

    search_state.finish()
    return []


def a_star_from_any(
        controller: Controller,
        starts: set[Position],
        goals: set[Position],
        traversable_fn,
        neighbor_fn,
        movement_directions=ORTHOGONAL_DIRECTIONS,
        max_expansions: int | None = None,
        state: AStarSearchState | None = None,
) -> list[Position]:
    """Find a shortest A* path from any start tile to any goal tile.

    The returned route includes its selected start tile, unlike
    :func:`a_star_to_any`, because callers such as conveyor planners must
    construct infrastructure on that first tile.  All steps have equal cost;
    the multi-source frontier lets nearby starts compete in a single bounded
    search instead of running one search for each possible source.
    """
    search_state = state or AStarSearchState()
    if not starts or not goals:
        search_state.finish()
        return []

    query = (
        "from_any",
        frozenset(starts),
        frozenset(goals),
        tuple(movement_directions),
    )
    if search_state.begin(query):
        for start in sorted(starts, key=lambda pos: (pos.y, pos.x)):
            search_state.came_from[start] = None
            search_state.g_score[start] = 0
            heuristic = min(chebyshev(start, goal) for goal in goals)
            heappush(
                search_state.queue,
                (heuristic, 0, search_state.sequence, start),
            )
            search_state.sequence += 1

    expansions = 0
    while search_state.queue:
        _, cost, _, current = search_state.queue[0]
        if cost != search_state.g_score[current]:
            heappop(search_state.queue)
            continue
        if current in goals:
            heappop(search_state.queue)
            path: list[Position] = []
            while current is not None:
                path.append(current)
                current = search_state.came_from[current]
            path.reverse()
            search_state.finish()
            return path
        if max_expansions is not None and expansions >= max_expansions:
            search_state.pending = True
            return []
        heappop(search_state.queue)
        expansions += 1

        for direction in movement_directions:
            next_pos = neighbor_fn(current, direction)
            if next_pos is None:
                continue
            if not traversable_fn(controller, next_pos):
                continue
            new_cost = cost + 1
            if new_cost >= search_state.g_score.get(next_pos, LARGE_NUMBER):
                continue
            search_state.g_score[next_pos] = new_cost
            search_state.came_from[next_pos] = current
            heuristic = min(chebyshev(next_pos, goal) for goal in goals)
            heappush(
                search_state.queue,
                (
                    new_cost + heuristic,
                    new_cost,
                    search_state.sequence,
                    next_pos,
                ),
            )
            search_state.sequence += 1

    search_state.finish()
    return []


def a_star_from_any_with_bridges(
        controller: Controller,
        starts: set[Position],
        goals: set[Position],
        traversable_fn,
        normal_step_cost: int,
        bridge_step_cost: int,
        neighbor_fn,
        max_expansions: int | None = None,
        bridge_landing_fn=None,
        existing_bridge_crossings: dict[Position, Position] | None = None,
        state: AStarSearchState | None = None,
) -> tuple[list[Position], dict[Position, Position], int] | None:
    """Find an A* route from ``starts`` to ``goals`` with short bridge jumps.

    A bridge jump is a cardinal leap of two or three tiles whose intermediate
    tiles contain at least one non-traversable obstacle.  When supplied,
    ``bridge_landing_fn`` additionally requires the far endpoint to be a
    currently walkable landing cell; a Launcher cannot throw a Builder onto
    bare ground.  The returned path includes both endpoints;
    ``bridge_targets[source]`` records every jump.  The caller can use the
    same route for walking from its first endpoint and for laying conveyors.
    ``existing_bridge_crossings`` maps a pre-existing Bridge's source endpoint
    to its landing endpoint; those crossings are reused at zero construction
    cost and are not returned as new ``bridge_targets``.
    """
    search_state = state or AStarSearchState()
    if not starts or not goals:
        search_state.finish()
        return None

    crossings = (
        ()
        if existing_bridge_crossings is None
        else tuple(sorted(
            existing_bridge_crossings.items(),
            key=lambda item: (
                item[0].y,
                item[0].x,
                item[1].y,
                item[1].x,
            ),
        ))
    )
    query = (
        "from_any_with_bridges",
        frozenset(starts),
        frozenset(goals),
        normal_step_cost,
        bridge_step_cost,
        crossings,
    )
    if search_state.begin(query):
        for start in sorted(starts, key=lambda pos: (pos.y, pos.x)):
            search_state.g_score[start] = 0
            heuristic = normal_step_cost * min(
                abs(start.x - goal.x) + abs(start.y - goal.y)
                for goal in goals
            )
            heappush(
                search_state.queue,
                (heuristic, 0, search_state.sequence, start),
            )
            search_state.sequence += 1

    expansions = 0
    while search_state.queue:
        _, cost, _, current = search_state.queue[0]
        if cost != search_state.g_score[current]:
            heappop(search_state.queue)
            continue
        if current in goals:
            heappop(search_state.queue)
            path = [current]
            bridge_targets: dict[Position, Position] = {}
            while current in search_state.came_from:
                previous, is_bridge = search_state.came_from[current]
                if is_bridge:
                    bridge_targets[previous] = current
                path.append(previous)
                current = previous
            path.reverse()
            search_state.finish()
            return path, bridge_targets, cost
        if max_expansions is not None and expansions >= max_expansions:
            search_state.pending = True
            return None
        heappop(search_state.queue)
        expansions += 1

        for direction in ORTHOGONAL_DIRECTIONS:
            next_pos = neighbor_fn(current, direction)
            if next_pos is None:
                continue
            if traversable_fn(controller, next_pos):
                new_cost = cost + normal_step_cost
                if new_cost < search_state.g_score.get(next_pos, LARGE_NUMBER):
                    search_state.g_score[next_pos] = new_cost
                    search_state.came_from[next_pos] = (current, False)
                    heuristic = normal_step_cost * min(
                        abs(next_pos.x - goal.x) + abs(next_pos.y - goal.y)
                        for goal in goals
                    )
                    heappush(
                        search_state.queue,
                        (
                            new_cost + heuristic,
                            new_cost,
                            search_state.sequence,
                            next_pos,
                        ),
                    )
                    search_state.sequence += 1

            for distance in (2, 3):
                bridge_positions: list[Position] = []
                bridge_target = current
                for _ in range(distance):
                    bridge_target = neighbor_fn(bridge_target, direction)
                    if bridge_target is None:
                        break
                    bridge_positions.append(bridge_target)
                if bridge_target is None:
                    continue
                if not traversable_fn(controller, bridge_target):
                    continue
                if (
                    bridge_landing_fn is not None
                    and not bridge_landing_fn(controller, bridge_target)
                ):
                    continue
                if not any(
                    not traversable_fn(controller, pos)
                    for pos in bridge_positions[:-1]
                ):
                    continue
                new_cost = cost + bridge_step_cost
                if new_cost >= search_state.g_score.get(
                    bridge_target,
                    LARGE_NUMBER,
                ):
                    continue
                search_state.g_score[bridge_target] = new_cost
                search_state.came_from[bridge_target] = (current, True)
                heuristic = normal_step_cost * min(
                    abs(bridge_target.x - goal.x) + abs(bridge_target.y - goal.y)
                    for goal in goals
                )
                heappush(
                    search_state.queue,
                    (
                        new_cost + heuristic,
                        new_cost,
                        search_state.sequence,
                        bridge_target,
                    ),
                )
                search_state.sequence += 1

        existing_landing = (
            None
            if existing_bridge_crossings is None
            else existing_bridge_crossings.get(current)
        )
        if (
            existing_landing is not None
            and traversable_fn(controller, existing_landing)
            and cost < search_state.g_score.get(existing_landing, LARGE_NUMBER)
        ):
            search_state.g_score[existing_landing] = cost
            search_state.came_from[existing_landing] = (current, False)
            heuristic = normal_step_cost * min(
                abs(existing_landing.x - goal.x) + abs(existing_landing.y - goal.y)
                for goal in goals
            )
            heappush(
                search_state.queue,
                (cost + heuristic, cost, search_state.sequence, existing_landing),
            )
            search_state.sequence += 1

    search_state.finish()
    return None
