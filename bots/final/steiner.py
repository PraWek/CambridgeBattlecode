from heapq import heappop, heappush

from cambc import Direction, Position


def incremental_steiner_branch(
        starts: list[Position],
        tree: set[Position],
        directions: list[Direction],
        neighbor_fn,
        can_use_tile,
        can_use_edge,
        receiver_accepts,
        tile_cost,
        max_expansions: int,
        heuristic_fn=None,
) -> tuple[Position, list[Position], dict[Position, Direction], Position] | None:
    """Return the cheapest admissible rectilinear branch into ``tree``.

    Repeating this operation for newly found deposits is an online Steiner
    approximation.  Capacity policy belongs to the caller: ``tree`` contains
    only receivers whose downstream lane still has residual throughput.
    """
    if not starts or not tree:
        return None

    queue: list[tuple[int, int, int, int, int, Position]] = []
    costs: dict[Position, int] = {}
    came_from: dict[Position, Position | None] = {}
    for start in starts:
        if start in tree or not can_use_tile(start):
            continue
        costs[start] = 0
        came_from[start] = None
        heuristic = 0 if heuristic_fn is None else heuristic_fn(start)
        heappush(queue, (heuristic, 0, 0, start.x, start.y, start))

    best_goal: tuple[int, Position, Position, Direction] | None = None
    expansions = 0
    while queue:
        priority, cost, steps, _, _, current = heappop(queue)
        if cost != costs.get(current):
            continue
        if best_goal is not None and priority >= best_goal[0]:
            break
        if expansions >= max_expansions:
            break
        expansions += 1

        for direction in directions:
            if not can_use_edge(current, direction):
                continue
            next_pos = neighbor_fn(current, direction)
            if next_pos is None:
                continue
            new_cost = cost + tile_cost(current, direction)
            if next_pos in tree:
                if not receiver_accepts(next_pos, current):
                    continue
                if best_goal is None or new_cost < best_goal[0]:
                    best_goal = (new_cost, current, next_pos, direction)
                continue
            if not can_use_tile(next_pos) or new_cost >= costs.get(next_pos, 10**9):
                continue
            costs[next_pos] = new_cost
            came_from[next_pos] = current
            heuristic = 0 if heuristic_fn is None else heuristic_fn(next_pos)
            heappush(
                queue,
                (
                    new_cost + heuristic,
                    new_cost,
                    steps + 1,
                    next_pos.x,
                    next_pos.y,
                    next_pos,
                ),
            )

    if best_goal is None:
        return None

    _, final_tile, anchor, final_direction = best_goal
    build_tiles = [final_tile]
    while came_from[build_tiles[-1]] is not None:
        build_tiles.append(came_from[build_tiles[-1]])
    build_tiles.reverse()
    approach = build_tiles[0]
    full_path = build_tiles + [anchor]
    conveyor_directions = {
        tile: tile.direction_to(full_path[index + 1])
        for index, tile in enumerate(build_tiles)
    }
    conveyor_directions[final_tile] = final_direction
    return approach, build_tiles, conveyor_directions, anchor
