from heapq import heappop, heappush

from cambc import Direction, Position


def incremental_steiner_branch(
        starts: list[Position],
        tree: set[Position],
        directions: list[Direction],
        can_use_tile,
        can_use_edge,
        receiver_accepts,
        tile_cost,
        max_expansions: int,
) -> tuple[Position, list[Position], dict[Position, Direction], Position] | None:
    """Return the cheapest new rectilinear branch from a terminal to a tree.

    Adding every new terminal by its shortest admissible path to the existing
    tree is the online/incremental Steiner-tree approximation.  Unlike choosing
    a few geometrically close anchors, this search considers the whole current
    tree and charges only for the new tiles that the branch actually adds.
    """
    if not starts or not tree:
        return None

    queue: list[tuple[int, int, int, int, Position]] = []
    costs: dict[Position, int] = {}
    came_from: dict[Position, Position | None] = {}
    for start in starts:
        if start in tree or not can_use_tile(start):
            continue
        if 0 >= costs.get(start, 10**9):
            continue
        costs[start] = 0
        came_from[start] = None
        heappush(queue, (0, 0, start.x, start.y, start))

    best_goal: tuple[int, Position, Position, Direction] | None = None
    expansions = 0
    while queue:
        cost, steps, _, _, current = heappop(queue)
        if cost != costs.get(current):
            continue
        if best_goal is not None and cost >= best_goal[0]:
            break
        if expansions >= max_expansions:
            break
        expansions += 1

        for direction in directions:
            if not can_use_edge(current, direction):
                continue
            next_pos = current.add(direction)
            edge_cost = tile_cost(current, direction)
            new_cost = cost + edge_cost
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
            heappush(queue, (new_cost, steps + 1, next_pos.x, next_pos.y, next_pos))

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
