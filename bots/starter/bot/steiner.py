from __future__ import annotations

from collections import deque
from cambc import Environment, Position

from bot.constants import ORTHOGONAL_DIRECTIONS

STEINER_EXPLORE_RADIUS_SQ = 100


def _in_bounds(pos: Position, map_w: int, map_h: int) -> bool:
    return 0 <= pos.x < map_w and 0 <= pos.y < map_h


def _is_ore(env) -> bool:
    return env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE)


def _is_buildable_for_conveyor(
    pos: Position,
    known_env: dict,
    map_w: int,
    map_h: int,
    blocked: set | None = None,
) -> bool:
    if not _in_bounds(pos, map_w, map_h):
        return False
    if blocked and pos in blocked:
        return False
    env = known_env.get(pos)
    if env is None:
        return True
    if env == Environment.WALL:
        return False
    if _is_ore(env):
        return False
    return True


def _core_tiles(core_pos: Position, map_w: int, map_h: int) -> list[Position]:
    tiles = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            p = Position(core_pos.x + dx, core_pos.y + dy)
            if _in_bounds(p, map_w, map_h):
                tiles.append(p)
    return tiles


def _collection_points(
    ore: Position,
    known_env: dict,
    map_w: int,
    map_h: int,
    blocked: set | None = None,
) -> list[Position]:
    pts = []
    for d in ORTHOGONAL_DIRECTIONS:
        adj = ore.add(d)
        if _is_buildable_for_conveyor(adj, known_env, map_w, map_h, blocked):
            pts.append(adj)
    return pts


def _bfs_shortest_paths(
    sources: set[Position],
    known_env: dict,
    map_w: int,
    map_h: int,
    blocked: set | None = None,
) -> dict[Position, Position]:
    """
    Multi-source BFS from `sources`.
    Returns prev[pos] = parent (the neighbour closer to a source).
    Used for finding shortest paths from any source to any target.
    """
    prev: dict[Position, Position] = {}
    visited: set[Position] = set(sources)
    queue: deque[Position] = deque(sources)

    while queue:
        pos = queue.popleft()
        for d in ORTHOGONAL_DIRECTIONS:
            nxt = pos.add(d)
            if nxt in visited:
                continue
            if not _is_buildable_for_conveyor(nxt, known_env, map_w, map_h, blocked):
                continue
            visited.add(nxt)
            prev[nxt] = pos
            queue.append(nxt)

    return prev


def _bfs_connect(
    tree_nodes: set[Position],
    targets: set[Position],
    known_env: dict,
    map_w: int,
    map_h: int,
    blocked: set | None = None,
) -> tuple[Position | None, dict[Position, Position]]:
    """
    BFS from tree_nodes outward; returns the first target found and prev-dict.
    Nodes already in tree_nodes can be traversed freely (they are existing tree nodes).
    """
    prev: dict[Position, Position] = {}
    visited: set[Position] = set()
    queue: deque[Position] = deque()

    for node in tree_nodes:
        visited.add(node)
        queue.append(node)

    while queue:
        pos = queue.popleft()

        if pos in targets and pos not in tree_nodes:
            return pos, prev

        for d in ORTHOGONAL_DIRECTIONS:
            nxt = pos.add(d)
            if nxt in visited:
                continue
            # Allow traversal through existing tree nodes or buildable tiles
            if nxt not in tree_nodes and not _is_buildable_for_conveyor(nxt, known_env, map_w, map_h, blocked):
                continue
            visited.add(nxt)
            prev[nxt] = pos
            queue.append(nxt)

    return None, {}


def _ore_priority(
    ore: Position,
    core_pos: Position,
    tree_nodes: set[Position],
    known_env: dict,
    map_w: int,
    map_h: int,
    blocked: set | None,
) -> int:
    """
    Priority = shortest BFS distance from the existing tree to the ore's collection points.
    Lower is better (greedy: connect cheapest ore first to share path segments).
    """
    col_pts = _collection_points(ore, known_env, map_w, map_h, blocked)
    if not col_pts:
        return 10**9

    col_set = set(col_pts)
    # BFS from tree to find distance to collection points
    visited: set[Position] = set(tree_nodes)
    queue: deque[tuple[Position, int]] = deque((n, 0) for n in tree_nodes)

    while queue:
        pos, dist = queue.popleft()
        if pos in col_set:
            return dist
        for d in ORTHOGONAL_DIRECTIONS:
            nxt = pos.add(d)
            if nxt in visited:
                continue
            if nxt not in tree_nodes and not _is_buildable_for_conveyor(nxt, known_env, map_w, map_h, blocked):
                continue
            visited.add(nxt)
            queue.append((nxt, dist + 1))

    # Unreachable — fallback: Chebyshev to core
    return max(abs(ore.x - core_pos.x), abs(ore.y - core_pos.y)) * 100


def compute_steiner_tree(
    core_pos: Position,
    ore_positions: list[Position],
    known_env: dict,
    map_w: int,
    map_h: int,
    radius_sq: int = STEINER_EXPLORE_RADIUS_SQ,
    blocked: set | None = None,
) -> dict[Position, Position]:

    def cheb(p: Position) -> int:
        return max(abs(p.x - core_pos.x), abs(p.y - core_pos.y))

    core_tile_set: set[Position] = set(_core_tiles(core_pos, map_w, map_h))
    tree_nodes: set[Position] = set(core_tile_set)
    parent: dict[Position, Position] = {}

    candidate_ores = [
        ore for ore in ore_positions
        if cheb(ore) ** 2 <= radius_sq
    ]

    remaining = list(candidate_ores)

    while remaining:
        best_ore = None
        best_dist = 10**9
        for ore in remaining:
            col_pts = _collection_points(ore, known_env, map_w, map_h, blocked)
            if not col_pts:
                continue
            col_set = set(col_pts)
            if col_set & tree_nodes:
                best_ore = ore
                best_dist = -1
                break
            d = _ore_priority(ore, core_pos, tree_nodes, known_env, map_w, map_h, blocked)
            if d < best_dist:
                best_dist = d
                best_ore = ore

        if best_ore is None:
            break

        remaining.remove(best_ore)
        col_pts = _collection_points(best_ore, known_env, map_w, map_h, blocked)
        if not col_pts:
            continue
        col_set = set(col_pts)

        if col_set & tree_nodes:
            continue

        found, prev = _bfs_connect(tree_nodes, col_set, known_env, map_w, map_h, blocked)
        if found is None:
            continue

        cur = found
        while cur not in tree_nodes:
            p = prev[cur]
            parent[cur] = p
            tree_nodes.add(cur)
            cur = p

    return parent
