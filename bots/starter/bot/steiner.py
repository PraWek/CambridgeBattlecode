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
    # Если тайл неисследован — считаем проходимым для планирования
    if env is None:
        return True
    if env == Environment.WALL:
        return False
    if _is_ore(env):
        return False
    return True


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


def _bfs_connect(
        tree_nodes: set[Position],
        targets: set[Position],
        known_env: dict,
        map_w: int,
        map_h: int,
        blocked: set | None = None,
) -> tuple[Position | None, dict[Position, Position]]:
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
            if nxt not in tree_nodes and not _is_buildable_for_conveyor(nxt, known_env, map_w, map_h, blocked):
                continue
            visited.add(nxt)
            prev[nxt] = pos
            queue.append(nxt)

    return None, {}


def compute_steiner_tree(
        core_pos: Position,
        ore_positions: list[Position],
        known_env: dict,
        map_w: int,
        map_h: int,
        radius_sq: int = STEINER_EXPLORE_RADIUS_SQ,
        blocked: set | None = None,
) -> dict[Position, Position]:
    """
    Жадный алгоритм Штейнера: последовательно подключает каждую руду
    к растущему дереву с помощью BFS. Возвращает parent[pos] = parent_pos
    - словарь, по которому строятся направления конвейеров
    """

    # Только сама клетка core уже принимает ресурсы. Соседние пустые клетки
    # должны получить конвейер, направленный в core, а не быть корнями дерева.
    tree_nodes: set[Position] = {core_pos}
    parent: dict[Position, Position] = {}

    def cheb(p: Position) -> int:
        return max(abs(p.x - core_pos.x), abs(p.y - core_pos.y))

    radius = int(radius_sq ** 0.5)
    candidate_ores = [
        ore for ore in ore_positions
        if cheb(ore) <= radius
    ]
    candidate_ores.sort(key=cheb)

    for ore in candidate_ores:
        col_pts = _collection_points(ore, known_env, map_w, map_h, blocked)
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
