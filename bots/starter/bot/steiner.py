"""
Greedy Minimum Steiner Tree approximation for conveyor routing.

Problem: connect N titanium ore deposits to the core using a minimum set
of conveyor tiles (rooted undirected Steiner tree, core = root).

Algorithm (greedy nearest-terminal):
  1. Initialize tree with all core footprint tiles (3×3 around core centre).
  2. For each ore (sorted by Chebyshev distance to core, nearest first):
       a. Compute the ore's "collection points" = orthogonal neighbours of ore
          that are buildable (not wall, not ore).
       b. If any collection point is already in the tree → ore is connected,
          continue to next ore.
       c. Run a multi-source BFS from every current tree node simultaneously.
       d. The first collection point reached is the connection target.
       e. Trace the BFS path back to the tree and add every new tile, recording
          parent[tile] = next_tile_toward_core.
  3. Return parent dict.

Usage:
  parent = compute_steiner_tree(core_pos, ore_list, known_env, map_w, map_h)

  For each tile P in parent:
      conveyor_direction = P.direction_to(parent[P])   # points toward root
      ct.build_conveyor(P, conveyor_direction)

Conveyor semantics recap:
  A conveyor at P pointing direction D outputs resources to P + D.
  So pointing toward parent (= toward core) carries resources core-ward. ✓
"""

from __future__ import annotations

from collections import deque
from cambc import Environment, Position

from bot.constants import ORTHOGONAL_DIRECTIONS

STEINER_EXPLORE_RADIUS_SQ = 100  # Chebyshev² ≤ 100  →  radius ≤ 10


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
    """
    Can we place a conveyor here?
    Requirements: in-bounds, observed, not a wall, not an ore deposit,
    not in the permanently-blocked set.
    Unknown tiles are skipped (conservative – avoids planning through walls we
    haven't seen yet).
    """
    if not _in_bounds(pos, map_w, map_h):
        return False
    if blocked and pos in blocked:
        return False
    env = known_env.get(pos)
    if env is None:
        return False  # unobserved – skip
    if env == Environment.WALL:
        return False
    if _is_ore(env):
        return False
    return True


def _core_tiles(core_pos: Position, map_w: int, map_h: int) -> list[Position]:
    """All cells of the 3×3 core footprint that lie in-bounds."""
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
    """
    Orthogonal neighbours of an ore tile where a conveyor can sit and collect
    the harvester's output.
    """
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
    """
    Multi-source BFS from tree_nodes to reach any position in targets.

    Returns (found_target, prev_dict) where prev_dict maps each visited node
    to the node it was reached from (so we can trace the path back).
    Nodes already in tree_nodes are seeds with no prev entry.
    Returns (None, {}) if no target is reachable.
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
            # Traversal rule: we can pass through tiles already in the tree
            # (they are passable by definition) or through buildable empty tiles.
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
    Compute a greedy Steiner tree connecting titanium ore deposits to the core.

    Args:
        core_pos:      Centre tile of the allied core (3×3 building).
        ore_positions: List of titanium ore tile positions to connect.
        known_env:     Dict mapping Position → Environment (from observe_tiles).
        map_w, map_h:  Map dimensions.
        radius_sq:     Max squared Chebyshev distance from core to include an ore.
                       Default = 100  (radius ≤ 10).
        blocked:       Set of positions that are permanently impassable (e.g. walls
                       discovered only when a build attempt failed).  These are
                       excluded from BFS traversal.

    Returns:
        parent: dict mapping each conveyor tile → its parent tile (one step
                closer to core). The conveyor at tile P must point in direction
                P.direction_to(parent[P]) to deliver resources toward the core.
                Core footprint tiles are NOT in the dict (they are the root).
    """
    core_tile_set: set[Position] = set(_core_tiles(core_pos, map_w, map_h))
    tree_nodes: set[Position] = set(core_tile_set)
    parent: dict[Position, Position] = {}

    # Filter ores to the explore radius and sort nearest-first (Chebyshev)
    def cheb(p: Position) -> int:
        return max(abs(p.x - core_pos.x), abs(p.y - core_pos.y))

    candidate_ores = [
        ore for ore in ore_positions
        if cheb(ore) ** 2 <= radius_sq
    ]
    candidate_ores.sort(key=cheb)

    for ore in candidate_ores:
        col_pts = _collection_points(ore, known_env, map_w, map_h, blocked)
        if not col_pts:
            continue  # ore completely surrounded by walls – unreachable

        col_set = set(col_pts)

        # If any collection point is already a tree node, ore is connected
        if col_set & tree_nodes:
            continue

        # BFS: multi-source from current tree to reach a collection point
        found, prev = _bfs_connect(tree_nodes, col_set, known_env, map_w, map_h, blocked)
        if found is None:
            continue  # this ore is unreachable from current tree given known map

        # Trace path: found → ... → some tree node, adding each new tile
        cur = found
        while cur not in tree_nodes:
            p = prev[cur]
            parent[cur] = p
            tree_nodes.add(cur)
            cur = p

    return parent
