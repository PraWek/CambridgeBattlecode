from heapq import heappop, heappush

from cambc import Direction, Position


_SECTOR_RECEIVER_OFFSETS = {
    Direction.NORTH: ((-1, -1), (0, -1), (1, -1)),
    Direction.EAST: ((1, -1), (1, 0), (1, 1)),
    Direction.SOUTH: ((-1, 1), (0, 1), (1, 1)),
    Direction.WEST: ((-1, -1), (-1, 0), (-1, 1)),
}
_SECTOR_ENTRY_OFFSETS = {
    Direction.NORTH: ((-1, -2), (0, -2), (1, -2)),
    Direction.EAST: ((2, -1), (2, 0), (2, 1)),
    Direction.SOUTH: ((-1, 2), (0, 2), (1, 2)),
    Direction.WEST: ((-2, -1), (-2, 0), (-2, 1)),
}


def sector_receiver_offsets(direction: Direction) -> tuple[tuple[int, int], ...]:
    return _SECTOR_RECEIVER_OFFSETS[direction]


def sector_entry_offsets(direction: Direction) -> tuple[tuple[int, int], ...]:
    return _SECTOR_ENTRY_OFFSETS[direction]


def dedicated_route_tree(core_receivers: set[Position]) -> set[Position]:
    """Return the only legal anchors for a one-source-per-line route."""
    return set(core_receivers)


def starts_new_line(
        harvesters_built: int,
        has_active_line: bool,
        sources_per_line: int,
) -> bool:
    """Return whether the next source must start a fresh transport trunk."""
    return (
        not has_active_line
        or harvesters_built % sources_per_line == 0
    )


def bridge_safe_execution_path(
        forward_path: list[Position],
        deferred_bridge_sources: set[Position],
) -> tuple[list[Position], int]:
    """Append a return leg so non-terminal bridges are constructed last.

    The forward leg walks around each obstacle and builds the landing-side
    conveyors.  Reversing that already validated walk then visits bridge
    sources from downstream to upstream, so every new bridge has a live
    receiver when it is placed.
    """
    if not forward_path or not deferred_bridge_sources:
        return list(forward_path), 0
    source_indexes = [
        index
        for index, pos in enumerate(forward_path)
        if pos in deferred_bridge_sources
    ]
    if not source_indexes:
        return list(forward_path), 0
    activation_index = len(forward_path)
    first_source_index = min(source_indexes)
    return (
        list(forward_path)
        + list(reversed(forward_path[first_source_index:-1])),
        activation_index,
    )


def receivers_need_capacity_relief(
        loads_and_busy: list[tuple[int, bool]],
        capacity: int,
) -> bool:
    """Return whether every usable output is measurably beyond its capacity."""
    return bool(loads_and_busy) and all(
        load > capacity or continuously_busy
        for load, continuously_busy in loads_and_busy
    )


def residual_capacity_tree(
        network: set[Position],
        loads: dict[Position, int],
        receiver_fn,
        is_core_receiver_fn,
        capacity: int,
) -> set[Position]:
    """Return merge points whose complete downstream path has spare capacity.

    A recovered harvester is not part of the current builder's local source
    counter.  Attaching it to the nearest remembered conveyor can therefore
    overload a downstream trunk.  Validate every tile to the core instead of
    trusting only the candidate merge tile's local load.
    """
    result: set[Position] = set()
    for start in network:
        if is_core_receiver_fn(start):
            continue
        current = start
        seen: set[Position] = set()
        while current in network and current not in seen:
            seen.add(current)
            if loads.get(current, 0) >= capacity:
                break
            receiver = receiver_fn(current)
            if receiver is None:
                break
            if is_core_receiver_fn(receiver):
                result.add(start)
                break
            current = receiver
    return result


def transport_max_flow(
        network: set[Position],
        harvester_outputs: dict[Position, list[Position]],
        receiver_fn,
        is_core_receiver_fn,
        tile_capacity: int,
) -> tuple[int, dict[Position, int], set[Position]]:
    """Return sustainable harvester flow and per-transport load.

    Flow is measured in quarter-stacks: one harvester contributes one unit
    every four rounds and one transport tile carries four such units.  A
    core-reaching transport graph is functional: every tile has one output,
    so each physical intake lane forms a capacity-limited tree.  Max-flow then
    reduces to assigning harvesters among those root lanes.

    The graph contains only already built directed transport.  Construction
    candidates are still generated elsewhere, which keeps this calculation
    small enough for the 2 ms per-unit budget.
    """
    if tile_capacity <= 0:
        return 0, {pos: 0 for pos in network}, set()

    transport_tiles = {
        pos
        for pos in network
        if not is_core_receiver_fn(pos) and receiver_fn(pos) is not None
    }
    root_by_tile: dict[Position, Position] = {}
    depth_by_tile: dict[Position, int] = {}
    for start in transport_tiles:
        if start in root_by_tile:
            continue
        trail: list[Position] = []
        trail_seen: set[Position] = set()
        current = start
        root: Position | None = None
        downstream_depth = 0
        while current in transport_tiles and current not in trail_seen:
            if current in root_by_tile:
                root = root_by_tile[current]
                downstream_depth = depth_by_tile[current]
                break
            trail.append(current)
            trail_seen.add(current)
            receiver = receiver_fn(current)
            if receiver is not None and is_core_receiver_fn(receiver):
                root = current
                downstream_depth = 0
                break
            if receiver not in transport_tiles:
                break
            current = receiver
        if root is None:
            continue
        for tile in reversed(trail):
            if tile == root:
                downstream_depth = 1
            else:
                downstream_depth += 1
            root_by_tile[tile] = root
            depth_by_tile[tile] = downstream_depth

    # Keep one shortest physical output for every harvester/root pair.  The
    # capacity decision operates on root lanes, but the selected output is
    # retained so loads can be projected back onto individual conveyors.
    harvester_positions = list(harvester_outputs)
    options: list[dict[Position, Position]] = []
    for outputs in harvester_outputs.values():
        by_root: dict[Position, Position] = {}
        for output in outputs:
            if is_core_receiver_fn(output):
                by_root[output] = output
                continue
            root = root_by_tile.get(output)
            if root is None:
                continue
            previous = by_root.get(root)
            if (
                previous is None
                or depth_by_tile.get(output, 10**9)
                < depth_by_tile.get(previous, 10**9)
            ):
                by_root[root] = output
        options.append(by_root)

    root_members: dict[Position, list[int]] = {}
    assignment: dict[int, Position] = {}

    def augment(harvester: int, seen_roots: set[Position], seen_harvesters: set[int]) -> bool:
        roots = sorted(
            options[harvester],
            key=lambda root: (
                len(root_members.get(root, ())),
                depth_by_tile.get(options[harvester][root], 0),
                root.x,
                root.y,
            ),
        )
        for root in roots:
            if root in seen_roots:
                continue
            seen_roots.add(root)
            members = root_members.setdefault(root, [])
            if len(members) < tile_capacity:
                members.append(harvester)
                assignment[harvester] = root
                return True
            for displaced in tuple(members):
                if displaced in seen_harvesters:
                    continue
                seen_harvesters.add(displaced)
                if not augment(displaced, seen_roots, seen_harvesters):
                    continue
                members.remove(displaced)
                members.append(harvester)
                assignment[harvester] = root
                return True
        return False

    for harvester in range(len(options)):
        augment(harvester, set(), {harvester})

    loads = {pos: 0 for pos in network}
    for harvester, root in assignment.items():
        current = options[harvester][root]
        seen: set[Position] = set()
        while current in transport_tiles and current not in seen:
            seen.add(current)
            loads[current] += 1
            receiver = receiver_fn(current)
            if receiver is None or is_core_receiver_fn(receiver):
                break
            current = receiver
    served_harvesters = {
        harvester_positions[index]
        for index in assignment
    }
    return len(assignment), loads, served_harvesters


def minimum_cost_flow_augmentation(
        starts: list[Position],
        anchors: set[Position],
        directions: list[Direction],
        neighbor_fn,
        offset_fn,
        usable_fn,
        bridge_crosses_block_fn,
        anchor_accepts_source_fn,
        conveyor_cost: int,
        bridge_cost: int,
        max_jump_distance: int,
        max_expansions: int,
        conveyor_cost_fn=None,
        anchor_cost_fn=None,
        edge_usable_fn=None,
        minimum_conveyor_cost: int = 0,
) -> tuple[list[Position], dict[Position, Position], int] | None:
    """Find the cheapest one-unit augmentation into residual transport.

    Anchors are residual-capacity nodes or unused core inputs.  Reaching one
    supplies a cheap augmentation candidate; the caller verifies the complete
    virtual directed graph before committing construction.  Dijkstra minimises
    the actual marginal cost, including bridges and compatible transport.
    """
    if not starts or not anchors:
        return None

    custom_costs = conveyor_cost_fn is not None or anchor_cost_fn is not None
    if conveyor_cost_fn is None:
        conveyor_cost_fn = lambda _pos, _direction: conveyor_cost
    if anchor_cost_fn is None:
        anchor_cost_fn = lambda _anchor: 0
    if edge_usable_fn is None:
        edge_usable_fn = lambda _pos, _direction: True
    cost_per_tile = min(
        minimum_conveyor_cost if custom_costs else conveyor_cost,
        max(1, bridge_cost // max(1, max_jump_distance)),
    )

    def heuristic(pos: Position) -> int:
        return cost_per_tile * min(
            abs(pos.x - anchor.x) + abs(pos.y - anchor.y)
            for anchor in anchors
        )

    queue: list[tuple[int, int, int, int, int, Position]] = []
    costs: dict[Position, int] = {}
    came_from: dict[Position, tuple[Position | None, bool]] = {}
    for start in starts:
        if start in anchors or not usable_fn(start):
            continue
        costs[start] = 0
        came_from[start] = (None, False)
        remaining = heuristic(start)
        heappush(queue, (remaining, remaining, 0, start.x, start.y, start))

    expansions = 0
    while queue:
        _, _, cost, _, _, current = heappop(queue)
        if cost != costs.get(current):
            continue
        if current in anchors:
            nodes = [current]
            bridge_targets: dict[Position, Position] = {}
            while True:
                previous, is_bridge = came_from[current]
                if previous is None:
                    break
                if is_bridge:
                    bridge_targets[previous] = current
                nodes.append(previous)
                current = previous
            nodes.reverse()
            return nodes, bridge_targets, cost
        if expansions >= max_expansions:
            break
        expansions += 1

        for direction in directions:
            if not edge_usable_fn(current, direction):
                continue
            next_pos = neighbor_fn(current, direction)
            if (
                next_pos is None
                or not usable_fn(next_pos)
                or (
                    next_pos in anchors
                    and not anchor_accepts_source_fn(next_pos, current)
                )
            ):
                continue
            new_cost = cost + conveyor_cost_fn(current, direction)
            if next_pos in anchors:
                new_cost += anchor_cost_fn(next_pos)
            if new_cost >= costs.get(next_pos, 10**9):
                continue
            costs[next_pos] = new_cost
            came_from[next_pos] = (current, False)
            remaining = heuristic(next_pos)
            heappush(
                queue,
                (
                    new_cost + remaining, remaining, new_cost,
                    next_pos.x, next_pos.y, next_pos,
                ),
            )

        for direction in directions:
            dx, dy = direction.delta()
            for distance in range(2, max_jump_distance + 1):
                target = offset_fn(current, dx * distance, dy * distance)
                if (
                    target is None
                    or not usable_fn(target)
                    or not bridge_crosses_block_fn(current, direction, distance)
                    or (
                        target in anchors
                        and not anchor_accepts_source_fn(target, current)
                    )
                ):
                    continue
                new_cost = cost + bridge_cost
                if target in anchors:
                    new_cost += anchor_cost_fn(target)
                if new_cost >= costs.get(target, 10**9):
                    continue
                costs[target] = new_cost
                came_from[target] = (current, True)
                remaining = heuristic(target)
                heappush(
                    queue,
                    (
                        new_cost + remaining, remaining, new_cost,
                        target.x, target.y, target,
                    ),
                )
    return None


def minimum_cost_bridge_route(
        starts: list[Position],
        anchors: set[Position],
        directions: list[Direction],
        neighbor_fn,
        offset_fn,
        usable_fn,
        bridge_crosses_block_fn,
        anchor_accepts_source_fn,
        conveyor_cost: int,
        bridge_cost: int,
        max_jump_distance: int,
        max_expansions: int,
) -> tuple[list[Position], dict[Position, Position], int] | None:
    """Compatibility wrapper for callers which use uniform construction cost."""
    return minimum_cost_flow_augmentation(
        starts,
        anchors,
        directions,
        neighbor_fn,
        offset_fn,
        usable_fn,
        bridge_crosses_block_fn,
        anchor_accepts_source_fn,
        conveyor_cost,
        bridge_cost,
        max_jump_distance,
        max_expansions,
    )
