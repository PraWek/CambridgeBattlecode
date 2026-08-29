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
