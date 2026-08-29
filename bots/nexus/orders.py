from collections.abc import Iterable, Mapping

from cambc import Direction, Position


def spawn_order_for(
        records: Iterable[tuple[int, Position, int]],
        current: Position,
        spawn_kinds: set[int],
        directions: Mapping[int, Direction],
) -> tuple[int, Position, Direction] | None:
    """Return only the handoff order addressed to this newborn's tile."""
    for kind, encoded_spawn, payload in records:
        if kind not in spawn_kinds or encoded_spawn != current:
            continue
        direction = directions.get(payload)
        if direction is not None:
            return kind, encoded_spawn, direction
    return None
