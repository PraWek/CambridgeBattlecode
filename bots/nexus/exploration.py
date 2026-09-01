from collections.abc import Callable, Iterable

from cambc import Direction, Position


def _turn_distance(direction: Direction, heading: Direction | None) -> int:
    """Return the smaller number of 45-degree turns from ``heading``."""
    if heading is None:
        return 0
    probe = heading
    clockwise = 0
    while probe != direction and clockwise < 8:
        probe = probe.rotate_right()
        clockwise += 1
    return min(clockwise, 8 - clockwise)


def target_approach_progress(
        origin: Position,
        target: Position,
        candidate: Position,
) -> int:
    """Measure progress beyond the origin toward a strategic map target."""
    origin_distance = max(abs(origin.x - target.x), abs(origin.y - target.y))
    candidate_distance = max(
        abs(candidate.x - target.x),
        abs(candidate.y - target.y),
    )
    return origin_distance - candidate_distance


def choose_information_gain_step(
        current: Position,
        directions: Iterable[Direction],
        neighbor: Callable[[Position, Direction], Position | None],
        viable: Callable[[Position], bool],
        vision_gain: Callable[[Position], int],
        total_visits: dict[Position, int],
        recent_visits: dict[Position, int],
        avoided: Callable[[Position], bool],
        forward_progress: Callable[[Position], int],
        sweep_bias: Callable[[Position, Position], int],
        heading: Direction | None,
        require_new_vision: bool,
) -> tuple[Direction, Position] | None:
    """Choose a stable local scout step without a global path search.

    Information gain comes before visit counts.  Heading continuity breaks
    ties, so an open-field scout draws long sector sweeps instead of changing
    direction whenever two equivalent cells swap dictionary order.
    """
    ranked: list[tuple[tuple[int, ...], Direction, Position]] = []
    for order, direction in enumerate(directions):
        candidate = neighbor(current, direction)
        if candidate is None or not viable(candidate):
            continue
        gain = vision_gain(candidate)
        if require_new_vision and gain == 0:
            continue
        current_progress = forward_progress(current)
        candidate_progress = forward_progress(candidate)
        rank = (
            int(avoided(candidate)),
            int(candidate_progress < 0),
            int(candidate_progress < current_progress),
            -gain,
            total_visits.get(candidate, 0),
            recent_visits.get(candidate, 0),
            _turn_distance(direction, heading),
            -candidate_progress,
            -sweep_bias(current, candidate),
            order,
        )
        ranked.append((rank, direction, candidate))
    if not ranked:
        return None
    _, direction, candidate = min(ranked, key=lambda item: item[0])
    return direction, candidate


def is_static_step_obstacle(
        terrain_blocked: bool,
        building_present: bool,
        building_passable: bool,
) -> bool:
    """Distinguish permanent geometry from a temporarily unavailable step."""
    return terrain_blocked or (building_present and not building_passable)


def should_recycle_stalled_builder(
        stuck_rounds: int,
        rounds_without_progress: int,
        threshold: int,
) -> bool:
    """Recycle only after both position and useful actions stopped changing.

    A move command is not proof of movement: simultaneous builders can both
    pass ``can_move`` and still collide.  Construction and repairs count as
    progress independently, so a stationary worker is kept while it is doing
    useful work.
    """
    return stuck_rounds >= threshold and rounds_without_progress >= threshold


def should_recycle_exhausted_scout(
        rounds_without_discovery: int,
        stuck_rounds: int,
        route_failures: int,
        cycle_count: int,
        discovery_threshold: int,
        stuck_threshold: int,
        route_failure_threshold: int,
        cycle_threshold: int,
        has_pending_repairs: bool,
) -> bool:
    """Recycle a scout that only revisits known terrain for a long time.

    Remembering an allied conveyor is not useful work by itself.  The old
    guard kept every scout alive after it had seen its first transport, which
    let two-tile patrol loops run until turn 2000.  A real outstanding repair
    still keeps the builder alive.
    """
    if has_pending_repairs or rounds_without_discovery < discovery_threshold:
        return False
    return (
        stuck_rounds >= stuck_threshold
        or route_failures >= route_failure_threshold
        or cycle_count >= cycle_threshold
    )
