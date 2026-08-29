from collections import deque

from cambc import Direction, EntityType, Position


BlueprintEntry = tuple[EntityType, Direction | None, Position | None]


class NetworkMemory:
    """Persistent per-builder memory of the allied transport network.

    Battlecode units do not share Python memory, so this is intentionally a
    compact local model.  Visible allied transports extend the model, while a
    mismatch against a remembered tile becomes a repair job.  Keeping this
    state outside BuilderBot makes route planning independent of controller
    scanning and keeps adversarial repair policy in one place.
    """

    def __init__(self, busy_window: int) -> None:
        self.blueprint: dict[Position, BlueprintEntry] = {}
        self.damaged_tiles: set[Position] = set()
        self.busy_history: dict[Position, deque[bool]] = {}
        self.busy_window = busy_window

    def remember(
            self,
            pos: Position,
            entity_type: EntityType,
            direction: Direction | None = None,
            bridge_target: Position | None = None,
    ) -> None:
        self.blueprint[pos] = (entity_type, direction, bridge_target)
        self.damaged_tiles.discard(pos)

    def audit(
            self,
            pos: Position,
            actual_type: EntityType | None,
            is_friendly: bool,
            direction: Direction | None,
            bridge_target: Position | None,
    ) -> None:
        expected = self.blueprint.get(pos)
        if expected is None:
            return
        expected_type, expected_direction, expected_target = expected
        matches = (
            is_friendly
            and actual_type == expected_type
            and (
                expected_type != EntityType.CONVEYOR
                or direction == expected_direction
            )
            and (
                expected_type != EntityType.BRIDGE
                or bridge_target == expected_target
            )
        )
        if matches:
            self.damaged_tiles.discard(pos)
        else:
            self.damaged_tiles.add(pos)

    def record_busy(self, pos: Position, busy: bool) -> None:
        history = self.busy_history.setdefault(
            pos,
            deque(maxlen=self.busy_window),
        )
        history.append(busy)

    def continuously_busy(self, pos: Position) -> bool:
        history = self.busy_history.get(pos)
        return (
            history is not None
            and len(history) >= self.busy_window
            and all(history)
        )
