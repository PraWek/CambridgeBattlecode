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
        # Only the builder which laid a partial branch may tear it down.  Keep
        # that ownership state beside the blueprint instead of letting route
        # replans silently lose it inside BuilderBot.
        self.unfinished_owned_tiles: set[Position] = set()
        self.abandoned_owned_tiles: set[Position] = set()
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

    def forget(self, pos: Position) -> None:
        """Drop an obsolete route entry before replacing its direction.

        Without this transition the repair pass can rebuild the old direction
        one turn after the connection planner deliberately destroyed it.  The
        two passes then alternate forever on the same tile.
        """
        self.blueprint.pop(pos, None)
        self.damaged_tiles.discard(pos)
        self.busy_history.pop(pos, None)
        self.unfinished_owned_tiles.discard(pos)
        self.abandoned_owned_tiles.discard(pos)

    def record_unfinished_owned_tile(self, pos: Position) -> None:
        """Record a transport laid for the builder's current branch."""
        self.unfinished_owned_tiles.add(pos)
        self.abandoned_owned_tiles.discard(pos)

    def replan_owned_branch(
            self,
            planned_tiles: set[Position],
            connected_tiles: set[Position],
    ) -> None:
        """Retain reused partial tiles and retire those omitted by a replan.

        A tile which has meanwhile gained a route to the core is useful even
        when the new local plan does not mention it.  It must never enter the
        demolition queue.
        """
        dropped = self.unfinished_owned_tiles - planned_tiles - connected_tiles
        reclaimed = self.abandoned_owned_tiles & planned_tiles
        self.abandoned_owned_tiles.update(dropped)
        self.unfinished_owned_tiles.intersection_update(planned_tiles)
        self.unfinished_owned_tiles.update(reclaimed)
        self.abandoned_owned_tiles.difference_update(planned_tiles)
        self.abandoned_owned_tiles.difference_update(connected_tiles)

    def abandon_owned_branch(self, connected_tiles: set[Position]) -> None:
        """Queue the still-disconnected portion of a cancelled branch."""
        self.abandoned_owned_tiles.update(
            self.unfinished_owned_tiles - connected_tiles
        )
        self.unfinished_owned_tiles.clear()
        self.abandoned_owned_tiles.difference_update(connected_tiles)

    def complete_owned_branch(self) -> None:
        """Release ownership after the current branch reaches the core."""
        self.unfinished_owned_tiles.clear()

    def actionable_abandoned_tiles(
            self,
            connected_tiles: set[Position],
            active_plan_tiles: set[Position],
    ) -> set[Position]:
        """Return obsolete owned transports which are still safe to remove."""
        rescued = connected_tiles | active_plan_tiles
        self.abandoned_owned_tiles.difference_update(rescued)
        return set(self.abandoned_owned_tiles)

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

    def patrol_tiles(self) -> set[Position]:
        """Return every allied transport tile this process has verified.

        Construction ownership remains separate in ``BuilderBot`` because it
        controls capacity-safe merges.  Patrol responsibility is deliberately
        transferable: a replacement scout can adopt and repair a line after
        its original builder dies.
        """
        return set(self.blueprint)
