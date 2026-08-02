from cambc import Controller, Direction, EntityType, Environment, Position, Team


_DIRECTIONAL_TYPES = {
    EntityType.CONVEYOR,
    EntityType.ARMOURED_CONVEYOR,
    EntityType.SPLITTER,
    EntityType.GUNNER,
    EntityType.SENTINEL,
    EntityType.BREACH,
}


class TileCache:
    """Per-unit persistent map memory populated with one batched scan per turn."""

    def __init__(self, map_width: int, map_height: int) -> None:
        self.map_width = map_width
        self.map_height = map_height
        self.environments: dict[Position, Environment] = {}
        self.buildings: dict[Position, tuple[EntityType, Team] | None] = {}
        self.building_ids: dict[Position, int] = {}
        self.builder_ids: dict[Position, int] = {}
        self.conveyor_directions: dict[Position, Direction] = {}

        self.entity_types: dict[int, EntityType] = {}
        self.entity_teams: dict[int, Team] = {}
        self.entity_positions: dict[int, Position] = {}
        self.entity_directions: dict[int, Direction] = {}
        self.marker_values: dict[int, int] = {}

        self.current_position: Position | None = None
        self.visible_tiles: set[Position] = set()
        self.newly_observed_tiles: set[Position] = set()
        self.observed_tiles: set[Position] = set()
        self.visible_entity_ids: set[int] = set()

    def scan_turn(
            self,
            controller: Controller,
            entity_id: int,
            split_initial_scan: bool = False,
    ) -> None:
        """Refresh visible dynamic state and query permanent terrain only once."""
        self.current_position = controller.get_position()
        self.visible_tiles = set(controller.get_nearby_tiles())

        unseen = self.visible_tiles - self.observed_tiles
        if split_initial_scan and not self.observed_tiles and len(unseen) > 1:
            ordered = sorted(unseen, key=lambda pos: (pos.x, pos.y))
            unseen = set(ordered[::2])
        self.newly_observed_tiles = unseen
        for pos in unseen:
            self.environments[pos] = controller.get_tile_env(pos)
        self.observed_tiles.update(unseen)

        # Clear only the current vision.  Information outside vision remains a
        # useful last-known map, while visible destruction/movement is exact.
        for pos in self.visible_tiles:
            self.buildings[pos] = None
            self.building_ids.pop(pos, None)
            self.builder_ids.pop(pos, None)
            self.conveyor_directions.pop(pos, None)

        entity_ids = set(controller.get_nearby_entities())
        entity_ids.add(entity_id)
        self.visible_entity_ids = entity_ids
        for nearby_id in entity_ids:
            entity_type = controller.get_entity_type(nearby_id)
            team = controller.get_team(nearby_id)
            pos = controller.get_position(nearby_id)
            self.entity_types[nearby_id] = entity_type
            self.entity_teams[nearby_id] = team
            self.entity_positions[nearby_id] = pos

            if entity_type == EntityType.BUILDER_BOT:
                self.builder_ids[pos] = nearby_id
                continue

            footprint = (pos,)
            if entity_type == EntityType.CORE:
                footprint = tuple(
                    Position(pos.x + dx, pos.y + dy)
                    for dx in range(-1, 2)
                    for dy in range(-1, 2)
                    if 0 <= pos.x + dx < self.map_width
                    and 0 <= pos.y + dy < self.map_height
                )
            for tile in footprint:
                self.buildings[tile] = (entity_type, team)
                self.building_ids[tile] = nearby_id

            if entity_type in _DIRECTIONAL_TYPES:
                direction = controller.get_direction(nearby_id)
                self.entity_directions[nearby_id] = direction
                if entity_type in {
                    EntityType.CONVEYOR,
                    EntityType.ARMOURED_CONVEYOR,
                    EntityType.SPLITTER,
                }:
                    self.conveyor_directions[pos] = direction

    def cache_friendly_marker_values(self, controller: Controller, team: Team) -> None:
        """Read visible friendly marker payloads once for this turn."""
        visible_markers = set()
        for entity_id in self.visible_entity_ids:
            if (
                self.entity_types.get(entity_id) == EntityType.MARKER
                and self.entity_teams.get(entity_id) == team
            ):
                self.marker_values[entity_id] = controller.get_marker_value(entity_id)
                visible_markers.add(entity_id)
        for entity_id in list(self.marker_values):
            if entity_id not in visible_markers:
                self.marker_values.pop(entity_id, None)

    def environment_at(self, pos: Position) -> Environment | None:
        return self.environments.get(pos)

    def building_at(self, pos: Position) -> tuple[EntityType, Team] | None:
        return self.buildings.get(pos)

    def building_id_at(self, pos: Position) -> int | None:
        return self.building_ids.get(pos)

    def builder_id_at(self, pos: Position) -> int | None:
        return self.builder_ids.get(pos)

    def entity_type(self, entity_id: int) -> EntityType | None:
        return self.entity_types.get(entity_id)

    def entity_team(self, entity_id: int) -> Team | None:
        return self.entity_teams.get(entity_id)

    def entity_position(self, entity_id: int) -> Position | None:
        return self.entity_positions.get(entity_id)

    def entity_direction(self, entity_id: int) -> Direction | None:
        return self.entity_directions.get(entity_id)

    def marker_ids(self) -> list[int]:
        return [
            entity_id
            for entity_id in self.visible_entity_ids
            if self.entity_types.get(entity_id) == EntityType.MARKER
        ]

    def remember_building(
            self,
            pos: Position,
            entity_id: int,
            entity_type: EntityType,
            team: Team,
            direction: Direction | None = None,
            marker_value: int | None = None,
    ) -> None:
        """Apply a successful same-turn build to the cache immediately."""
        self.buildings[pos] = (entity_type, team)
        self.building_ids[pos] = entity_id
        self.entity_types[entity_id] = entity_type
        self.entity_teams[entity_id] = team
        self.entity_positions[entity_id] = pos
        if direction is not None:
            self.entity_directions[entity_id] = direction
            if entity_type in {
                EntityType.CONVEYOR,
                EntityType.ARMOURED_CONVEYOR,
                EntityType.SPLITTER,
            }:
                self.conveyor_directions[pos] = direction
        if marker_value is not None:
            self.marker_values[entity_id] = marker_value

    def forget_building(self, pos: Position) -> None:
        """Apply a successful same-turn destroy to the cache immediately."""
        entity_id = self.building_ids.pop(pos, None)
        self.buildings[pos] = None
        self.conveyor_directions.pop(pos, None)
        if entity_id is not None:
            self.entity_types.pop(entity_id, None)
            self.entity_teams.pop(entity_id, None)
            self.entity_positions.pop(entity_id, None)
            self.entity_directions.pop(entity_id, None)
            self.marker_values.pop(entity_id, None)
