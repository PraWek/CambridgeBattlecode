"""Per-turn tile and entity observations shared by all RC bot roles."""

from cambc import Controller, Direction, EntityType, Environment, Position, Team


TileRecord = tuple[Environment, int | None, EntityType | None, Team | None]
EntityRecord = tuple[Position, EntityType, Team]


_DIRECTIONAL_TYPES = {
    EntityType.CONVEYOR,
    EntityType.SPLITTER,
    EntityType.ARMOURED_CONVEYOR,
    EntityType.GUNNER,
    EntityType.SENTINEL,
    EntityType.BREACH,
}
_DYNAMIC_DIRECTION_TYPES = {EntityType.GUNNER}
_CORE_FOOTPRINT_OFFSETS = tuple(
    (dx, dy)
    for dx in range(-1, 2)
    for dy in range(-1, 2)
)


class TileCache:
    """Keep permanent terrain and the latest visible entity state for one bot.

    ``tiles`` is the canonical tile dictionary requested by the bot: every
    entry has the form ``Position(x, y) -> (env, building_id, type, team)``.
    Terrain is immutable, while the last three values are refreshed from the
    batched building query whenever that tile is in vision.
    """

    def __init__(self, map_width: int, map_height: int) -> None:
        """Initialize persistent map knowledge and transient per-turn views."""
        self.map_width = map_width
        self.map_height = map_height

        self.tiles: dict[Position, TileRecord] = {}
        # These indexes mirror the canonical cached observations and keep bot
        # algorithms readable without ever returning to the Controller API.
        self.environments: dict[Position, Environment] = {}
        self.buildings: dict[Position, tuple[EntityType, Team] | None] = {}
        self.conveyor_directions: dict[Position, Direction] = {}

        self.observed_tiles: set[Position] = set()
        self.visible_tiles: set[Position] = set()
        self.newly_visible_tiles: set[Position] = set()
        self.newly_observed_tiles: set[Position] = set()
        self._last_vision_set: set[Position] = set()
        self._scan_turn_count = 0
        self._first_turn_deferred_tiles: set[Position] | None = None

        # Entity metadata is immutable except for a builder's position and a
        # gunner's rotation.  Keeping it lets later scans reuse type/team and
        # static-building positions instead of asking the API again.
        self.entities: dict[int, EntityRecord] = {}
        self.entity_directions: dict[int, Direction] = {}
        self.visible_entity_ids: set[int] = set()
        self.visible_entity_order: list[int] = []
        self.visible_builder_ids: dict[Position, int] = {}
        self.marker_values: dict[int, int] = {}
        self.current_position: Position | None = None

    def scan_turn(
            self,
            controller: Controller,
            own_id: int,
            split_initial_scan: bool = False,
    ) -> None:
        """Populate the cache once from this turn's batched vision queries.

        A newborn builder may split its first terrain scan across two turns.
        The first half is closest to its spawn tile, so it still learns its
        nearby core and immediate exits; the other half is queried on its
        next turn. Other roles retain the complete single-turn scan.
        """
        visible_tiles = set(controller.get_nearby_tiles())
        self.visible_tiles = visible_tiles
        self.newly_visible_tiles = visible_tiles - self._last_vision_set
        self.newly_observed_tiles = set()

        missing_visible_tiles = {
            pos for pos in visible_tiles if pos not in self.environments
        }
        if self._scan_turn_count == 0 and split_initial_scan:
            origin = controller.get_position()
            ordered_tiles = sorted(
                missing_visible_tiles,
                key=lambda pos: (origin.distance_squared(pos), pos.y, pos.x),
            )
            first_half_count = (len(ordered_tiles) + 1) // 2
            tiles_to_scan = ordered_tiles[:first_half_count]
            self._first_turn_deferred_tiles = set(ordered_tiles[first_half_count:])
        elif self._scan_turn_count == 1 and self._first_turn_deferred_tiles is not None:
            # A moved builder can no longer sense a few cells from its first
            # disc. They stay unknown until naturally seen again; querying
            # them here would be illegal and would defeat the bounded scan.
            tiles_to_scan = self._first_turn_deferred_tiles & visible_tiles
            self._first_turn_deferred_tiles = None
        else:
            # Environment never changes. After bootstrap, query only the
            # newly visible / previously missed portion of vision.
            tiles_to_scan = missing_visible_tiles

        for pos in tiles_to_scan:
            self._remember_environment(controller, pos)

        # Buildings and builders are dynamic.  Reset only visible, non-wall
        # cells before repopulating them from two batched ID lists.  A known
        # wall can never contain either, so it receives no further work.
        for pos in visible_tiles:
            env = self.environments.get(pos)
            if env is None or env == Environment.WALL:
                continue
            self._set_building(pos, None, None, None)

        building_ids = controller.get_nearby_buildings()
        unit_ids = controller.get_nearby_units()
        self.visible_entity_ids = set()
        self.visible_entity_order = []
        self.visible_builder_ids = {}
        self.marker_values = {}

        seen_ids: set[int] = set()
        for entity_id in building_ids:
            seen_ids.add(entity_id)
            pos, entity_type, team = self._entity_for_scan(controller, entity_id)
            self._remember_visible_entity(entity_id)
            direction = self._direction_for_scan(controller, entity_id, entity_type)
            for tile in self._building_tiles(pos, entity_type):
                if tile not in visible_tiles or tile not in self.environments:
                    continue
                self._set_building(tile, entity_id, entity_type, team)
                if direction is not None:
                    self.conveyor_directions[tile] = direction

        for entity_id in unit_ids:
            if entity_id in seen_ids:
                pos, entity_type, team = self.entities[entity_id]
            else:
                pos, entity_type, team = self._entity_for_scan(controller, entity_id)
                self._direction_for_scan(controller, entity_id, entity_type)
            self._remember_visible_entity(entity_id)
            if entity_type == EntityType.BUILDER_BOT:
                self.visible_builder_ids[pos] = entity_id

        own = self.entities.get(own_id)
        self.current_position = own[0] if own is not None else controller.get_position()
        self._last_vision_set = visible_tiles
        self._scan_turn_count += 1

    def cache_friendly_marker_values(self, controller: Controller, own_team: Team) -> None:
        """Read each visible friendly marker once for consumers of the cache."""
        for entity_id in self.visible_entity_order:
            entity = self.entities.get(entity_id)
            if entity is None:
                continue
            _, entity_type, team = entity
            if entity_type != EntityType.MARKER or team != own_team:
                continue
            try:
                self.marker_values[entity_id] = controller.get_marker_value(entity_id)
            except Exception:
                # A marker can disappear between the batched scan and this
                # read because another entity acted earlier in the round.
                continue

    def environment_at(self, pos: Position) -> Environment | None:
        """Return cached immutable terrain at ``pos`` without an API call."""
        return self.environments.get(pos)

    def building_id_at(self, pos: Position) -> int | None:
        """Return the last visible building ID on ``pos`` from the tile cache."""
        record = self.tiles.get(pos)
        return None if record is None else record[1]

    def building_at(self, pos: Position) -> tuple[EntityType, Team] | None:
        """Return cached building type and team at ``pos``."""
        return self.buildings.get(pos)

    def entity_position(self, entity_id: int) -> Position | None:
        """Return a cached entity position, if this bot has observed it."""
        entity = self.entities.get(entity_id)
        return None if entity is None else entity[0]

    def entity_type(self, entity_id: int) -> EntityType | None:
        """Return a cached entity type, if this bot has observed it."""
        entity = self.entities.get(entity_id)
        return None if entity is None else entity[1]

    def entity_team(self, entity_id: int) -> Team | None:
        """Return a cached entity team, if this bot has observed it."""
        entity = self.entities.get(entity_id)
        return None if entity is None else entity[2]

    def entity_direction(self, entity_id: int) -> Direction | None:
        """Return a cached direction for a directional entity."""
        return self.entity_directions.get(entity_id)

    def builder_id_at(self, pos: Position) -> int | None:
        """Return the currently visible builder occupying ``pos``."""
        return self.visible_builder_ids.get(pos)

    def marker_ids(self) -> tuple[int, ...]:
        """Return visible marker IDs already classified during the scan."""
        return tuple(
            entity_id
            for entity_id in self.visible_entity_order
            if self.entity_type(entity_id) == EntityType.MARKER
        )

    def remember_building(
            self,
            pos: Position,
            building_id: int,
            entity_type: EntityType,
            team: Team,
            direction: Direction | None = None,
            marker_value: int | None = None,
    ) -> None:
        """Apply this bot's own successful build to the cache immediately."""
        self.entities[building_id] = (pos, entity_type, team)
        self._remember_visible_entity(building_id)
        for tile in self._building_tiles(pos, entity_type):
            if tile not in self.environments:
                continue
            self._set_building(tile, building_id, entity_type, team)
            if direction is not None:
                self.conveyor_directions[tile] = direction
        if direction is not None:
            self.entity_directions[building_id] = direction
        if marker_value is not None:
            self.marker_values[building_id] = marker_value

    def forget_building(self, pos: Position) -> None:
        """Apply this bot's own successful destruction to the cache immediately."""
        building_id = self.building_id_at(pos)
        if building_id is not None:
            self.entities.pop(building_id, None)
            self.entity_directions.pop(building_id, None)
            self.marker_values.pop(building_id, None)
            self.visible_entity_ids.discard(building_id)
        record = self.tiles.get(pos)
        if record is None:
            return
        self._set_building(pos, None, None, None)

    def _remember_environment(self, controller: Controller, pos: Position) -> None:
        """Store one previously unseen tile's permanent environment."""
        env = controller.get_tile_env(pos)
        self.environments[pos] = env
        self.buildings[pos] = None
        self.tiles[pos] = (env, None, None, None)
        self.observed_tiles.add(pos)
        self.newly_observed_tiles.add(pos)

    def _remember_visible_entity(self, entity_id: int) -> None:
        """Record a visible entity once while preserving batched-query order."""
        if entity_id in self.visible_entity_ids:
            return
        self.visible_entity_ids.add(entity_id)
        self.visible_entity_order.append(entity_id)

    def _entity_for_scan(
            self,
            controller: Controller,
            entity_id: int,
    ) -> EntityRecord:
        """Return entity metadata, querying only mutable or newly seen fields."""
        known = self.entities.get(entity_id)
        if known is None:
            entity = (
                controller.get_position(entity_id),
                controller.get_entity_type(entity_id),
                controller.get_team(entity_id),
            )
        elif known[1] == EntityType.BUILDER_BOT:
            entity = (controller.get_position(entity_id), known[1], known[2])
        else:
            entity = known
        self.entities[entity_id] = entity
        return entity

    def _direction_for_scan(
            self,
            controller: Controller,
            entity_id: int,
            entity_type: EntityType,
    ) -> Direction | None:
        """Refresh a direction only when that entity type can change it."""
        if entity_type not in _DIRECTIONAL_TYPES:
            return None
        if (
            entity_type not in _DYNAMIC_DIRECTION_TYPES
            and entity_id in self.entity_directions
        ):
            return self.entity_directions[entity_id]
        direction = controller.get_direction(entity_id)
        self.entity_directions[entity_id] = direction
        return direction

    def _building_tiles(self, pos: Position, entity_type: EntityType) -> tuple[Position, ...]:
        """Return tiles occupied by a building, expanding the 3x3 core."""
        if entity_type != EntityType.CORE:
            return (pos,)
        return tuple(
            Position(pos.x + dx, pos.y + dy)
            for dx, dy in _CORE_FOOTPRINT_OFFSETS
            if 0 <= pos.x + dx < self.map_width and 0 <= pos.y + dy < self.map_height
        )

    def _set_building(
            self,
            pos: Position,
            building_id: int | None,
            entity_type: EntityType | None,
            team: Team | None,
    ) -> None:
        """Synchronize all tile indexes after a building observation changes."""
        env = self.environments[pos]
        self.tiles[pos] = (env, building_id, entity_type, team)
        self.buildings[pos] = (
            None if building_id is None or entity_type is None or team is None
            else (entity_type, team)
        )
        if building_id is None or entity_type not in {
            EntityType.CONVEYOR,
            EntityType.ARMOURED_CONVEYOR,
        }:
            self.conveyor_directions.pop(pos, None)
