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
_BASE_SYMMETRIES = ("rotational", "vertical", "horizontal")
_DIAGONAL_SYMMETRIES = ("main_diagonal", "anti_diagonal")


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

        # ``Position`` is used as the key across every cache index.  Allocate
        # its complete canonical set once, rather than constructing equivalent
        # coordinates while scanning, expanding a Core footprint, or accepting
        # positions returned by the Controller.  Besides avoiding allocation
        # pressure in a turn, this makes every cached coordinate refer to the
        # same object identity.
        self._positions: tuple[tuple[Position, ...], ...] = tuple(
            tuple(Position(x, y) for y in range(map_height))
            for x in range(map_width)
        )

        self.tiles: dict[Position, TileRecord] = {}
        # These indexes mirror the canonical cached observations and keep bot
        # algorithms readable without ever returning to the Controller API.
        self.environments: dict[Position, Environment] = {}
        self.buildings: dict[Position, tuple[EntityType, Team] | None] = {}
        self.conveyor_directions: dict[Position, Direction] = {}

        # Direct observations remain separate from terrain inferred after a
        # map symmetry is proven.  Inference can accelerate planning, but it
        # must never become evidence for proving the symmetry itself.
        self.inferred_tiles: set[Position] = set()
        self.possible_symmetries: set[str] = set(_BASE_SYMMETRIES)
        if map_width == map_height:
            self.possible_symmetries.update(_DIAGONAL_SYMMETRIES)
        self.confirmed_symmetry: str | None = None
        self.observed_core_positions: dict[Team, Position] = {}
        self.inferred_core_positions: dict[Team, Position] = {}
        self.inferred_core_tiles: set[Position] = set()

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
        visible_tiles = {
            self._canonical_position(pos)
            for pos in controller.get_nearby_tiles()
        }
        self.visible_tiles = visible_tiles
        self.newly_visible_tiles = visible_tiles - self._last_vision_set
        self.newly_observed_tiles = set()

        missing_visible_tiles = {
            # Symmetry-inferred terrain is useful for planning, but its first
            # physical sighting must still be queried and recorded as real
            # evidence.
            pos for pos in visible_tiles if pos not in self.observed_tiles
        }
        if self._scan_turn_count == 0 and split_initial_scan:
            origin = self._canonical_position(controller.get_position())
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
            if entity_type == EntityType.CORE:
                self._remember_observed_core(pos, team)
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
        self.current_position = (
            own[0]
            if own is not None
            else self._canonical_position(controller.get_position())
        )
        self._update_symmetry()
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
        return self.environments.get(self._canonical_position(pos))

    def building_id_at(self, pos: Position) -> int | None:
        """Return the last visible building ID on ``pos`` from the tile cache."""
        record = self.tiles.get(self._canonical_position(pos))
        return None if record is None else record[1]

    def building_at(self, pos: Position) -> tuple[EntityType, Team] | None:
        """Return cached building type and team at ``pos``."""
        return self.buildings.get(self._canonical_position(pos))

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
        return self.visible_builder_ids.get(self._canonical_position(pos))

    def marker_ids(self) -> tuple[int, ...]:
        """Return visible marker IDs already classified during the scan."""
        return tuple(
            entity_id
            for entity_id in self.visible_entity_order
            if self.entity_type(entity_id) == EntityType.MARKER
        )

    def mirrored_position(
            self,
            pos: Position,
            symmetry: str | None = None,
    ) -> Position | None:
        """Return ``pos`` reflected by a confirmed or explicitly named symmetry."""
        selected = self.confirmed_symmetry if symmetry is None else symmetry
        if selected not in _BASE_SYMMETRIES + _DIAGONAL_SYMMETRIES:
            return None
        if (
            selected in _DIAGONAL_SYMMETRIES
            and self.map_width != self.map_height
        ):
            return None
        return self._mirror_position(pos, selected)

    def core_position_for_team(self, team: Team) -> Position | None:
        """Return a directly seen core, or a core inferred from confirmed symmetry."""
        return self.observed_core_positions.get(team) or self.inferred_core_positions.get(team)

    def enemy_core_position(self, own_team: Team | None) -> Position | None:
        """Return the cached counterpart core position for ``own_team``."""
        if own_team is None:
            return None
        return self.core_position_for_team(self._opposing_team(own_team))

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
        pos = self._canonical_position(pos)
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
        pos = self._canonical_position(pos)
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

    def _update_symmetry(self) -> None:
        """Confirm map symmetry from direct observations, then backfill static data."""
        if self.confirmed_symmetry is not None:
            self._backfill_symmetric_static_data(self.newly_observed_tiles)
            return

        for symmetry in tuple(self.possible_symmetries):
            if (
                not self._symmetry_matches_new_terrain(symmetry)
                or not self._symmetry_matches_known_cores(symmetry)
            ):
                self.possible_symmetries.discard(symmetry)

        if len(self.possible_symmetries) != 1:
            return
        self.confirmed_symmetry = next(iter(self.possible_symmetries))
        self._backfill_symmetric_static_data(self.observed_tiles)

    def _symmetry_matches_new_terrain(self, symmetry: str) -> bool:
        """Compare fresh terrain only with its directly observed counterpart."""
        for pos in self.newly_observed_tiles:
            counterpart = self._mirror_position(pos, symmetry)
            if counterpart not in self.observed_tiles:
                continue
            if self.environments[counterpart] != self.environments[pos]:
                return False
        return True

    def _symmetry_matches_known_cores(self, symmetry: str) -> bool:
        """Reject a symmetry when directly seen cores cannot be paired by it."""
        for team, core_pos in self.observed_core_positions.items():
            counterpart = self._mirror_position(core_pos, symmetry)
            opposing_team = self._opposing_team(team)
            known_counterpart = self.observed_core_positions.get(opposing_team)
            if known_counterpart is not None and known_counterpart != counterpart:
                return False
            # If the expected centre is visible and directly terrain-scanned,
            # the batched building scan is authoritative for that tile.
            if counterpart not in self.visible_tiles or counterpart not in self.observed_tiles:
                continue
            building = self.buildings.get(counterpart)
            if (
                building is None
                or building[0] != EntityType.CORE
                or building[1] != opposing_team
            ):
                return False
        return True

    def _backfill_symmetric_static_data(
            self,
            source_tiles: set[Position],
    ) -> None:
        """Mirror newly known immutable terrain and every known Core footprint."""
        if self.confirmed_symmetry is None:
            return
        for pos in source_tiles:
            self._remember_inferred_environment(
                self._mirror_position(pos, self.confirmed_symmetry),
                self.environments[pos],
            )

        for team, core_pos in tuple(self.observed_core_positions.items()):
            opposing_team = self._opposing_team(team)
            if opposing_team in self.observed_core_positions:
                continue
            counterpart = self._mirror_position(core_pos, self.confirmed_symmetry)
            self.inferred_core_positions[opposing_team] = counterpart
            self._remember_inferred_core(counterpart, opposing_team)

    def _remember_inferred_environment(self, pos: Position, env: Environment) -> None:
        """Store symmetry-derived immutable terrain without marking it observed."""
        pos = self._canonical_position(pos)
        if pos in self.observed_tiles:
            return
        record = self.tiles.get(pos)
        building_id = None if record is None else record[1]
        entity_type = None if record is None else record[2]
        team = None if record is None else record[3]
        self.environments[pos] = env
        self.tiles[pos] = (env, building_id, entity_type, team)
        self.buildings.setdefault(
            pos,
            None if entity_type is None or team is None else (entity_type, team),
        )
        self.inferred_tiles.add(pos)

    def _remember_observed_core(self, pos: Position, team: Team) -> None:
        """Record one immutable Core centre discovered by the batched scan."""
        pos = self._canonical_position(pos)
        self.observed_core_positions[team] = pos
        self.inferred_core_positions.pop(team, None)

    def _remember_inferred_core(self, core_pos: Position, team: Team) -> None:
        """Fill the inferred 3x3 Core footprint without inventing an entity ID."""
        for tile in self._building_tiles(core_pos, EntityType.CORE):
            if tile not in self.environments:
                # Cores are generated on traversable empty terrain.  Usually
                # the mirrored source footprint has already supplied this;
                # this fallback keeps the static footprint internally usable.
                self._remember_inferred_environment(tile, Environment.EMPTY)
            record = self.tiles[tile]
            if record[1] is not None:
                # A direct building observation is always more authoritative
                # than a prediction, even on a map guaranteed symmetric.
                continue
            self.tiles[tile] = (self.environments[tile], None, EntityType.CORE, team)
            self.buildings[tile] = (EntityType.CORE, team)
            self.inferred_core_tiles.add(tile)

    def _mirror_position(self, pos: Position, symmetry: str) -> Position:
        """Return a preallocated coordinate transformed by one legal symmetry."""
        pos = self._canonical_position(pos)
        if symmetry == "vertical":
            return self._positions[self.map_width - 1 - pos.x][pos.y]
        if symmetry == "horizontal":
            return self._positions[pos.x][self.map_height - 1 - pos.y]
        if symmetry == "main_diagonal":
            return self._positions[pos.y][pos.x]
        if symmetry == "anti_diagonal":
            return self._positions[self.map_width - 1 - pos.y][self.map_height - 1 - pos.x]
        return self._positions[self.map_width - 1 - pos.x][self.map_height - 1 - pos.y]

    @staticmethod
    def _opposing_team(team: Team) -> Team:
        """Return the only opposing team in a two-team match."""
        return Team.B if team == Team.A else Team.A

    def _remember_environment(self, controller: Controller, pos: Position) -> None:
        """Store one previously unseen tile's permanent environment."""
        pos = self._canonical_position(pos)
        env = controller.get_tile_env(pos)
        self.environments[pos] = env
        self.buildings[pos] = None
        self.tiles[pos] = (env, None, None, None)
        self.observed_tiles.add(pos)
        self.inferred_tiles.discard(pos)
        self.inferred_core_tiles.discard(pos)
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
                self._canonical_position(controller.get_position(entity_id)),
                controller.get_entity_type(entity_id),
                controller.get_team(entity_id),
            )
        elif known[1] == EntityType.BUILDER_BOT:
            entity = (
                self._canonical_position(controller.get_position(entity_id)),
                known[1],
                known[2],
            )
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
            return (self._canonical_position(pos),)
        return tuple(
            self._positions[pos.x + dx][pos.y + dy]
            for dx, dy in _CORE_FOOTPRINT_OFFSETS
            if 0 <= pos.x + dx < self.map_width and 0 <= pos.y + dy < self.map_height
        )

    def _canonical_position(self, pos: Position) -> Position:
        """Return the preallocated cache coordinate matching an in-bounds position."""
        return self._positions[pos.x][pos.y]

    def _set_building(
            self,
            pos: Position,
            building_id: int | None,
            entity_type: EntityType | None,
            team: Team | None,
    ) -> None:
        """Synchronize all tile indexes after a building observation changes."""
        pos = self._canonical_position(pos)
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
