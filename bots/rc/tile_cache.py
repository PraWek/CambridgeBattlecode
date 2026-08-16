"""Per-turn tile and entity observations shared by all RC bot roles."""

from collections import deque

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
# Controller calls have a material CPU cost on crowded maps.  Seven calls leave
# room for navigation after a partial entity scan; terrain and the own unit
# are always completed before a role is allowed to act.
_SCAN_API_CALL_LIMIT = 7
# Confirming a symmetry may happen after a scout has seen hundreds of tiles.
# Mirroring all of them in one bot turn exceeds the 2 ms limit, so drain the
# historical cache over several later turns instead.  This work shares a
# 2 ms turn with the ordinary scan and role logic, so process only one tile
# at a time.  The historical source uses direct-observation order and a
# cursor, avoiding one large scheduling pass at the exact turn symmetry
# becomes known.
_SYMMETRY_BACKFILL_TILES_PER_TURN = 1


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
        # Consumers can yield their current turn when this expensive
        # transition happens instead of chaining another heavy planner after
        # the first full symmetry backfill is scheduled.
        self.symmetry_confirmed_this_turn = False
        self.observed_core_positions: dict[Team, Position] = {}
        self.inferred_core_positions: dict[Team, Position] = {}
        self.inferred_core_tiles: set[Position] = set()
        self._symmetry_backfill_pending: deque[Position] = deque()
        self._symmetry_backfill_historical_index = 0
        self._symmetry_backfill_historical_end = 0

        self.observed_tiles: set[Position] = set()
        self._observed_tile_order: list[Position] = []
        self.visible_tiles: set[Position] = set()
        self.newly_visible_tiles: set[Position] = set()
        self.newly_observed_tiles: set[Position] = set()
        # Terrain may be physically read during a partial scan, while the
        # entity phase still runs out of API budget before ``_update_symmetry``
        # can compare it.  Keep such tiles as evidence until a complete scan
        # actually consumes them; otherwise the only observation that could
        # disprove a symmetry would be silently lost on the next turn.
        self._pending_symmetry_tiles: set[Position] = set()
        self._last_vision_set: set[Position] = set()
        # Terrain that was visible but did not fit in a previous scan budget.
        # Keep it until the builder sees it again and can legally query it.
        self._pending_terrain_tiles: set[Position] = set()
        self.scan_incomplete_this_turn = False
        self.role_cache_ready_this_turn = False
        self.scan_api_calls_this_turn = 0

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
    ) -> None:
        """Populate the cache once from this turn's batched vision queries.

        Every Controller read made by the cache consumes one of the bounded
        scan calls.  When the budget is exhausted, this method stops
        immediately; terrain not yet queried stays queued for a later turn.
        Callers must yield when ``scan_incomplete_this_turn`` is true.
        """
        self.symmetry_confirmed_this_turn = False
        self.scan_incomplete_this_turn = False
        self.role_cache_ready_this_turn = False
        self.scan_api_calls_this_turn = 0
        if not self._take_scan_api_call():
            return
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
        # A terrain tile may have been inferred through symmetry, but it must
        # still be physically read before becoming evidence.  Give pending
        # cells deterministic priority, without an extra get_position() call.
        tiles_to_scan = sorted(
            (self._pending_terrain_tiles & visible_tiles) | missing_visible_tiles,
            key=lambda pos: (pos.y, pos.x),
        )
        for index, pos in enumerate(tiles_to_scan):
            if not self._remember_environment(controller, pos):
                self._pending_terrain_tiles.update(tiles_to_scan[index:])
                self._finish_incomplete_scan()
                return
            self._pending_terrain_tiles.discard(pos)

        # Buildings and builders are dynamic.  Reset only visible, non-wall
        # cells before repopulating them from two batched ID lists.  A known
        # wall can never contain either, so it receives no further work.
        for pos in visible_tiles:
            env = self.environments.get(pos)
            if env is None or env == Environment.WALL:
                continue
            self._set_building(pos, None, None, None)

        if not self._take_scan_api_call():
            self._finish_incomplete_scan()
            return
        unit_ids = controller.get_nearby_units()
        if not self._take_scan_api_call():
            self._finish_incomplete_scan()
            return
        building_ids = controller.get_nearby_buildings()
        self.visible_entity_ids = set()
        self.visible_entity_order = []
        self.visible_builder_ids = {}
        self.marker_values = {}

        # Cache our mutable position first.  In a crowded vision this keeps
        # the bot's own start-of-turn coordinates available as soon as a scan
        # can complete, instead of putting it behind unrelated buildings.
        # The role needs its own position first.  Process Core/buildings next
        # so a newly spawned Intruder can find its friendly Core even when a
        # dense group of nearby builders prevents a complete entity scan.
        own_unit_ids = [entity_id for entity_id in unit_ids if entity_id == own_id]
        other_unit_ids = [entity_id for entity_id in unit_ids if entity_id != own_id]
        for entity_id in own_unit_ids:
            entity = self._entity_for_scan(controller, entity_id)
            if entity is None:
                self._finish_incomplete_scan()
                return
            pos, entity_type, team = entity
            self._remember_visible_entity(entity_id)
            self.current_position = pos
            self.role_cache_ready_this_turn = True
            self._direction_for_scan(controller, entity_id, entity_type)
            if self.scan_incomplete_this_turn:
                self._finish_incomplete_scan()
                return
            if entity_type == EntityType.BUILDER_BOT:
                self.visible_builder_ids[pos] = entity_id

        for entity_id in building_ids:
            entity = self._entity_for_scan(controller, entity_id)
            if entity is None:
                self._finish_incomplete_scan()
                return
            pos, entity_type, team = entity
            self._remember_visible_entity(entity_id)
            if entity_type == EntityType.CORE:
                self._remember_observed_core(pos, team)
            direction = self._direction_for_scan(controller, entity_id, entity_type)
            if self.scan_incomplete_this_turn:
                self._finish_incomplete_scan()
                return
            for tile in self._building_tiles(pos, entity_type):
                if tile not in visible_tiles or tile not in self.environments:
                    continue
                self._set_building(tile, entity_id, entity_type, team)
                if direction is not None:
                    self.conveyor_directions[tile] = direction

        for entity_id in other_unit_ids:
            entity = self._entity_for_scan(controller, entity_id)
            if entity is None:
                self._finish_incomplete_scan()
                return
            pos, entity_type, team = entity
            self._remember_visible_entity(entity_id)
            self._direction_for_scan(controller, entity_id, entity_type)
            if self.scan_incomplete_this_turn:
                self._finish_incomplete_scan()
                return
            if entity_type == EntityType.BUILDER_BOT:
                self.visible_builder_ids[pos] = entity_id

        own = self.entities.get(own_id)
        if own is None:
            self._finish_incomplete_scan()
            return
        self.current_position = own[0]
        self._update_symmetry()
        self._last_vision_set = visible_tiles

    def cache_friendly_marker_values(self, controller: Controller, own_team: Team) -> bool:
        """Read each visible friendly marker once for consumers of the cache."""
        for entity_id in self.visible_entity_order:
            entity = self.entities.get(entity_id)
            if entity is None:
                continue
            _, entity_type, team = entity
            if entity_type != EntityType.MARKER or team != own_team:
                continue
            if not self._take_scan_api_call():
                return False
            try:
                self.marker_values[entity_id] = controller.get_marker_value(entity_id)
            except Exception:
                # A marker can disappear between the batched scan and this
                # read because another entity acted earlier in the round.
                continue
        return True

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
        """Confirm symmetry and amortize its terrain inference across turns."""
        if self.confirmed_symmetry is not None:
            self._schedule_symmetric_backfill(self.newly_observed_tiles)
            self._backfill_symmetric_cores()
            self._continue_symmetric_backfill()
            self._pending_symmetry_tiles.clear()
            return

        # A tile enters this set when its environment is read, not merely when
        # the turn begins.  It therefore survives an incomplete scan and is
        # compared during the first later scan that reaches this method.
        for symmetry in tuple(self.possible_symmetries):
            if (
                not self._symmetry_matches_pending_terrain(symmetry)
                or not self._symmetry_matches_known_cores(symmetry)
            ):
                self.possible_symmetries.discard(symmetry)

        # Every pending tile has now been compared against every directly seen
        # counterpart.  If a counterpart was not seen yet, its own eventual
        # direct observation will be queued and perform that comparison then.
        self._pending_symmetry_tiles.clear()

        if len(self.possible_symmetries) != 1:
            return
        self.confirmed_symmetry = next(iter(self.possible_symmetries))
        self.symmetry_confirmed_this_turn = True
        # Infer the opposing Core immediately: it supplies a useful target
        # for scouting, while the much larger terrain mirror is deferred.
        self._backfill_symmetric_cores()
        self._symmetry_backfill_historical_index = 0
        self._symmetry_backfill_historical_end = len(self._observed_tile_order)
        self._continue_symmetric_backfill()

    def _symmetry_matches_pending_terrain(self, symmetry: str) -> bool:
        """Compare unprocessed direct terrain observations with their counterparts."""
        for pos in self._pending_symmetry_tiles:
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

    def _schedule_symmetric_backfill(self, source_tiles: set[Position]) -> None:
        """Queue terrain first observed after symmetry confirmation."""
        self._symmetry_backfill_pending.extend(source_tiles)

    def _continue_symmetric_backfill(self) -> None:
        """Mirror one bounded terrain batch after symmetry is confirmed."""
        if self.confirmed_symmetry is None:
            return
        for _ in range(_SYMMETRY_BACKFILL_TILES_PER_TURN):
            if self._symmetry_backfill_pending:
                pos = self._symmetry_backfill_pending.popleft()
            elif (
                self._symmetry_backfill_historical_index
                < self._symmetry_backfill_historical_end
            ):
                pos = self._observed_tile_order[
                    self._symmetry_backfill_historical_index
                ]
                self._symmetry_backfill_historical_index += 1
            else:
                return
            self._remember_inferred_environment(
                self._mirror_position(pos, self.confirmed_symmetry),
                self.environments[pos],
            )

    def _backfill_symmetric_cores(self) -> None:
        """Infer only Core footprints immediately after symmetry is known."""
        if self.confirmed_symmetry is None:
            return
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
        self.inferred_tiles.update((pos,))

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
            self.inferred_core_tiles.update((tile,))

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

    def _take_scan_api_call(self) -> bool:
        """Reserve one Controller call, or stop this cache turn at its budget."""
        if self.scan_api_calls_this_turn >= _SCAN_API_CALL_LIMIT:
            self.scan_incomplete_this_turn = True
            return False
        self.scan_api_calls_this_turn += 1
        return True

    def _finish_incomplete_scan(self) -> None:
        """Record the partial vision update before yielding the bot's turn."""
        self.scan_incomplete_this_turn = True
        self._last_vision_set = self.visible_tiles

    def _remember_environment(self, controller: Controller, pos: Position) -> bool:
        """Store one previously unseen tile's permanent environment."""
        if not self._take_scan_api_call():
            return False
        pos = self._canonical_position(pos)
        env = controller.get_tile_env(pos)
        self.environments[pos] = env
        self.buildings[pos] = None
        self.tiles[pos] = (env, None, None, None)
        self.observed_tiles.update((pos,))
        self._observed_tile_order.append(pos)
        self.inferred_tiles.discard(pos)
        self.inferred_core_tiles.discard(pos)
        self.newly_observed_tiles.update((pos,))
        self._pending_symmetry_tiles.update((pos,))
        return True

    def _remember_visible_entity(self, entity_id: int) -> None:
        """Record a visible entity once while preserving batched-query order."""
        if entity_id in self.visible_entity_ids:
            return
        self.visible_entity_ids.update((entity_id,))
        self.visible_entity_order.append(entity_id)

    def _entity_for_scan(
            self,
            controller: Controller,
            entity_id: int,
    ) -> EntityRecord | None:
        """Return entity metadata, querying only mutable or newly seen fields."""
        known = self.entities.get(entity_id)
        if known is None:
            if not self._take_scan_api_call():
                return None
            pos = self._canonical_position(controller.get_position(entity_id))
            if not self._take_scan_api_call():
                return None
            entity_type = controller.get_entity_type(entity_id)
            if not self._take_scan_api_call():
                return None
            entity = (
                pos,
                entity_type,
                controller.get_team(entity_id),
            )
        elif known[1] == EntityType.BUILDER_BOT:
            if not self._take_scan_api_call():
                return None
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
        if not self._take_scan_api_call():
            return None
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

    def position_at(self, x: int, y: int) -> Position | None:
        """Return the canonical tile at ``(x, y)``, or ``None`` off-map."""
        if not (0 <= x < self.map_width and 0 <= y < self.map_height):
            return None
        return self._positions[x][y]

    def canonicalize(self, pos: Position) -> Position:
        """Convert an in-bounds external coordinate to its cached identity."""
        canonical = self.position_at(pos.x, pos.y)
        if canonical is None:
            raise ValueError(f"position outside map: {pos}")
        return canonical

    def offset(self, pos: Position, dx: int, dy: int) -> Position | None:
        """Return the canonical tile offset from ``pos``, or ``None`` off-map."""
        return self.position_at(pos.x + dx, pos.y + dy)

    def _canonical_position(self, pos: Position) -> Position:
        """Backward-compatible private alias for :meth:`canonicalize`."""
        return self.canonicalize(pos)

    def neighbor(self, pos: Position, direction: Direction) -> Position | None:
        """Return the canonical adjacent tile, or ``None`` beyond the map edge."""
        dx, dy = direction.delta()
        return self.offset(pos, dx, dy)

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
