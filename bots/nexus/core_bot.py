from cambc import Controller, Direction, EntityType, Environment, GameConstants, Position

from base import BaseBot
from constants import (
    AXIONITE_TITANIUM_THRESHOLD,
    BUILDER_DIRECTION_CODES,
    BUILDER_WORK_DIRECTIONS,
    ECONOMY_EXPANSION_INTERVAL_ROUNDS,
    ECONOMY_EXPANSION_START_ROUND,
    MARKER_KIND_SECTOR_ORE_AX,
    MARKER_KIND_SECTOR_ORE_TI,
    MARKER_KIND_SPAWN_DIRECTION,
    MARKER_KIND_SPAWN_ORE_AX,
    MARKER_KIND_SPAWN_ORE_TI,
    MAX_ADDITIONAL_BUILDER_SPAWNS,
    MAX_ECONOMY_BUILDERS,
    ORE_TYPES,
)
from economy import desired_builder_count
from geometry import encode_marker


class CoreBot(BaseBot):
    """Keeps four directional builders alive and gives each sector its ore order.

    A bot process has no shared memory with the core.  Sector markers close to
    the core therefore act as a small, persistent order board: a newborn
    builder reads the marker for the tile it was spawned from before it leaves
    the core footprint.
    """

    def __init__(self, map_width: int, map_height: int) -> None:
        """Initialize the core's map memory, sector orders, and marker board."""
        super().__init__(map_width, map_height)
        self.core_pos: Position | None = None
        self.known_env = self.tile_cache.environments
        self.known_buildings = self.tile_cache.buildings

        self.initial_spawned_directions: set[Direction] = set()
        self.replacement_direction_index = 0
        self.replacement_builders_spawned = 0
        self.sector_targets: dict[Direction, tuple[Position, Environment]] = {}
        self.sector_marker_pads: dict[Direction, Position] = {}
        self.sector_marker_values: dict[Direction, int | None] = {}
        self.spawn_order_pad: Position | None = None
        self.spawn_order_target: Position | None = None
        self.marker_sync_cursor = 0

    def run(self, controller: Controller) -> None:
        """Observe the map, update sector orders, and maintain the economy fleet."""
        if self._scan_turn(controller):
            return
        self.observe_tiles()
        self.core_pos = self.get_cached_position()
        if not self.sector_marker_pads:
            self.sector_marker_pads, self.spawn_order_pad = self.find_sector_marker_pads(controller)
            self.sector_marker_values = {
                direction: None for direction in self.sector_marker_pads
            }

        self.refresh_sector_targets(controller)
        spawned = self.try_spawn_missing_builder(controller)
        if not spawned:
            self.clear_completed_spawn_order(controller)
            # Orders can change after a nearby harvester is built.  One marker
            # write per round is a game rule, so update the board round-robin.
            self.sync_one_changed_sector_marker(controller)

    def observe_tiles(self) -> None:
        """Refresh the core's local terrain and building observations."""
        # BaseBot has already refreshed these shared cache indexes.  This
        # method deliberately keeps the role-level observation boundary while
        # performing no additional Controller queries.
        return

    def find_sector_marker_pads(
            self,
            controller: Controller,
    ) -> tuple[dict[Direction, Position], Position | None]:
        """Choose safe tiles for sector orders and spawn handoff markers."""
        if self.core_pos is None:
            return {}, None

        # Corners of the core action radius do not obstruct the four cardinal
        # exits used by the builders.  The remaining entries are fallbacks for
        # maps where a corner contains a wall or ore.
        preferred_offsets = (
            (-2, -2), (2, -2), (2, 2), (-2, 2),
            (-2, 0), (0, -2), (2, 0), (0, 2),
        )
        offsets = list(preferred_offsets)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if (
                    dx * dx + dy * dy <= GameConstants.CORE_ACTION_RADIUS_SQ
                    and (abs(dx) == 2 or abs(dy) == 2)
                    and (dx, dy) not in offsets
                ):
                    offsets.append((dx, dy))
        pads: list[Position] = []
        for dx, dy in offsets:
            pos = self.tile_cache.offset(self.core_pos, dx, dy)
            if pos is None or not controller.can_place_marker(pos):
                continue
            pads.append(pos)
            if len(pads) == len(BUILDER_WORK_DIRECTIONS) + 1:
                break
        sector_pads = {
            direction: pad
            for direction, pad in zip(BUILDER_WORK_DIRECTIONS, pads[:len(BUILDER_WORK_DIRECTIONS)])
        }
        spawn_order_pad = pads[len(BUILDER_WORK_DIRECTIONS)] if len(pads) > len(BUILDER_WORK_DIRECTIONS) else None
        return sector_pads, spawn_order_pad

    def refresh_sector_targets(self, controller: Controller) -> None:
        """Assign each direction its nearest eligible observed ore deposit."""
        if self.core_pos is None:
            return

        # A deposit remains ore after harvesting, so the harvester itself is
        # the authoritative acknowledgement that clears a core-issued order.
        for direction, target in list(self.sector_targets.items()):
            if self.is_harvester_on_tile(target[0]):
                del self.sector_targets[direction]

        titanium, _ = controller.get_global_resources()
        used_targets = {target[0] for target in self.sector_targets.values()}
        candidates: dict[Direction, list[tuple[Position, Environment]]] = {
            direction: [] for direction in BUILDER_WORK_DIRECTIONS
        }
        for pos, env in self.known_env.items():
            if env not in ORE_TYPES or self.is_harvester_on_tile(pos):
                continue
            if env == Environment.ORE_AXIONITE and titanium <= AXIONITE_TITANIUM_THRESHOLD:
                continue
            candidates[self.sector_for(pos)].append((pos, env))

        for direction in BUILDER_WORK_DIRECTIONS:
            if direction in self.sector_targets:
                continue
            available = [target for target in candidates[direction] if target[0] not in used_targets]
            if not available:
                continue
            target = min(
                available,
                key=lambda item: (
                    self.core_distance(item[0]),
                    item[0].x,
                    item[0].y,
                ),
            )
            self.sector_targets[direction] = target
            used_targets.add(target[0])

    def sector_for(self, pos: Position) -> Direction:
        """Return the cardinal sector containing ``pos`` relative to the core."""
        if self.core_pos is None:
            return Direction.NORTH
        dx = pos.x - self.core_pos.x
        dy = pos.y - self.core_pos.y
        if abs(dy) >= abs(dx):
            return Direction.NORTH if dy < 0 else Direction.SOUTH
        return Direction.EAST if dx > 0 else Direction.WEST

    def core_distance(self, pos: Position) -> int:
        """Return the Chebyshev distance from the core to ``pos``."""
        if self.core_pos is None:
            return 0
        return max(abs(pos.x - self.core_pos.x), abs(pos.y - self.core_pos.y))

    def is_harvester_on_tile(self, pos: Position) -> bool:
        """Report whether the last observation shows a harvester at ``pos``."""
        building = self.known_buildings.get(pos)
        return building is not None and building[0] == EntityType.HARVESTER

    def try_spawn_missing_builder(self, controller: Controller) -> bool:
        """Spawn one missing directional builder, with a safe fallback tile."""
        if self.core_pos is None or controller.get_unit_count() >= GameConstants.MAX_TEAM_UNITS:
            return False
        # This bot's core creates only builder bots.  Entity IDs outside the
        # core's vision cannot be queried reliably, while get_unit_count() is
        # global, so subtracting the core gives the authoritative live fleet.
        living_builders = max(0, controller.get_unit_count() - 1)
        desired_builders = desired_builder_count(
            controller.get_current_round(),
            len(BUILDER_WORK_DIRECTIONS),
            MAX_ECONOMY_BUILDERS,
            ECONOMY_EXPANSION_START_ROUND,
            ECONOMY_EXPANSION_INTERVAL_ROUNDS,
        )
        if living_builders >= desired_builders:
            return False

        if len(self.initial_spawned_directions) < len(BUILDER_WORK_DIRECTIONS):
            directions = [
                direction
                for direction in BUILDER_WORK_DIRECTIONS
                if direction not in self.initial_spawned_directions
            ]
        else:
            if self.replacement_builders_spawned >= MAX_ADDITIONAL_BUILDER_SPAWNS:
                return False
            direction = BUILDER_WORK_DIRECTIONS[
                self.replacement_direction_index % len(BUILDER_WORK_DIRECTIONS)
            ]
            directions = [direction]

        fallback_positions = [
            pos
            for dx, dy in (
                (0, 0), (1, -1), (1, 1), (-1, 1), (-1, -1),
                (0, -1), (1, 0), (0, 1), (-1, 0),
            )
            if (pos := self.tile_cache.offset(self.core_pos, dx, dy)) is not None
        ]
        for direction in directions:
            preferred = self.tile_cache.neighbor(self.core_pos, direction)
            if preferred is None:
                continue
            positions = [preferred]
            if self.spawn_order_pad is not None:
                positions.extend(pos for pos in fallback_positions if pos != preferred)
            for spawn_pos in positions:
                if not controller.can_spawn(spawn_pos):
                    continue
                # The handoff marker is independent from spawning.  It makes a
                # fallback core tile safe: the new bot still receives the
                # original sector rather than deriving it from that tile.
                wrote_order = self.write_spawn_order(controller, direction, spawn_pos)
                # Never let a newborn consume a stale handoff order.  If the
                # dedicated pad is unavailable, wait for it rather than spawn
                # a builder with another sector's command.
                if self.spawn_order_pad is not None and not wrote_order:
                    continue
                if self.spawn_order_pad is None:
                    self.sync_sector_marker(controller, direction)
                controller.spawn_builder(spawn_pos)
                if len(self.initial_spawned_directions) < len(BUILDER_WORK_DIRECTIONS):
                    self.initial_spawned_directions.add(direction)
                else:
                    self.replacement_direction_index = (
                        self.replacement_direction_index + 1
                    ) % len(BUILDER_WORK_DIRECTIONS)
                    self.replacement_builders_spawned += 1
                return True
        return False

    def write_spawn_order(
            self,
            controller: Controller,
            direction: Direction,
            spawn_pos: Position,
    ) -> bool:
        """Write a newborn builder's direction and optional ore target to a marker."""
        if self.spawn_order_pad is None:
            return False
        target = self.sector_targets.get(direction)
        if target is None:
            value = encode_marker(
                MARKER_KIND_SPAWN_DIRECTION,
                spawn_pos,
                BUILDER_DIRECTION_CODES[direction],
            )
        else:
            ore_pos, env = target
            kind = (
                MARKER_KIND_SPAWN_ORE_TI
                if env == Environment.ORE_TITANIUM
                else MARKER_KIND_SPAWN_ORE_AX
            )
            value = encode_marker(kind, ore_pos, BUILDER_DIRECTION_CODES[direction])
        if not controller.can_place_marker(self.spawn_order_pad):
            return False
        marker_id = controller.place_marker(self.spawn_order_pad, value)
        self.tile_cache.remember_building(
            self.spawn_order_pad,
            marker_id,
            EntityType.MARKER,
            self.team,
            marker_value=value,
        )
        self.spawn_order_target = None if target is None else target[0]
        return True

    def clear_completed_spawn_order(self, controller: Controller) -> None:
        """Remove a temporary spawn order once its assigned ore is harvested."""
        if self.spawn_order_pad is None or self.spawn_order_target is None:
            return
        if not self.is_harvester_on_tile(self.spawn_order_target):
            return
        building_id = self.tile_cache.building_id_at(self.spawn_order_pad)
        building = self.tile_cache.building_at(self.spawn_order_pad)
        if building_id is not None and building is not None and building[0] == EntityType.MARKER:
            if building[1] == self.team and controller.can_destroy(self.spawn_order_pad):
                controller.destroy(self.spawn_order_pad)
                self.tile_cache.forget_building(self.spawn_order_pad)
        self.spawn_order_target = None

    def desired_sector_marker_value(self, direction: Direction) -> int | None:
        """Encode the current ore order for one sector, if it has one."""
        target = self.sector_targets.get(direction)
        if target is None:
            return None
        pos, env = target
        kind = (
            MARKER_KIND_SECTOR_ORE_TI
            if env == Environment.ORE_TITANIUM
            else MARKER_KIND_SECTOR_ORE_AX
        )
        return encode_marker(kind, pos, BUILDER_DIRECTION_CODES[direction])

    def sync_one_changed_sector_marker(self, controller: Controller) -> None:
        """Synchronize one changed sector marker within the per-turn action limit."""
        directions = tuple(self.sector_marker_pads)
        if not directions:
            return
        for offset in range(len(directions)):
            index = (self.marker_sync_cursor + offset) % len(directions)
            direction = directions[index]
            if self.sector_marker_values.get(direction) == self.desired_sector_marker_value(direction):
                continue
            self.marker_sync_cursor = (index + 1) % len(directions)
            self.sync_sector_marker(controller, direction)
            return

    def sync_sector_marker(self, controller: Controller, direction: Direction) -> None:
        """Create, update, or remove the persistent marker for one sector."""
        pad = self.sector_marker_pads.get(direction)
        if pad is None:
            return
        desired = self.desired_sector_marker_value(direction)
        current = self.sector_marker_values.get(direction)
        if desired == current:
            return

        if desired is None:
            # Destruction is free and removes a completed mine from the order
            # board instead of leaving a stale target for a replacement bot.
            building_id = self.tile_cache.building_id_at(pad)
            building = self.tile_cache.building_at(pad)
            if building_id is not None and building is not None and building[0] == EntityType.MARKER:
                if building[1] == self.team and controller.can_destroy(pad):
                    controller.destroy(pad)
                    self.tile_cache.forget_building(pad)
            self.sector_marker_values[direction] = None
            return

        if controller.can_place_marker(pad):
            marker_id = controller.place_marker(pad, desired)
            self.tile_cache.remember_building(
                pad,
                marker_id,
                EntityType.MARKER,
                self.team,
                marker_value=desired,
            )
            self.sector_marker_values[direction] = desired
