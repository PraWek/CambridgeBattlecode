from collections import deque
from heapq import heappop, heappush

from cambc import Controller, Direction, EntityType, Environment, GameConstants, Position

from base import BaseBot
from constants import (
    AXIONITE_TITANIUM_THRESHOLD,
    BUILDER_CODE_DIRECTIONS,
    CONNECTION_A_STAR_MAX_EXPANSIONS,
    CONNECTION_DEFER_NEW_TILES,
    CONNECTION_DEFER_MAX_ROUNDS,
    CONNECTION_STALL_ROUNDS,
    BRIDGE_MAX_JUMP_DISTANCE,
    BRIDGE_ROUTE_MAX_EXPANSIONS,
    DIRECTIONS,
    MARKER_KIND_ORE_AX,
    MARKER_KIND_ORE_TI,
    MARKER_KIND_SECTOR_ORE_AX,
    MARKER_KIND_SECTOR_ORE_TI,
    MARKER_KIND_SPAWN_DIRECTION,
    MARKER_KIND_SPAWN_ORE_AX,
    MARKER_KIND_SPAWN_ORE_TI,
    MAX_HARVESTERS_PER_LINE,
    NETWORK_PATROL_GOAL_LIMIT,
    NETWORK_PATROL_MAX_EXPANSIONS,
    ORE_SURVEY_NEW_TILES_REQUIRED,
    ORE_PATH_A_STAR_MAX_EXPANSIONS,
    ORE_TARGET_STALL_ROUNDS,
    ORE_TYPES,
    ORTHOGONAL_DIRECTIONS,
    PASSABLE_BUILDINGS,
    PLANNED_HARVESTERS_PER_LINE,
    RESOURCE_AXIONITE,
    RESOURCE_TITANIUM,
    HARVESTER_GUARD_LATEST_ROUND,
    SCOUT_DISTANCE_WEIGHT,
    SCOUT_CONFIRMED_STALL_KILL_ROUNDS,
    SCOUT_ESCAPE_FAILURE_KILL_ROUNDS,
    SCOUT_DEAD_END_AVOID_ROUNDS,
    SCOUT_DEAD_END_PENALTY,
    SCOUT_FORWARD_PROGRESS_WEIGHT,
    SCOUT_FRONTIER_CANDIDATE_LIMIT,
    SCOUT_INWARD_STEP_PENALTY,
    IDLE_TARGET_RETRY_ROUNDS,
    SCOUT_KILL_CYCLES,
    SCOUT_KILL_ROUTE_FAILURES,
    SCOUT_KILL_STUCK_ROUNDS,
    SCOUT_LATERAL_DEVIATION_WEIGHT,
    SCOUT_NEW_VISION_WEIGHT,
    SCOUT_NO_DISCOVERY_KILL_ROUNDS,
    SCOUT_ORE_HINT_PROGRESS_WEIGHT,
    SCOUT_PATH_GOAL_LIMIT,
    SCOUT_PATH_MAX_EXPANSIONS,
    SCOUT_PERSISTENT_REVISIT_PENALTY,
    SCOUT_REPLAN_STUCK_ROUNDS,
    SCOUT_RETURN_TO_BASE_WEIGHT,
    SCOUT_REVISIT_STEP_PENALTY,
    SCOUT_ROUTE_MEMORY_TILES,
    SCOUT_SECTOR_BONUS,
    STEINER_MAX_EXPANSIONS,
    TRANSPORT_BUSY_OBSERVATION_TURNS,
    TRANSPORT_BUSY_SAMPLE_LIMIT,
    YIELD_ROUTE_AVOID_ROUNDS,
)
from geometry import decode_marker
from exploration import (
    choose_information_gain_step,
    is_static_step_obstacle,
    should_recycle_stalled_builder,
)
from navigation import a_star_to_any
from network_memory import NetworkMemory
from network_planner import (
    dedicated_route_tree,
    sector_entry_offsets,
    sector_receiver_offsets,
    starts_new_line,
)
from orders import spawn_order_for
from steiner import incremental_steiner_branch


_VISION_RADIUS = int(GameConstants.BUILDER_BOT_VISION_RADIUS_SQ ** 0.5)
SCOUT_VISION_OFFSETS = tuple(
    (dx, dy)
    for dx in range(-_VISION_RADIUS, _VISION_RADIUS + 1)
    for dy in range(-_VISION_RADIUS, _VISION_RADIUS + 1)
    if dx * dx + dy * dy <= GameConstants.BUILDER_BOT_VISION_RADIUS_SQ
)


class BuilderBot(BaseBot):
    def __init__(self, map_width: int, map_height: int) -> None:
        """Initialize exploration memory, mining state, and conveyor planning state."""
        super().__init__(map_width, map_height)
        self.core_pos: Position | None = None
        self.enemy_estimate: Position | None = None
        self.work_direction: Direction | None = None
        self.spawn_direction: Direction | None = None

        # These aliases are indexes maintained by BaseBot.tile_cache, never
        # separate map copies.  All planning below is therefore a dictionary
        # lookup instead of another Controller tile query.
        self.known_env = self.tile_cache.environments
        self.known_buildings = self.tile_cache.buildings
        self.known_conveyor_directions = self.tile_cache.conveyor_directions
        self.known_bridge_targets: dict[Position, Position] = {}
        self.known_bridge_ids: dict[Position, int] = {}
        self.connected_network_cache: set[Position] | None = None
        self.observed_network_state: dict[
            Position,
            tuple[EntityType, Direction | None, Position | None] | None,
        ] = {}
        self.network_memory = NetworkMemory(TRANSPORT_BUSY_OBSERVATION_TURNS)
        # Compatibility aliases keep the routing code terse while the memory,
        # damage detection, and saturation history live in their own module.
        self.transport_busy_history = self.network_memory.busy_history
        self.network_blueprint = self.network_memory.blueprint
        self.damaged_network_tiles = self.network_memory.damaged_tiles
        # Ownership is the distributed capacity contract.  Bots cannot share
        # Python counters, so a builder may merge only into transport it laid
        # itself; other allied lanes are crossed, never silently overloaded.
        self.owned_network_tiles: set[Position] = set()
        self.active_line_tiles: set[Position] = set()
        self.harvesters_built_count = 0
        self.connection_starts_new_line = False
        self.last_plan_starts_new_line = False
        self.observed_tiles = self.tile_cache.observed_tiles
        self.inferred_ores: dict[Position, Environment] = {}
        self.reported_ores: dict[Position, Environment] = {}
        self.assigned_ores: dict[Position, Environment] = {}
        self.scout_frontier: set[Position] = set()
        self.scout_frontier_initialized = False
        self.unreachable_scout_targets: set[Position] = set()
        self.scout_retry_pending = False
        self.permanently_blocked: set[Position] = set()
        self.yield_blocked_until: dict[Position, int] = {}
        self.current_round = 0

        self.target_ore: Position | None = None
        self.target_resource = RESOURCE_TITANIUM
        self.target_is_connection = False
        self.scout_target: Position | None = None
        self.scout_target_direct = False
        self.path: list[Position] = []
        self.path_index = 0
        self.conveyor_path_tiles: set[Position] = set()
        self.conveyor_directions: dict[Position, Direction] = {}
        self.bridge_targets: dict[Position, Position] = {}
        self.connection_anchor: Position | None = None
        self.dedicated_line_committed = False
        # Tiles laid by this builder for its current, not-yet-connected mine.
        # They may be safely reused after a newly discovered obstacle forces a
        # replan; conveyors made by any other builder remain immutable.
        self.unfinished_branch_tiles: set[Position] = set()
        self.pending_network_ores: set[Position] = set()
        self.harvester_built = False
        self.harvester_fail_count = 0
        self.skipped_ores: set[Position] = set()
        self.deferred_ores_until: dict[Position, int] = {}
        self.deferred_connections_observed_count: dict[Position, int] = {}
        self.deferred_connections_until_round: dict[Position, int] = {}
        self.harvester_guard_tiles: set[Position] = set()
        self.ore_retry_observed_count = 0
        self.next_select_round = 0
        self.replan_after_yield = False
        self.mode = "scout"
        self.titanium_unlocked = False
        self.scout_heading: Direction | None = None
        self.scout_sweep_direction: Direction | None = None

        self.last_pos: Position | None = None
        self.stuck_rounds = 0
        self.rounds_alive = 0
        self.last_progress_round = 0
        self.recent_route: deque[Position] = deque()
        self.recent_route_visits: dict[Position, int] = {}
        self.scout_total_visits: dict[Position, int] = {}
        self.scout_avoid_until: dict[Position, int] = {}
        self.scout_cycle_replan = False
        self.scout_cycle_count = 0
        self.scout_route_failures = 0
        self.scout_rounds_without_discovery = 0
        self.idle_escape_failures = 0

    def run(self, controller: Controller) -> None:
        """Execute one turn of scouting, mining, or conveyor construction."""
        if self._scan_turn(controller, read_markers=True):
            return
        self.rounds_alive += 1
        self.current_round = controller.get_current_round()
        current = self.get_cached_position()
        if self.last_pos is not None and current == self.last_pos:
            self.stuck_rounds += 1
            if self.target_ore is None and self.scout_target is None:
                # Controller.can_move() can accept a command which loses a
                # simultaneous collision.  Count confirmed positions on the
                # following turn instead of treating the issued command as
                # progress.
                self.idle_escape_failures += 1
            else:
                self.idle_escape_failures = 0
        else:
            self.stuck_rounds = 0
            self.idle_escape_failures = 0
            self.last_progress_round = self.rounds_alive
            self.remember_route_position(current)
        self.last_pos = current

        if self.tile_cache.newly_observed_tiles:
            self.scout_rounds_without_discovery = 0
            self.scout_cycle_count = 0
            self.scout_route_failures = 0
        else:
            self.scout_rounds_without_discovery += 1

        # ``can_move`` describes whether a command may be submitted, not
        # whether it survives simultaneous movement resolution.  Use the
        # position observed on the following turn, together with the last
        # successful build/repair/attack, to recycle a genuinely wedged bot.
        if should_recycle_stalled_builder(
            self.stuck_rounds,
            self.rounds_alive - self.last_progress_round,
            SCOUT_CONFIRMED_STALL_KILL_ROUNDS,
        ):
            controller.self_destruct()
            return

        # A temporary collision or a newly revealed wall should trigger a
        # cheap replan, not kill a useful builder.  Cyclic movement is detected
        # separately because it resets the ordinary "same position" counter.
        if (
            self.target_ore is None
            and self.scout_target is not None
            and (self.scout_cycle_replan or self.stuck_rounds >= SCOUT_REPLAN_STUCK_ROUNDS)
        ):
            self.cancel_scout_route()

        # Never recycle a builder merely because it failed to retain a target.
        # A TLE also looks like inactivity, and the old 32-round rule therefore
        # created an endless spawn/TLE/self-destruct loop on game 4's map.
        # Genuine exhaustion still has the much longer guarded condition below.
        if (
            self.target_ore is None
            and not self.owned_network_tiles
            and self.scout_rounds_without_discovery >= SCOUT_NO_DISCOVERY_KILL_ROUNDS
            and (
                self.stuck_rounds >= SCOUT_KILL_STUCK_ROUNDS
                or self.scout_route_failures >= SCOUT_KILL_ROUTE_FAILURES
                or self.scout_cycle_count >= SCOUT_KILL_CYCLES
            )
        ):
            controller.self_destruct()
            return

        self.observe_tiles(controller)
        if self.core_pos is None:
            self.core_pos = self.find_home_core()
        if self.core_pos is None:
            return
        if self.enemy_estimate is None:
            self.enemy_estimate = self.tile_cache.position_at(
                self.map_width - 1 - self.core_pos.x,
                self.map_height - 1 - self.core_pos.y,
            )
        if self.work_direction is None:
            self.spawn_direction = self.core_pos.direction_to(current)
        self.read_ore_markers()
        if self.work_direction is None:
            self.work_direction = self.spawn_direction
            if self.work_direction not in ORTHOGONAL_DIRECTIONS:
                self.work_direction = Direction.NORTH
        if self.scout_heading is None:
            self.scout_heading = self.work_direction
            self.scout_sweep_direction = self.work_direction.rotate_right().rotate_right()

        self.titanium_unlocked = controller.get_global_resources()[0] > AXIONITE_TITANIUM_THRESHOLD
        if self.try_repair_nearby_network(controller, current):
            return
        if (
            self.target_is_connection
            and self.target_ore is not None
            and self.rounds_alive - self.last_progress_round >= CONNECTION_STALL_ROUNDS
        ):
            ore = self.target_ore
            self.defer_connection_for_survey(ore)
            self.clear_ore_target()
            self.replan_after_yield = False
        elif (
            self.target_ore is not None
            and not self.target_is_connection
            and self.rounds_alive - self.last_progress_round >= ORE_TARGET_STALL_ROUNDS
        ):
            ore = self.target_ore
            self.defer_ore_for_survey(ore)
            self.clear_ore_target()
            self.replan_after_yield = False
        if not self.target_is_connection and self.try_build_nearby_harvester(controller, current):
            return
        if self.replan_after_yield:
            self.replan_after_yield = False
            if not self.replan_active_ore_target(controller):
                self.select_new_target(controller)
        else:
            self.maybe_select_new_target(controller)
        if self.target_ore is None and self.scout_target is None:
            emergency_step = self.choose_right_hand_scout_step(current)
            if emergency_step is not None:
                self.try_move_scout_step(
                    controller,
                    current,
                    current.direction_to(emergency_step),
                )
            if self.idle_escape_failures >= SCOUT_ESCAPE_FAILURE_KILL_ROUNDS:
                controller.self_destruct()
            return

        if (
            self.target_ore is not None
            and not self.target_is_connection
            and current.distance_squared(self.target_ore) <= GameConstants.ACTION_RADIUS_SQ
        ):
            if self.is_harvester_on_tile(self.target_ore):
                if self.ore_network_needs_work(self.target_ore):
                    self.start_ore_connection(controller, current, self.target_ore)
                else:
                    self.clear_ore_target()
                    self.select_new_target(controller)
                return
            if controller.can_build_harvester(self.target_ore):
                # Do not leave an isolated harvester behind.  Reserve a valid
                # branch to the core-connected conveyor tree *before* placing
                # the harvester, while the ore approach is still known to be
                # reachable.  The next turns build this exact branch.
                if not self.assign_connection_target(controller, current, self.target_ore):
                    self.defer_ore_for_survey(self.target_ore)
                    self.clear_ore_target()
                    self.select_new_target(controller)
                    return
                harvester_id = controller.build_harvester(self.target_ore)
                self.record_harvester_built(self.target_ore, harvester_id)
                return

            self.harvester_fail_count += 1
            if self.harvester_fail_count >= 5:
                self.skipped_ores.add(self.target_ore)
                self.clear_ore_target()
                self.harvester_fail_count = 0
                self.select_new_target(controller)
                return

        if self.try_build_harvester_guard(controller, current):
            return
        self.follow_path_and_build(controller)

    def maybe_select_new_target(self, controller: Controller) -> None:
        """Choose a fresh job when the current non-connection job is complete or absent."""
        if self.target_is_connection:
            return
        current_round = controller.get_current_round()
        need_target = (
            self.harvester_built
            or self.target_ore is not None and self.is_harvester_on_tile(self.target_ore)
            or self.target_ore is None and self.scout_target is None and current_round >= self.next_select_round
            or self.target_ore is None and self.scout_target is not None and self.mineable_ores()
        )
        if not need_target:
            return
        self.select_new_target(controller)
        if self.target_ore is None and self.scout_target is None:
            self.next_select_round = current_round + (
                1 if self.scout_retry_pending else IDLE_TARGET_RETRY_ROUNDS
            )

    def observe_tiles(self, controller: Controller) -> None:
        """Refresh known terrain, buildings, conveyor directions, and scout frontier."""
        network_changed = False
        busy_candidates: list[tuple[int, Position, int]] = []
        current = self.get_cached_position()
        for pos in self.tile_cache.visible_tiles:
            self.inferred_ores.pop(pos, None)
            self.reported_ores.pop(pos, None)
            building = self.known_buildings.get(pos)
            building_id = self.tile_cache.building_id_at(pos)
            if building is not None and building[0] == EntityType.BRIDGE:
                if building_id is not None and self.known_bridge_ids.get(pos) != building_id:
                    self.known_bridge_targets[pos] = controller.get_bridge_target(building_id)
                    self.known_bridge_ids[pos] = building_id
            else:
                self.known_bridge_targets.pop(pos, None)
                self.known_bridge_ids.pop(pos, None)

            transport_state = None
            if (
                building is not None
                and building[1] == self.team
                and building[0] in {
                    EntityType.CONVEYOR,
                    EntityType.ARMOURED_CONVEYOR,
                    EntityType.BRIDGE,
                }
            ):
                transport_state = (
                    building[0],
                    self.known_conveyor_directions.get(pos),
                    self.known_bridge_targets.get(pos),
                )
            previous_state = self.observed_network_state.get(pos)
            if previous_state != transport_state and (
                previous_state is not None or transport_state is not None
            ):
                network_changed = True
            self.observed_network_state[pos] = transport_state

            self.network_memory.audit(
                pos,
                None if building is None else building[0],
                building is not None and building[1] == self.team,
                self.known_conveyor_directions.get(pos),
                self.known_bridge_targets.get(pos),
            )

            if (
                building is not None
                and building[1] == self.team
                and building[0] in {
                    EntityType.CONVEYOR,
                    EntityType.ARMOURED_CONVEYOR,
                    EntityType.BRIDGE,
                }
                and building_id is not None
            ):
                if pos not in self.network_blueprint:
                    self.network_memory.remember(
                        pos,
                        building[0],
                        self.known_conveyor_directions.get(pos),
                        self.known_bridge_targets.get(pos),
                    )
                # Stored-resource reads are one of the expensive controller
                # calls.  Structural source counts handle normal capacity;
                # sample only the nearest transport to spot a persistently
                # full merge without inspecting the entire visible network.
                busy_candidates.append((current.distance_squared(pos), pos, building_id))
        busy_candidates.sort(key=lambda item: (item[0], item[1].x, item[1].y))
        for _, pos, building_id in busy_candidates[:TRANSPORT_BUSY_SAMPLE_LIMIT]:
            self.network_memory.record_busy(
                pos,
                controller.get_stored_resource(building_id) is not None,
            )
        if network_changed:
            self.connected_network_cache = None
        for pos in self.tile_cache.newly_observed_tiles:
            env = self.known_env[pos]
            self.update_scout_frontier(pos)
            if env in ORE_TYPES:
                self.infer_symmetric(pos, env)

    def try_repair_nearby_network(self, controller: Controller, current: Position) -> bool:
        """Restore a visible broken transport tile before extending the network.

        Empty or friendly-road damage is repaired immediately.  Hostile
        conveyors are deliberately not fought for ten turns here: the
        connection planner treats them as blocks and may bridge around them.
        """
        candidates = sorted(
            (
                pos for pos in self.damaged_network_tiles
                if current.distance_squared(pos) <= GameConstants.ACTION_RADIUS_SQ
            ),
            key=lambda pos: (self.core_distance(pos), pos.x, pos.y),
        )
        for pos in candidates:
            expected = self.network_blueprint.get(pos)
            if expected is None:
                self.damaged_network_tiles.discard(pos)
                continue
            building = self.known_buildings.get(pos)
            if building is not None:
                if (
                    building[0] == EntityType.ROAD
                    and building[1] == self.team
                    and controller.can_destroy(pos)
                ):
                    controller.destroy(pos)
                    self.tile_cache.forget_building(pos)
                    self.connected_network_cache = None
                    return True
                # An enemy rewrite or incompatible allied transport is fed
                # back to the capacity planner as an obstacle; a detour/bridge
                # is cheaper and faster than repeatedly firing at it.
                continue

            entity_type, direction, bridge_target = expected
            if entity_type == EntityType.BRIDGE and bridge_target is not None:
                if not controller.can_build_bridge(pos, bridge_target):
                    continue
                entity_id = controller.build_bridge(pos, bridge_target)
                self.tile_cache.remember_building(
                    pos,
                    entity_id,
                    EntityType.BRIDGE,
                    self.team,
                )
                self.known_bridge_targets[pos] = bridge_target
                self.known_bridge_ids[pos] = entity_id
            elif (
                entity_type in {EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR}
                and direction is not None
                and controller.can_build_conveyor(pos, direction)
            ):
                entity_id = controller.build_conveyor(pos, direction)
                self.tile_cache.remember_building(
                    pos,
                    entity_id,
                    EntityType.CONVEYOR,
                    self.team,
                    direction=direction,
                )
                # Armoured tiles are restored as ordinary conveyors when
                # titanium is available but axionite is not; continuity wins.
                self.network_memory.remember(pos, EntityType.CONVEYOR, direction)
            else:
                continue
            self.damaged_network_tiles.discard(pos)
            self.connected_network_cache = None
            self.last_progress_round = self.rounds_alive
            return True
        return False

    def read_ore_markers(self) -> None:
        """Read core orders and shared ore hints from nearby marker buildings."""
        records: list[tuple[int, Position, int]] = []
        for entity_id in self.tile_cache.marker_ids():
            marker_value = self.tile_cache.marker_values.get(entity_id)
            if marker_value is None:
                continue
            try:
                kind, pos, payload = decode_marker(
                    marker_value,
                    self.tile_cache.position_at,
                )
            except Exception:
                continue
            records.append((kind, pos, payload))

        # A newly created builder has no durable memory yet.  The core writes
        # one handoff marker in the same round as spawn, which is necessary
        # when a map edge forces it to use a non-cardinal core tile.
        if self.work_direction is None:
            current = self.get_cached_position()
            order = spawn_order_for(
                records,
                current,
                {
                    MARKER_KIND_SPAWN_DIRECTION,
                    MARKER_KIND_SPAWN_ORE_TI,
                    MARKER_KIND_SPAWN_ORE_AX,
                },
                BUILDER_CODE_DIRECTIONS,
            )
            if order is not None:
                kind, pos, direction = order
                self.work_direction = direction
                if kind == MARKER_KIND_SPAWN_ORE_TI:
                    self.assigned_ores[pos] = Environment.ORE_TITANIUM
                elif kind == MARKER_KIND_SPAWN_ORE_AX:
                    self.assigned_ores[pos] = Environment.ORE_AXIONITE

        if self.work_direction is None and self.spawn_direction in ORTHOGONAL_DIRECTIONS:
            self.work_direction = self.spawn_direction

        for kind, pos, payload in records:
            if kind == MARKER_KIND_ORE_TI and pos not in self.observed_tiles:
                self.reported_ores[pos] = Environment.ORE_TITANIUM
            elif kind == MARKER_KIND_ORE_AX and pos not in self.observed_tiles:
                self.reported_ores[pos] = Environment.ORE_AXIONITE
            elif (
                kind in {MARKER_KIND_SECTOR_ORE_TI, MARKER_KIND_SECTOR_ORE_AX}
                and self.work_direction is not None
                and BUILDER_CODE_DIRECTIONS.get(payload) == self.work_direction
            ):
                # Unlike a symmetry hint, a sector order comes from a core
                # that has directly observed the ore.  It is therefore a real
                # mining target even before this builder sees the tile itself.
                self.assigned_ores[pos] = (
                    Environment.ORE_TITANIUM
                    if kind == MARKER_KIND_SECTOR_ORE_TI
                    else Environment.ORE_AXIONITE
                )

    def infer_symmetric(self, pos: Position, env: Environment) -> None:
        """Record an ore hint only after the cache proves map symmetry."""
        mirror = self.tile_cache.mirrored_position(pos)
        if mirror is not None and mirror not in self.observed_tiles:
            self.inferred_ores[mirror] = env

    def update_scout_frontier(self, known_pos: Position) -> None:
        """Update unknown boundary cells adjacent to a newly known tile."""
        self.scout_frontier_initialized = True
        self.scout_frontier.discard(known_pos)
        # A wall or ore tile cannot be the known side of an exploration edge.
        # Adding all of its unknown neighbours creates phantom frontiers behind
        # solid barriers, which repeatedly win the sector score but have no
        # legal route from the explored component.
        env = self.known_env.get(known_pos)
        building = self.known_buildings.get(known_pos)
        if (
            env == Environment.WALL
            or env in ORE_TYPES
            or (building is not None and building[0] not in PASSABLE_BUILDINGS)
            or (
                building is not None
                and building[0] == EntityType.CORE
                and building[1] != self.team
            )
        ):
            return
        for direction in DIRECTIONS:
            probe = self.tile_cache.neighbor(known_pos, direction)
            if probe is None:
                continue
            if probe in self.observed_tiles:
                self.scout_frontier.discard(probe)
            else:
                self.scout_frontier.add(probe)

    def rebuild_scout_frontier(self) -> None:
        """Reconstruct the exploration frontier from all observed tiles."""
        for known_pos in self.observed_tiles:
            self.update_scout_frontier(known_pos)

    def find_home_core(self) -> Position | None:
        """Locate the friendly core while it is within local vision."""
        for entity_id in self.tile_cache.visible_entity_ids:
            if (
                self.tile_cache.entity_type(entity_id) == EntityType.CORE
                and self.tile_cache.entity_team(entity_id) == self.team
            ):
                return self.tile_cache.entity_position(entity_id)
        return None

    def known_titanium_ores(self) -> list[Position]:
        """Return all directly observed titanium deposits."""
        return [pos for pos, env in self.known_env.items() if env == Environment.ORE_TITANIUM]

    def known_axionite_ores(self) -> list[Position]:
        """Return all directly observed axionite deposits."""
        return [pos for pos, env in self.known_env.items() if env == Environment.ORE_AXIONITE]

    def resource_at(self, pos: Position) -> Environment | None:
        """Return the observed or core-assigned ore type at ``pos``."""
        return self.known_env.get(pos) or self.assigned_ores.get(pos)

    def is_ore_eligible(self, ore: Position) -> bool:
        """Return whether this ore type is currently allowed to be mined."""
        resource = self.resource_at(ore)
        return resource == Environment.ORE_TITANIUM or (
            resource == Environment.ORE_AXIONITE and self.titanium_unlocked
        )

    def mineable_ores(self) -> list[Position]:
        """Return eligible, unharvested, non-deferred local and assigned ores."""
        known = self.known_titanium_ores()
        if self.titanium_unlocked:
            known += self.known_axionite_ores()
        known += [ore for ore in self.assigned_ores if ore not in self.known_env]
        return [
            ore
            for ore in known
            if (
                self.is_ore_eligible(ore)
                and not self.is_harvester_on_tile(ore)
                and not self.ore_is_deferred(ore)
            )
        ]

    def observed_ores(self) -> list[Position]:
        """Return eligible ore deposits that this builder has directly seen."""
        return [
            pos
            for pos, env in self.known_env.items()
            if pos in self.observed_tiles and env in ORE_TYPES and self.is_ore_eligible(pos)
        ]

    def is_harvester_on_tile(self, pos: Position) -> bool:
        """Report whether the last observation shows a harvester at ``pos``."""
        building = self.known_buildings.get(pos)
        return building is not None and building[0] == EntityType.HARVESTER

    def try_build_nearby_harvester(self, controller: Controller, current: Position) -> bool:
        """Preflight and build a harvester on the best reachable adjacent ore."""
        candidates = [
            ore
            for ore in self.observed_ores()
            if (
                not self.is_harvester_on_tile(ore)
                and current.distance_squared(ore) <= GameConstants.ACTION_RADIUS_SQ
            )
        ]
        candidates.sort(
            key=lambda ore: (
                0 if ore == self.target_ore and not self.target_is_connection else 1,
                self.work_direction_priority(ore),
                current.distance_squared(ore),
            )
        )
        for ore in candidates:
            if self.ore_is_deferred(ore):
                continue
            if not controller.can_build_harvester(ore):
                continue
            # A harvester is useful only with an already planned connection to
            # the core-connected tree.  Planning first also prevents a bot
            # from switching back to exploration immediately after the build.
            if not self.assign_connection_target(controller, current, ore):
                self.defer_ore_for_survey(ore)
                if ore == self.target_ore:
                    self.clear_ore_target()
                continue
            harvester_id = controller.build_harvester(ore)
            self.record_harvester_built(ore, harvester_id)
            return True
        return False

    def record_harvester_built(self, ore: Position, harvester_id: int) -> None:
        """Update local state after successfully placing a harvester on ``ore``."""
        self.tile_cache.remember_building(ore, harvester_id, EntityType.HARVESTER, self.team)
        self.harvester_built = True
        self.last_progress_round = self.rounds_alive
        self.harvester_fail_count = 0
        self.skipped_ores.discard(ore)
        self.assigned_ores.pop(ore, None)
        self.reported_ores.pop(ore, None)
        self.deferred_ores_until.pop(ore, None)
        self.pending_network_ores.add(ore)
        if self.connection_starts_new_line:
            self.active_line_tiles.clear()
        self.harvesters_built_count += 1
        if self.current_round <= HARVESTER_GUARD_LATEST_ROUND:
            branch_receivers = {
                receiver
                for direction in ORTHOGONAL_DIRECTIONS
                if (receiver := self.tile_cache.neighbor(ore, direction)) is not None
                and receiver in self.conveyor_path_tiles
            }
            self.harvester_guard_tiles = {
                guard
                for direction in ORTHOGONAL_DIRECTIONS
                if (guard := self.tile_cache.neighbor(ore, direction)) is not None
                if (
                    guard not in branch_receivers
                    and self.traversable_for_connection(guard)
                )
            }

    def try_build_harvester_guard(self, controller: Controller, current: Position) -> bool:
        """Seal unused outputs of a new harvester with rejecting conveyors."""
        ore = self.target_ore
        if not self.target_is_connection or ore is None or not self.harvester_guard_tiles:
            return False
        # First give the harvester one real receiver.  Its initial production
        # can then enter the branch while the remaining sides are protected.
        has_live_receiver = any(
            self.has_expected_tree_conveyor(controller, receiver)
            for direction in ORTHOGONAL_DIRECTIONS
            if (receiver := self.tile_cache.neighbor(ore, direction)) is not None
            and receiver in self.conveyor_path_tiles
        )
        if not has_live_receiver:
            return False

        ordered = sorted(
            self.harvester_guard_tiles,
            key=lambda pos: (current.distance_squared(pos), pos.x, pos.y),
        )
        for guard in ordered:
            if current.distance_squared(guard) > GameConstants.ACTION_RADIUS_SQ:
                self.harvester_guard_tiles.discard(guard)
                continue
            building = self.known_buildings.get(guard)
            if building is not None:
                if (
                    building[0] == EntityType.ROAD
                    and building[1] == self.team
                    and controller.can_destroy(guard)
                ):
                    controller.destroy(guard)
                    self.tile_cache.forget_building(guard)
                else:
                    self.harvester_guard_tiles.discard(guard)
                    continue
            direction = guard.direction_to(ore)
            if direction not in ORTHOGONAL_DIRECTIONS:
                self.harvester_guard_tiles.discard(guard)
                continue
            if not controller.can_build_conveyor(guard, direction):
                continue
            conveyor_id = controller.build_conveyor(guard, direction)
            self.tile_cache.remember_building(
                guard,
                conveyor_id,
                EntityType.CONVEYOR,
                self.team,
                direction=direction,
            )
            self.network_blueprint[guard] = (EntityType.CONVEYOR, direction, None)
            self.harvester_guard_tiles.discard(guard)
            self.last_progress_round = self.rounds_alive
            return True
        return False

    def defer_ore_for_survey(self, ore: Position) -> None:
        """Temporarily resume exploration until a fully known branch exists."""
        self.deferred_ores_until[ore] = self.current_round + 12
        self.ore_retry_observed_count = max(
            self.ore_retry_observed_count,
            len(self.observed_tiles) + ORE_SURVEY_NEW_TILES_REQUIRED,
        )

    def defer_connection_for_survey(self, ore: Position) -> None:
        """Suspend an impossible branch until exploration reveals a new route."""
        self.pending_network_ores.add(ore)
        self.deferred_connections_observed_count[ore] = max(
            self.deferred_connections_observed_count.get(ore, 0),
            len(self.observed_tiles) + CONNECTION_DEFER_NEW_TILES,
        )
        self.deferred_connections_until_round[ore] = max(
            self.deferred_connections_until_round.get(ore, 0),
            self.current_round + CONNECTION_DEFER_MAX_ROUNDS,
        )

    def connection_is_deferred(self, ore: Position) -> bool:
        """Return whether retrying this disconnected harvester would repeat known work."""
        required_observations = self.deferred_connections_observed_count.get(ore)
        if required_observations is None:
            return False
        if (
            len(self.observed_tiles) >= required_observations
            or self.current_round >= self.deferred_connections_until_round.get(ore, 0)
        ):
            self.deferred_connections_observed_count.pop(ore, None)
            self.deferred_connections_until_round.pop(ore, None)
            return False
        return True

    def ore_is_deferred(self, ore: Position) -> bool:
        """Return whether ore processing is temporarily postponed for scouting."""
        return self.deferred_ores_until.get(ore, -1) >= self.current_round

    def clear_ore_target(self) -> None:
        """Reset the current ore job and its associated path and branch state."""
        self.target_ore = None
        self.target_is_connection = False
        self.path = []
        self.path_index = 0
        self.conveyor_path_tiles = set()
        self.conveyor_directions = {}
        self.bridge_targets = {}
        self.harvester_guard_tiles.clear()
        self.connection_anchor = None
        self.mode = "scout"

    def assign_ore_target(
            self,
            controller: Controller,
            current: Position,
            ore: Position,
            connecting: bool,
    ) -> bool:
        """Plan movement to an ore deposit or its preflighted conveyor branch."""
        if connecting:
            return self.assign_connection_target(controller, current, ore)

        approaches = self.ore_action_approaches(ore)
        if not approaches:
            return False
        path = a_star_to_any(
            controller,
            current,
            set(approaches),
            self.traversable_for_ore_path,
            self.tile_cache.neighbor,
            movement_directions=DIRECTIONS,
            max_expansions=ORE_PATH_A_STAR_MAX_EXPANSIONS,
        )
        if not path and current not in approaches:
            return False
        self.target_ore = ore
        self.target_resource = (
            RESOURCE_TITANIUM
            if self.resource_at(ore) == Environment.ORE_TITANIUM
            else RESOURCE_AXIONITE
        )
        self.target_is_connection = False
        self.scout_target = None
        self.path = path
        self.path_index = 0
        self.conveyor_path_tiles = set()
        self.conveyor_directions = {}
        self.bridge_targets = {}
        self.connection_anchor = None
        self.mode = "ore"
        return True

    def start_ore_connection(self, controller: Controller, current: Position, ore: Position) -> None:
        """Start or retain responsibility for connecting a harvested ore to the tree."""
        self.pending_network_ores.add(ore)
        if not self.assign_connection_target(controller, current, ore):
            # Keeping an unrouteable connection as the active target traps a
            # builder in a survey loop.  Preserve the pending harvester, but
            # resume ordinary exploration until the local map has changed.
            self.defer_connection_for_survey(ore)
            self.clear_ore_target()

    def replan_active_ore_target(self, controller: Controller) -> bool:
        """Recompute the path for the active ore job after a route becomes invalid."""
        if self.target_ore is None:
            return False
        if self.release_connected_ore_target():
            return False
        ore = self.target_ore
        connecting = self.target_is_connection
        if self.assign_ore_target(controller, self.get_cached_position(), ore, connecting):
            return True
        if connecting:
            self.defer_connection_for_survey(ore)
            self.clear_ore_target()
            return False
        self.defer_ore_for_survey(ore)
        self.clear_ore_target()
        return False

    def release_connected_ore_target(self) -> bool:
        """Clear a connection job that another builder has already completed."""
        if (
            not self.target_is_connection
            or self.target_ore is None
            or not self.harvester_is_connected(self.target_ore)
        ):
            return False
        self.pending_network_ores.discard(self.target_ore)
        self.deferred_connections_observed_count.pop(self.target_ore, None)
        self.deferred_connections_until_round.pop(self.target_ore, None)
        if self.connection_anchor is not None and self.is_core_receiver_tile(self.connection_anchor):
            self.dedicated_line_committed = True
        self.unfinished_branch_tiles.clear()
        self.clear_ore_target()
        return True

    def ore_network_needs_work(self, ore: Position) -> bool:
        """Return whether the harvester on ``ore`` lacks a path to the core."""
        return not self.harvester_is_connected(ore)

    def connection_candidates(self) -> set[Position]:
        """Return harvested ores that still need to join the conveyor tree."""
        candidates = set(self.pending_network_ores)
        for ore in self.observed_ores():
            if self.is_harvester_on_tile(ore) and self.ore_network_needs_work(ore):
                candidates.add(ore)
        return {
            ore
            for ore in candidates
            if (
                self.is_harvester_on_tile(ore)
                and self.ore_network_needs_work(ore)
                and not self.connection_is_deferred(ore)
            )
        }

    def ore_action_approaches(self, ore: Position) -> list[Position]:
        """Return movement tiles from which a builder can act on an ore deposit."""
        return [
            approach
            for direction in DIRECTIONS
            if (approach := self.tile_cache.neighbor(ore, direction)) is not None
            and self.traversable_for_ore_path(None, approach)
        ]

    def core_receiver_tiles(self) -> set[Position]:
        """Return in-bounds tiles that deliver resources directly into the core."""
        if self.core_pos is None:
            return set()
        return {
            pos
            for dx in range(-1, 2)
            for dy in range(-1, 2)
            if (pos := self.tile_cache.offset(self.core_pos, dx, dy)) is not None
        }

    def sector_core_receiver_tiles(self) -> set[Position]:
        """Reserve one Core edge for this builder's distributed lane family."""
        if self.core_pos is None or self.work_direction is None:
            return self.core_receiver_tiles()
        offsets = sector_receiver_offsets(self.work_direction)
        return {
            pos
            for dx, dy in offsets
            if (pos := self.tile_cache.offset(self.core_pos, dx, dy)) is not None
        }

    def sector_core_entry_tiles(self) -> set[Position]:
        """Return the three non-overlapping outer intake tiles for this sector."""
        if self.core_pos is None or self.work_direction is None:
            return set()
        offsets = sector_entry_offsets(self.work_direction)
        return {
            pos
            for dx, dy in offsets
            if (pos := self.tile_cache.offset(self.core_pos, dx, dy)) is not None
        }

    def core_entry_tiles(self) -> set[Position]:
        """Return every cardinal perimeter tile that can feed the 3x3 core."""
        if self.core_pos is None:
            return set()
        entries: set[Position] = set()
        for direction in ORTHOGONAL_DIRECTIONS:
            for dx, dy in sector_entry_offsets(direction):
                pos = self.tile_cache.offset(self.core_pos, dx, dy)
                if pos is not None:
                    entries.add(pos)
        return entries

    def network_plan_receiver_accepts(
            self,
            receiver: Position,
            source: Position,
            allowed_core_entries: set[Position] | None = None,
    ) -> bool:
        """Apply the sector intake reservation while planning new transport."""
        if self.is_core_receiver_tile(receiver):
            entries = (
                self.sector_core_entry_tiles()
                if allowed_core_entries is None
                else allowed_core_entries
            )
            return source in entries
        return self.network_receiver_accepts(receiver, source)

    def is_core_receiver_tile(self, pos: Position) -> bool:
        """Return whether a conveyor at ``pos`` can send resources into the core."""
        return pos in self.core_receiver_tiles()

    def known_connected_network(self) -> set[Position]:
        """Return the part of the observed conveyor graph that truly reaches core."""
        if self.connected_network_cache is not None:
            return self.connected_network_cache

        connected = self.core_receiver_tiles()
        # Reverse edges let us traverse the directed conveyor graph from the
        # core in linear time.  The earlier fixed-point scan became quadratic
        # on a long line and could consume the per-turn time budget.
        incoming: dict[Position, list[Position]] = {}
        for pos, building in self.known_buildings.items():
            if building is None or building[1] != self.team:
                continue
            if building[0] in {EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR}:
                direction = self.known_conveyor_directions.get(pos)
                if direction is not None:
                    receiver = self.tile_cache.neighbor(pos, direction)
                    if receiver is not None:
                        incoming.setdefault(receiver, []).append(pos)
            elif building[0] == EntityType.BRIDGE:
                target = self.known_bridge_targets.get(pos)
                if target is not None:
                    incoming.setdefault(target, []).append(pos)

        queue = deque(connected)
        while queue:
            receiver = queue.popleft()
            for source in incoming.get(receiver, []):
                if source in connected:
                    continue
                connected.add(source)
                queue.append(source)
        self.connected_network_cache = connected
        return connected

    def network_receiver_accepts(self, receiver: Position, source: Position) -> bool:
        """Return whether a connected receiver can accept flow from ``source``."""
        if self.is_core_receiver_tile(receiver):
            return True
        building = self.known_buildings.get(receiver)
        if building is None or building[1] != self.team:
            return False
        if building[0] == EntityType.BRIDGE:
            return True
        direction = self.known_conveyor_directions.get(receiver)
        return direction is not None and direction != receiver.direction_to(source)

    def transport_receiver(self, pos: Position) -> Position | None:
        """Return the next tile in the known allied transport graph."""
        building = self.known_buildings.get(pos)
        if building is None or building[1] != self.team:
            return None
        if building[0] in {EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR}:
            direction = self.known_conveyor_directions.get(pos)
            return None if direction is None else self.tile_cache.neighbor(pos, direction)
        if building[0] == EntityType.BRIDGE:
            return self.known_bridge_targets.get(pos)
        return None

    def known_network_loads(self, network: set[Position]) -> dict[Position, int]:
        """Estimate how many known harvesters consume each downstream lane."""
        loads = {pos: 0 for pos in network}
        harvesters = sorted(
            (
                pos
                for pos, building in self.known_buildings.items()
                if building is not None
                and building[1] == self.team
                and building[0] == EntityType.HARVESTER
            ),
            key=lambda pos: (pos.x, pos.y),
        )
        for ore in harvesters:
            receivers = [
                receiver
                for direction in ORTHOGONAL_DIRECTIONS
                if (receiver := self.tile_cache.neighbor(ore, direction)) is not None
                and receiver in network
                and self.network_receiver_accepts(receiver, ore)
            ]
            if not receivers:
                continue
            # A harvester chooses among accepting outputs fairly.  Assigning it
            # to the least loaded receiver is a conservative approximation of
            # that balancing without double-counting one source four times.
            receiver = min(receivers, key=lambda pos: (loads.get(pos, 0), pos.x, pos.y))
            seen: set[Position] = set()
            while receiver in network and receiver not in seen:
                seen.add(receiver)
                loads[receiver] = loads.get(receiver, 0) + 1
                if self.is_core_receiver_tile(receiver):
                    break
                next_receiver = self.transport_receiver(receiver)
                if next_receiver is None:
                    break
                receiver = next_receiver
        return loads

    def transport_lane_is_saturated(self, pos: Position, loads: dict[Position, int]) -> bool:
        """Combine structural source count with four observed occupancy turns."""
        if self.is_core_receiver_tile(pos):
            return False
        if loads.get(pos, 0) >= MAX_HARVESTERS_PER_LINE:
            return True
        return self.network_memory.continuously_busy(pos)

    def harvester_is_connected(self, ore: Position) -> bool:
        """Return whether a harvester has a directed allied conveyor path to core."""
        network = self.known_connected_network()
        for direction in ORTHOGONAL_DIRECTIONS:
            receiver = self.tile_cache.neighbor(ore, direction)
            if receiver is None:
                continue
            if receiver in network and self.network_receiver_accepts(receiver, ore):
                return True
        return False

    def connection_plan(
            self,
            controller: Controller,
            ore: Position,
    ) -> tuple[
        Position,
        list[Position],
        dict[Position, Direction],
        dict[Position, Position],
        Position,
    ] | None:
        """Return the cheapest capacity-safe branch, with bridges if required."""
        network = self.known_connected_network()
        if not network:
            return None
        core_receivers = self.sector_core_receiver_tiles()
        if not core_receivers:
            return None

        direct_tree = dedicated_route_tree(core_receivers)
        start_new_line = starts_new_line(
            self.harvesters_built_count,
            bool(self.active_line_tiles),
            PLANNED_HARVESTERS_PER_LINE,
        )
        active_merge_tree = direct_tree | (
            network & self.active_line_tiles
        )
        primary_tree = direct_tree if start_new_line else active_merge_tree
        # The active-line counter belongs to this process, so it stays exact
        # without global communication.  Four sources exactly fill one owned
        # trunk; the next source starts a physically separate route.
        conveyor_cost = controller.get_conveyor_cost()[0]

        def plan_to(tree: set[Position], allowed_core_entries: set[Position]):
            def can_use_tile(pos: Position) -> bool:
                return pos not in network and self.traversable_for_connection(pos)

            def can_use_edge(pos: Position, direction: Direction) -> bool:
                building = self.known_buildings.get(pos)
                if building is None or building[1] != self.team:
                    return True
                if building[0] not in {EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR}:
                    return building[0] == EntityType.ROAD
                return self.known_conveyor_directions.get(pos) == direction

            def tile_cost(pos: Position, _direction: Direction) -> int:
                building = self.known_buildings.get(pos)
                if (
                    building is not None
                    and building[1] == self.team
                    and building[0] in {EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR}
                ):
                    return 0
                return conveyor_cost

            return incremental_steiner_branch(
                starts=self.buildable_approaches(ore),
                tree=tree,
                directions=ORTHOGONAL_DIRECTIONS,
                neighbor_fn=self.tile_cache.neighbor,
                can_use_tile=can_use_tile,
                can_use_edge=can_use_edge,
                receiver_accepts=lambda receiver, source: self.network_plan_receiver_accepts(
                    receiver,
                    source,
                    allowed_core_entries,
                ),
                tile_cost=tile_cost,
                max_expansions=STEINER_MAX_EXPANSIONS,
                heuristic_fn=lambda pos: conveyor_cost * min(
                    abs(pos.x - anchor.x) + abs(pos.y - anchor.y)
                    for anchor in tree
                ),
            )

        sector_entries = self.sector_core_entry_tiles()
        plan = plan_to(primary_tree, sector_entries)
        if plan is not None:
            approach, build_tiles, directions, anchor = plan
            self.last_plan_starts_new_line = start_new_line
            return approach, build_tiles, directions, {}, anchor

        # Edge maps can leave a sector with only one usable intake.  Once it
        # is occupied, reserve any still-free perimeter port rather than
        # forcing every later mine through that same bottleneck.  Existing
        # transport remains excluded, so this is a new physical lane, not an
        # unmeasured cross-process merge.
        global_receivers = self.core_receiver_tiles()
        global_entries = self.core_entry_tiles()
        global_tree = dedicated_route_tree(global_receivers)
        plan = plan_to(global_tree, global_entries)
        if plan is not None:
            approach, build_tiles, directions, anchor = plan
            self.last_plan_starts_new_line = True
            return approach, build_tiles, directions, {}, anchor

        # If an older lane blocks an independent route, jump it before
        # considering a merge.  The previous ordering merged immediately and
        # produced exactly the narrow high-utilisation trunks seen in the
        # submitted replays even though a bridge route was affordable.
        bridge_plan = self.bridge_connection_plan(
            controller,
            ore,
            network,
            primary_tree,
            sector_entries,
        )
        bridge_starts_new_line = start_new_line
        if bridge_plan is None:
            bridge_plan = self.bridge_connection_plan(
                controller,
                ore,
                network,
                global_tree,
                global_entries,
            )
            bridge_starts_new_line = bridge_plan is not None
        self.last_plan_starts_new_line = bridge_starts_new_line and bridge_plan is not None
        return bridge_plan

    def bridge_connection_plan(
            self,
            controller: Controller,
            ore: Position,
            network: set[Position],
            anchors: set[Position],
            allowed_core_entries: set[Position],
    ) -> tuple[
        Position,
        list[Position],
        dict[Position, Direction],
        dict[Position, Position],
        Position,
    ] | None:
        """Find a minimum-cost fallback that can jump blocked transport tiles."""
        best = None
        conveyor_cost = controller.get_conveyor_cost()[0]
        bridge_cost = controller.get_bridge_cost()[0]
        for approach in self.buildable_approaches(ore):
            result = self.transport_route_with_bridges(
                approach,
                anchors,
                network,
                conveyor_cost,
                bridge_cost,
            )
            if result is None:
                continue
            nodes, directions, bridge_targets, cost = result
            anchor = nodes[-1]
            build_tiles = nodes[:-1]
            source = ore if not build_tiles else build_tiles[-1]
            if not self.network_plan_receiver_accepts(
                anchor,
                source,
                allowed_core_entries,
            ):
                continue
            if any(
                self.is_incompatible_existing_conveyor(tile, direction)
                for tile, direction in directions.items()
            ):
                continue
            candidate = (approach, build_tiles, directions, bridge_targets, anchor)
            score = (cost, len(bridge_targets), len(build_tiles))
            if best is None or score < best[0]:
                best = (score, candidate)
        return None if best is None else best[1]

    def transport_route_with_bridges(
            self,
            start: Position,
            anchors: set[Position],
            network: set[Position],
            conveyor_cost: int,
            bridge_cost: int,
    ) -> tuple[
        list[Position],
        dict[Position, Direction],
        dict[Position, Position],
        int,
    ] | None:
        """Dijkstra route using costly bridge jumps only across real blocks."""
        queue = [(0, start.x, start.y, start)]
        costs = {start: 0}
        came_from: dict[Position, tuple[Position, bool]] = {}
        expansions = 0

        def usable(pos: Position) -> bool:
            return pos in anchors or (pos not in network and self.traversable_for_connection(pos))

        while queue:
            cost, _, _, current = heappop(queue)
            if cost != costs.get(current):
                continue
            if current in anchors:
                nodes = [current]
                bridge_targets: dict[Position, Position] = {}
                while current != start:
                    previous, is_bridge = came_from[current]
                    if is_bridge:
                        bridge_targets[previous] = current
                    nodes.append(previous)
                    current = previous
                nodes.reverse()
                directions = {
                    node: node.direction_to(nodes[index + 1])
                    for index, node in enumerate(nodes[:-1])
                    if node not in bridge_targets
                }
                return nodes, directions, bridge_targets, cost
            if expansions >= BRIDGE_ROUTE_MAX_EXPANSIONS:
                break
            expansions += 1

            for direction in ORTHOGONAL_DIRECTIONS:
                next_pos = self.tile_cache.neighbor(current, direction)
                if next_pos is None:
                    continue
                if not usable(next_pos):
                    continue
                new_cost = cost + conveyor_cost
                if new_cost >= costs.get(next_pos, 10**9):
                    continue
                costs[next_pos] = new_cost
                came_from[next_pos] = (current, False)
                heappush(queue, (new_cost, next_pos.x, next_pos.y, next_pos))

            for direction in ORTHOGONAL_DIRECTIONS:
                dx, dy = direction.delta()
                for distance in range(2, BRIDGE_MAX_JUMP_DISTANCE + 1):
                    target = self.tile_cache.offset(
                        current,
                        dx * distance,
                        dy * distance,
                    )
                    if target is None:
                        continue
                    if not usable(target):
                        continue
                    intermediates = [
                        intermediate
                        for step in range(1, distance)
                        if (
                            intermediate := self.tile_cache.offset(
                                current,
                                dx * step,
                                dy * step,
                            )
                        ) is not None
                    ]
                    if not any(
                        pos in network or not self.traversable_for_connection(pos)
                        for pos in intermediates
                    ):
                        continue
                    new_cost = cost + bridge_cost
                    if new_cost >= costs.get(target, 10**9):
                        continue
                    costs[target] = new_cost
                    came_from[target] = (current, True)
                    heappush(queue, (new_cost, target.x, target.y, target))
        return None

    def assign_connection_target(self, controller: Controller, current: Position, ore: Position) -> bool:
        """Commit the shortest safe branch plan and path toward its first tile."""
        plan = self.connection_plan(controller, ore)
        if plan is None:
            return False
        approach, build_tiles, directions, bridge_targets, anchor = plan
        path_to_approach: list[Position] = []
        if build_tiles:
            path_to_approach = a_star_to_any(
                controller,
                current,
                {approach},
                self.traversable_for_planning,
                self.tile_cache.neighbor,
                movement_directions=DIRECTIONS,
                max_expansions=CONNECTION_A_STAR_MAX_EXPANSIONS,
            )
            if not path_to_approach and current != approach:
                return False
        path = list(path_to_approach)
        for tile in build_tiles:
            if not path or path[-1] != tile:
                path.append(tile)
            bridge_target = bridge_targets.get(tile)
            if bridge_target is None or bridge_target == anchor:
                continue
            detour = a_star_to_any(
                controller,
                tile,
                {bridge_target},
                self.traversable_for_planning,
                self.tile_cache.neighbor,
                movement_directions=DIRECTIONS,
                max_expansions=CONNECTION_A_STAR_MAX_EXPANSIONS,
            )
            if not detour:
                return False
            path.extend(detour)
        self.target_ore = ore
        self.target_resource = (
            RESOURCE_TITANIUM
            if self.resource_at(ore) == Environment.ORE_TITANIUM
            else RESOURCE_AXIONITE
        )
        self.target_is_connection = True
        self.scout_target = None
        self.path = path
        self.path_index = 0
        self.conveyor_path_tiles = set(build_tiles)
        self.conveyor_directions = directions
        self.bridge_targets = bridge_targets
        self.connection_anchor = anchor
        self.connection_starts_new_line = self.last_plan_starts_new_line
        self.mode = "connect"
        return True

    def traversable_for_planning(self, _controller: Controller | None, pos: Position) -> bool:
        """Return whether a builder may move through ``pos`` while planning a path."""
        if (
            not self.in_bounds(pos)
            or pos in self.permanently_blocked
            or self.is_yield_blocked(pos)
        ):
            return False
        env = self.known_env.get(pos)
        if env == Environment.WALL or env in ORE_TYPES:
            return False
        building = self.known_buildings.get(pos)
        return building is None or (
            building[0] in PASSABLE_BUILDINGS
            and (building[0] != EntityType.CORE or building[1] == self.team)
        )

    def traversable_for_ore_path(self, controller: Controller | None, pos: Position) -> bool:
        """Allow ore-route A* to use only already observed passable cells.

        A deposit may appear on the edge of vision.  Its approach and every
        intermediate route tile must nevertheless be known: otherwise A*
        treats an unknown cell as empty and expands into unexplored terrain.
        Scouting keeps using ``traversable_for_planning`` so it can still take
        its deliberate one-step move into a new frontier cell.
        """
        return pos in self.known_env and self.traversable_for_planning(controller, pos)

    def traversable_for_connection(self, pos: Position) -> bool:
        """Whether a branch conveyor can be safely installed on ``pos``.

        Movement may cross an opponent's road or conveyor, but a transport
        branch cannot use it: it would either have no ownership or require an
        impossible overwrite.  Keep the two predicates separate so A* still
        has normal movement freedom while branch planning is ownership-safe.
        """
        if (
            not self.in_bounds(pos)
            or pos in self.permanently_blocked
            or self.is_yield_blocked(pos)
            # A conveyor branch is a committed construction job.  Unlike
            # scouting, it must not invent a shortcut through unseen terrain:
            # an unknown wall would otherwise leave a half-built branch with
            # its last conveyor pointing into the obstacle.
            or pos not in self.known_env
        ):
            return False
        env = self.known_env.get(pos)
        if env == Environment.WALL or env in ORE_TYPES:
            return False
        building = self.known_buildings.get(pos)
        if building is None:
            return True
        building_type, building_team = building
        # An already laid conveyor may belong to another builder's live route
        # that is outside this bot's vision.  It is not rewriteable.  The
        # connection planner validates its existing direction before using it
        # as part of a branch; roads carry no resource flow and may safely be
        # converted on the next action.
        if building_team != self.team:
            # Enemy roads remain walkable.  A builder can deliberately enter
            # one, destroy it from underfoot, and replace it with the planned
            # conveyor.  Treating every rival road as a permanent wall made a
            # single scouting trail sever otherwise short mining branches.
            return building_type == EntityType.ROAD
        if building_type == EntityType.ROAD:
            return True
        return building_type in {EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR}

    def is_incompatible_existing_conveyor(self, pos: Position, direction: Direction) -> bool:
        """Return whether an allied conveyor at ``pos`` points the wrong way."""
        building = self.known_buildings.get(pos)
        if (
            building is None
            or building[1] != self.team
            or building[0] not in {EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR}
        ):
            return False
        return self.known_conveyor_directions.get(pos) != direction

    def buildable_approaches(self, ore_pos: Position) -> list[Position]:
        """Return cardinal ore-adjacent tiles suitable for the first branch conveyor."""
        return [
            approach
            for direction in ORTHOGONAL_DIRECTIONS
            if (approach := self.tile_cache.neighbor(ore_pos, direction)) is not None
            if (
                self.is_core_receiver_tile(approach)
                or self.traversable_for_connection(approach)
            )
        ]

    def work_direction_progress(self, pos: Position) -> int:
        """Measure signed progress from the core along this builder's sector direction."""
        if self.core_pos is None or self.work_direction is None:
            return 0
        forward = self.tile_cache.neighbor(self.core_pos, self.work_direction)
        if forward is None:
            return 0
        return (
            (pos.x - self.core_pos.x) * (forward.x - self.core_pos.x)
            + (pos.y - self.core_pos.y) * (forward.y - self.core_pos.y)
        )

    def work_direction_priority(self, pos: Position) -> int:
        """Prefer positions ahead of the core over positions outside the sector."""
        if self.core_pos is None or self.work_direction is None:
            return 0
        return 0 if self.work_direction_progress(pos) > 0 else 1

    def core_distance(self, pos: Position) -> int:
        """Return the Chebyshev distance from the core to ``pos``."""
        if self.core_pos is None:
            return 0
        return max(abs(pos.x - self.core_pos.x), abs(pos.y - self.core_pos.y))

    def work_direction_lateral_offset(self, pos: Position) -> int:
        """Return the perpendicular distance from this builder's sector axis."""
        if self.core_pos is None or self.work_direction is None:
            return 0
        forward = self.tile_cache.neighbor(self.core_pos, self.work_direction)
        if forward is None:
            return 0
        return abs(
            (pos.x - self.core_pos.x) * (forward.y - self.core_pos.y)
            - (pos.y - self.core_pos.y) * (forward.x - self.core_pos.x)
        )

    def remember_route_position(self, pos: Position) -> None:
        """Maintain a bounded recent-route history used to penalize revisits."""
        if self.recent_route and self.recent_route[-1] == pos:
            return
        self.scout_total_visits[pos] = self.scout_total_visits.get(pos, 0) + 1
        if pos in self.recent_route and self.scout_total_visits[pos] >= 3:
            route = list(self.recent_route)
            cycle_start = len(route) - 1 - route[::-1].index(pos)
            cycle = route[cycle_start:]
            # A repeated A-B-A-B oscillation is the most common local failure,
            # so a two-position cycle is already sufficient evidence here.
            if len(cycle) >= 2:
                avoid_until = self.current_round + SCOUT_DEAD_END_AVOID_ROUNDS
                for cycle_pos in cycle:
                    self.scout_avoid_until[cycle_pos] = max(
                        avoid_until,
                        self.scout_avoid_until.get(cycle_pos, 0),
                    )
                if self.target_ore is None:
                    self.scout_cycle_replan = True
                    self.scout_cycle_count += 1
        self.recent_route.append(pos)
        self.recent_route_visits[pos] = self.recent_route_visits.get(pos, 0) + 1
        if len(self.recent_route) <= SCOUT_ROUTE_MEMORY_TILES:
            return
        old_pos = self.recent_route.popleft()
        old_count = self.recent_route_visits[old_pos] - 1
        if old_count == 0:
            del self.recent_route_visits[old_pos]
        else:
            self.recent_route_visits[old_pos] = old_count

    def scout_path_step_cost(self, origin: Position, pos: Position) -> int:
        """Penalize loops while still allowing a real dead-end retreat."""
        cost = self.recent_route_visits.get(pos, 0) * SCOUT_REVISIT_STEP_PENALTY
        cost += min(self.scout_total_visits.get(pos, 0), 6) * SCOUT_PERSISTENT_REVISIT_PENALTY
        if self.scout_avoid_until.get(pos, 0) > self.current_round:
            cost += SCOUT_DEAD_END_PENALTY
        inward_steps = max(0, self.core_distance(origin) - self.core_distance(pos))
        return cost + inward_steps * SCOUT_INWARD_STEP_PENALTY

    def newly_visible_tiles(self, centre: Position) -> int:
        """Count unknown tiles that would enter vision from ``centre``."""
        visible = 0
        for dx, dy in SCOUT_VISION_OFFSETS:
            pos = self.tile_cache.offset(centre, dx, dy)
            if pos is not None and pos not in self.observed_tiles:
                visible += 1
        return visible

    def ore_hint_progress(self, current: Position, candidate: Position) -> int:
        """Measure how much a candidate step approaches an inferred or reported ore hint."""
        if not self.inferred_ores and not self.reported_ores:
            return 0
        current_distance = 10**9
        candidate_distance = 10**9
        for hints in (self.inferred_ores, self.reported_ores):
            for hint in hints:
                current_distance = min(
                    current_distance,
                    max(abs(current.x - hint.x), abs(current.y - hint.y)),
                )
                candidate_distance = min(
                    candidate_distance,
                    max(abs(candidate.x - hint.x), abs(candidate.y - hint.y)),
                )
        return max(0, current_distance - candidate_distance)

    def scout_frontier_pre_score(self, current: Position, candidate: Position) -> tuple[int, int]:
        """Score a frontier cheaply before calculating expensive visibility details."""
        forward_progress = self.work_direction_progress(candidate)
        in_sector = int(self.work_direction is None or forward_progress > 0)
        returning = max(0, self.core_distance(current) - self.core_distance(candidate))
        distance = max(abs(candidate.x - current.x), abs(candidate.y - current.y))
        return 0, (
            in_sector * SCOUT_SECTOR_BONUS
            + forward_progress * SCOUT_FORWARD_PROGRESS_WEIGHT
            - returning * SCOUT_RETURN_TO_BASE_WEIGHT
            - distance * SCOUT_DISTANCE_WEIGHT
            + self.ore_hint_progress(current, candidate) * SCOUT_ORE_HINT_PROGRESS_WEIGHT
        )

    def scout_frontier_score(self, current: Position, candidate: Position) -> tuple[int, int]:
        """Score a frontier by sector progress, new vision, routing, and ore hints."""
        forward_progress = self.work_direction_progress(candidate)
        in_sector = int(self.work_direction is None or forward_progress > 0)
        returning = max(0, self.core_distance(current) - self.core_distance(candidate))
        distance = max(abs(candidate.x - current.x), abs(candidate.y - current.y))
        score = (
            in_sector * SCOUT_SECTOR_BONUS
            + forward_progress * SCOUT_FORWARD_PROGRESS_WEIGHT
            + self.newly_visible_tiles(candidate) * SCOUT_NEW_VISION_WEIGHT
            - returning * SCOUT_RETURN_TO_BASE_WEIGHT
            - distance * SCOUT_DISTANCE_WEIGHT
            - self.work_direction_lateral_offset(candidate) * SCOUT_LATERAL_DEVIATION_WEIGHT
            + self.ore_hint_progress(current, candidate) * SCOUT_ORE_HINT_PROGRESS_WEIGHT
            + self.snake_sweep_bias(current, candidate)
        )
        return 0, score

    def scout_frontier_has_known_entry(self, frontier: Position) -> bool:
        """Return whether a frontier touches a known tile the builder can occupy."""
        for direction in DIRECTIONS:
            entry = self.tile_cache.neighbor(frontier, direction)
            if entry is None:
                continue
            if entry in self.known_env and self.traversable_for_planning(None, entry):
                return True
        return False

    def select_new_target(self, controller: Controller) -> None:
        """Choose work in priority order: connections, mineable ore, then scouting."""
        self.harvester_built = False
        self.harvester_fail_count = 0
        self.clear_ore_target()
        self.scout_target = None
        self.scout_target_direct = False
        self.scout_retry_pending = False
        current = self.get_cached_position()
        ore_sort_key = lambda pos: (
            0 if pos in self.assigned_ores else 1,
            self.work_direction_priority(pos),
            current.distance_squared(pos),
        )

        repair_plan = self.choose_network_repair_plan(controller, current)
        if repair_plan is not None:
            self.scout_target, self.path = repair_plan
            self.scout_target_direct = False
            self.path_index = 0
            self.mode = "repair"
            return

        for ore in sorted(self.connection_candidates(), key=ore_sort_key):
            if not self.assign_ore_target(controller, current, ore, connecting=True):
                self.defer_connection_for_survey(ore)
                continue
            return

        if len(self.observed_tiles) >= self.ore_retry_observed_count:
            self.ore_retry_observed_count = 0
            for ore in sorted(set(self.mineable_ores()), key=ore_sort_key):
                if self.is_harvester_on_tile(ore) or ore in self.skipped_ores:
                    continue
                if not self.assign_ore_target(controller, current, ore, connecting=False):
                    # The known map does not yet contain a route.  Require
                    # actual new observations before cycling through known ore
                    # targets again; a short round-based delay did not stop
                    # loops on maps with many isolated deposits.
                    self.defer_ore_for_survey(ore)
                    continue
                return

        scout_plan = self.choose_scout_plan(controller)
        if scout_plan is not None:
            self.scout_route_failures = 0
            self.scout_target, self.path = scout_plan
            self.scout_target_direct = False
            self.path_index = 0
            self.mode = "scout"
            return

        self.scout_route_failures += 1

        patrol_plan = self.choose_network_patrol_plan(controller, current)
        if patrol_plan is not None:
            self.scout_target, self.path = patrol_plan
            self.scout_target_direct = False
            self.path_index = 0
            self.mode = "patrol"
            return

        # No usable frontier was found, so take one local right-hand step and
        # let the next turn select a fresh frontier from the newly seen tiles.
        step = self.choose_right_hand_scout_step(current)
        if step is not None:
            self.scout_target = step
            self.scout_target_direct = False
            self.path = [step]
            self.path_index = 0
            self.mode = "scout"
            return
        self.scout_retry_pending = True

    def choose_network_patrol_plan(
            self,
            controller: Controller,
            current: Position,
    ) -> tuple[Position, list[Position]] | None:
        """Visit the builder's least-patrolled transport after exploration."""
        candidates = [pos for pos in self.owned_network_tiles if pos != current]
        candidates.sort(
            key=lambda pos: (
                self.scout_total_visits.get(pos, 0),
                -current.distance_squared(pos),
                pos.x,
                pos.y,
            )
        )
        targets = set(candidates[:NETWORK_PATROL_GOAL_LIMIT])
        if not targets:
            return None
        path = a_star_to_any(
            controller,
            current,
            targets,
            self.traversable_for_ore_path,
            self.tile_cache.neighbor,
            movement_directions=DIRECTIONS,
            extra_step_cost_fn=lambda pos: min(self.scout_total_visits.get(pos, 0), 8),
            max_expansions=NETWORK_PATROL_MAX_EXPANSIONS,
        )
        if not path:
            return None
        return path[-1], path

    def choose_network_repair_plan(
            self,
            controller: Controller,
            current: Position,
    ) -> tuple[Position, list[Position]] | None:
        """Route the owning builder back to a rebuildable visible break."""
        repairable = [
            pos
            for pos in self.damaged_network_tiles
            if (
                pos in self.tile_cache.visible_tiles
                and (
                    self.known_buildings.get(pos) is None
                    or self.known_buildings[pos] == (EntityType.ROAD, self.team)
                )
            )
        ]
        repairable.sort(key=lambda pos: (current.distance_squared(pos), pos.x, pos.y))
        for damaged in repairable:
            approaches = {
                approach
                for direction in DIRECTIONS
                if (approach := self.tile_cache.neighbor(damaged, direction)) is not None
                if (
                    approach != damaged
                    and self.traversable_for_ore_path(None, approach)
                )
            }
            if current in approaches:
                return current, []
            if not approaches:
                continue
            path = a_star_to_any(
                controller,
                current,
                approaches,
                self.traversable_for_ore_path,
                self.tile_cache.neighbor,
                movement_directions=DIRECTIONS,
                max_expansions=CONNECTION_A_STAR_MAX_EXPANSIONS,
            )
            if path:
                return path[-1], path
        return None

    def snake_sweep_bias(self, current: Position, candidate: Position) -> int:
        """Bias exploration toward the current lateral leg of the snake sweep."""
        if self.scout_sweep_direction is None:
            return 0
        dx, dy = self.scout_sweep_direction.delta()
        lateral_step = (candidate.x - current.x) * dx + (candidate.y - current.y) * dy
        if lateral_step > 0:
            return 3
        if lateral_step < 0:
            return -3
        return 0

    def scout_step_is_viable(self, pos: Position) -> bool:
        """Return whether a one-step right-hand scouting move may use ``pos``."""
        return (
            self.traversable_for_planning(None, pos)
            and self.tile_cache.builder_id_at(pos) is None
        )

    def scout_a_star_traversable(self, pos: Position, targets: set[Position]) -> bool:
        """Use known terrain and enter unknown space only at a frontier goal."""
        if not self.traversable_for_planning(None, pos):
            return False
        return pos in targets or pos in self.known_env or self.is_core_receiver_tile(pos)

    def choose_right_hand_scout_step(
            self,
            current: Position,
            require_new_vision: bool = False,
    ) -> Position | None:
        """Choose the least-visited useful escape, including a reverse step."""
        if self.work_direction is None:
            return None
        selected = choose_information_gain_step(
            current=current,
            directions=DIRECTIONS,
            neighbor=self.tile_cache.neighbor,
            viable=self.scout_step_is_viable,
            vision_gain=self.newly_visible_tiles,
            total_visits=self.scout_total_visits,
            recent_visits=self.recent_route_visits,
            avoided=lambda pos: self.scout_avoid_until.get(pos, 0) > self.current_round,
            forward_progress=self.work_direction_progress,
            sweep_bias=self.snake_sweep_bias,
            heading=self.scout_heading,
            require_new_vision=require_new_vision,
        )
        if selected is None:
            return None
        direction, candidate = selected
        self.scout_heading = direction
        return candidate

    def choose_scout_plan(self, controller: Controller) -> tuple[Position, list[Position]] | None:
        """Prefer an O(vision) information-gain step, then bounded frontier A*."""
        if not self.scout_frontier and self.known_env and not self.scout_frontier_initialized:
            self.rebuild_scout_frontier()
        current = self.get_cached_position()
        # While new territory is one move away, global A* is wasted work.  A
        # sector-biased least-visited step gives broad snake-like coverage and
        # keeps ordinary scouting comfortably under the 2 ms turn limit.
        local_step = self.choose_right_hand_scout_step(
            current,
            require_new_vision=True,
        )
        if local_step is not None:
            return local_step, [local_step]
        candidates = [
            pos for pos in self.scout_frontier
            if (
                pos not in self.observed_tiles
                and pos not in self.permanently_blocked
                and pos not in self.unreachable_scout_targets
                and self.scout_frontier_has_known_entry(pos)
            )
        ]
        candidates.sort(key=lambda pos: self.scout_frontier_pre_score(current, pos), reverse=True)
        candidates = candidates[:SCOUT_FRONTIER_CANDIDATE_LIMIT]
        candidates.sort(key=lambda pos: self.scout_frontier_score(current, pos), reverse=True)
        candidates = candidates[:SCOUT_PATH_GOAL_LIMIT]
        if not candidates:
            return None
        # One multi-goal search is deliberately cheaper than trying each
        # attractive frontier separately under the 2 ms per-unit budget.  The
        # score selects the useful region; A* then chooses its cheapest
        # reachable boundary cell and naturally ignores frontiers behind walls.
        targets = set(candidates)
        path = a_star_to_any(
            controller,
            current,
            targets,
            lambda _controller, pos: self.scout_a_star_traversable(pos, targets),
            self.tile_cache.neighbor,
            movement_directions=DIRECTIONS,
            extra_step_cost_fn=lambda pos: self.scout_path_step_cost(current, pos),
            max_expansions=SCOUT_PATH_MAX_EXPANSIONS,
        )
        if path:
            return path[-1], path
        if current in targets:
            return current, []
        return None

    def cancel_scout_route(self) -> None:
        """Drop a stalled or cyclic route so target selection runs immediately."""
        self.scout_target = None
        self.scout_target_direct = False
        self.path = []
        self.path_index = 0
        self.scout_cycle_replan = False
        self.next_select_round = self.current_round

    def reject_scout_target(self, target: Position | None) -> None:
        """Forget an unreachable frontier target and clear its obsolete path."""
        if target is not None:
            self.unreachable_scout_targets.add(target)
            self.scout_frontier.discard(target)
        self.scout_target = None
        self.scout_target_direct = False
        self.path = []
        self.path_index = 0
        self.conveyor_path_tiles = set()

    def follow_path_and_build(self, controller: Controller) -> None:
        """Advance along the active path, constructing branch conveyors when required."""
        current = self.get_cached_position()
        if self.scout_target is not None and self.scout_target_direct:
            self.follow_direct_scout_target(controller, current)
            return
        # The path can begin on the ore-adjacent tile the builder already
        # occupies when it erects the harvester.  Build that first conveyor
        # before consuming the path node; otherwise the builder would walk
        # away and leave the new harvester disconnected.
        if self.target_is_connection and current in self.conveyor_path_tiles:
            if self.ensure_tree_conveyor(controller, current):
                return
            if not self.has_expected_tree_conveyor(controller, current):
                self.replan_or_wait_for_connection_tile(controller, current)
                return
        while self.path_index < len(self.path) and current == self.path[self.path_index]:
            self.path_index += 1
        if self.path_index >= len(self.path):
            if self.scout_target is not None:
                if current == self.scout_target:
                    self.scout_target = None
                else:
                    self.reject_scout_target(self.scout_target)
                    self.select_new_target(controller)
            elif self.target_ore is not None:
                if self.target_is_connection:
                    if not self.harvester_is_connected(self.target_ore):
                        if self.replan_active_ore_target(controller):
                            return
                        self.replan_after_yield = True
                        return
                    else:
                        self.pending_network_ores.discard(self.target_ore)
                        if (
                            self.connection_anchor is not None
                            and self.is_core_receiver_tile(self.connection_anchor)
                        ):
                            self.dedicated_line_committed = True
                        self.unfinished_branch_tiles.clear()
                self.clear_ore_target()
                self.select_new_target(controller)
            return

        next_pos = self.path[self.path_index]
        direction = current.direction_to(next_pos)
        if direction == Direction.CENTRE:
            self.path_index += 1
            return
        if current.distance_squared(next_pos) > 2:
            if self.target_is_connection:
                self.schedule_replan_after_yield()
                return
            self.select_new_target(controller)
            return
        if self.target_is_connection and next_pos in self.conveyor_path_tiles:
            if self.is_enemy_road(next_pos):
                if controller.can_move(direction):
                    controller.move(direction)
                return
            if self.ensure_tree_conveyor(controller, next_pos):
                return
            if not self.has_expected_tree_conveyor(controller, next_pos):
                self.replan_or_wait_for_connection_tile(controller, next_pos)
                return
        if not self.is_cached_tile_passable(next_pos):
            self.try_prepare_tile(controller, next_pos)
        if controller.can_move(direction):
            controller.move(direction)
        elif not self.is_cached_tile_passable(next_pos):
            self.replan_after_blocked_step(controller, next_pos)

    def follow_direct_scout_target(self, controller: Controller, current: Position) -> None:
        """Advance straight to a frontier and use the right-hand rule around a block."""
        target = self.scout_target
        if target is None:
            self.scout_target_direct = False
            return
        if current == target:
            self.scout_target = None
            self.scout_target_direct = False
            return

        direct_direction = current.direction_to(target)
        if self.try_move_scout_step(controller, current, direct_direction):
            return

        # The direct line is no longer usable.  Keep the same target, but
        # inspect the immediate clockwise detour before giving it up.
        self.scout_heading = direct_direction
        detour = self.choose_right_hand_scout_step(current)
        if detour is None:
            self.reject_scout_target(target)
            self.scout_retry_pending = True
            return
        detour_direction = current.direction_to(detour)
        if not self.try_move_scout_step(controller, current, detour_direction):
            self.scout_retry_pending = True

    def try_move_scout_step(
            self,
            controller: Controller,
            current: Position,
            direction: Direction,
    ) -> bool:
        """Prepare and take one legal scouting step in ``direction`` if possible."""
        if direction == Direction.CENTRE:
            return False
        next_pos = self.tile_cache.neighbor(current, direction)
        if next_pos is None:
            return False
        if not self.is_cached_tile_passable(next_pos):
            self.try_prepare_tile(controller, next_pos)
        if not controller.can_move(direction):
            return False
        controller.move(direction)
        self.scout_heading = direction
        return True

    def try_prepare_tile(self, controller: Controller, target: Position) -> None:
        """Prepare a blocked next tile by building a road or branch conveyor."""
        if self.target_is_connection and target in self.conveyor_path_tiles:
            self.ensure_tree_conveyor(controller, target)
            return
        if self.tile_cache.builder_id_at(target) is not None:
            return
        if controller.can_build_road(target):
            road_id = controller.build_road(target)
            self.tile_cache.remember_building(target, road_id, EntityType.ROAD, self.team)
            self.last_progress_round = self.rounds_alive
            return
        # can_build_road is also false while resources/actions are temporarily
        # unavailable.  Only immutable terrain or a durable incompatible
        # building belongs in the permanent obstacle set.
        env = self.known_env.get(target)
        building = self.known_buildings.get(target)
        if (
            env == Environment.WALL
            or env in ORE_TYPES
            or (
                building is not None
                and building[0] not in PASSABLE_BUILDINGS
                and building[0] != EntityType.BUILDER_BOT
            )
        ):
            self.mark_staticly_blocked(target)

    def replan_after_blocked_step(self, controller: Controller, target: Position) -> None:
        """Yield or recompute the active job when the next path tile is blocked."""
        # is_tile_passable() is also false while another builder occupies an
        # otherwise usable road.  For allied builders, entity IDs reflect spawn
        # order: the later builder yields, while the earlier one keeps priority.
        blocking_id = self.tile_cache.builder_id_at(target)
        if blocking_id is not None:
            if self.tile_cache.entity_team(blocking_id) == self.team:
                if self.entity_id is not None and blocking_id < self.entity_id:
                    self.yield_to_higher_priority_builder(controller, target)
                return
            self.schedule_replan_after_yield()
            return
        env = self.known_env.get(target)
        building = self.known_buildings.get(target)
        cached_passable_building = (
            building is not None
            and building[0] in PASSABLE_BUILDINGS
            and (building[0] != EntityType.CORE or building[1] == self.team)
        )
        if not is_static_step_obstacle(
            env == Environment.WALL or env in ORE_TYPES,
            building is not None,
            cached_passable_building,
        ):
            # The step can be temporarily unaffordable or the action may have
            # been consumed; a friendly transport can also be momentarily
            # occupied between this bot's scan and action.  Preserve both as
            # traversable and let the ordinary stall logic replan.
            return
        self.mark_staticly_blocked(target)
        if self.replan_active_ore_target(controller):
            return
        if self.target_is_connection:
            self.replan_after_yield = True
            return
        if self.scout_target is not None:
            self.reject_scout_target(self.scout_target)
        else:
            self.path = []
            self.path_index = 0
        self.select_new_target(controller)

    def yield_to_higher_priority_builder(self, controller: Controller, blocking_pos: Position) -> None:
        """Move aside for an earlier allied builder and schedule a fresh route."""
        current = self.get_cached_position()
        self.yield_blocked_until[blocking_pos] = self.current_round + YIELD_ROUTE_AVOID_ROUNDS
        retreat_positions: list[Position] = []
        seen: set[Position] = set()
        for pos in reversed(self.recent_route):
            if pos == current or pos == blocking_pos or pos in seen:
                continue
            seen.add(pos)
            if current.distance_squared(pos) <= 2:
                retreat_positions.append(pos)

        # A newly spawned bot may not have route history yet.  In that case use
        # any free neighbouring road, preferring a step away from the blocker.
        route_history = set(self.recent_route)
        fallback_positions = [
            candidate
            for direction in DIRECTIONS
            if (candidate := self.tile_cache.neighbor(current, direction)) is not None
            if (
                candidate != blocking_pos
                and candidate not in route_history
            )
        ]
        fallback_positions.sort(
            key=lambda pos: pos.distance_squared(blocking_pos),
            reverse=True,
        )

        for retreat_pos in fallback_positions + retreat_positions:
            direction = current.direction_to(retreat_pos)
            if direction == Direction.CENTRE or not controller.can_move(direction):
                continue
            controller.move(direction)
            self.schedule_replan_after_yield()
            return

        # Even when there is no immediately passable retreat tile, discard the
        # stale route so the next turn can attempt a different route or goal.
        self.schedule_replan_after_yield()

    def schedule_replan_after_yield(self) -> None:
        """Discard the current route so the next turn can plan around a conflict."""
        if self.target_ore is not None:
            self.path = []
            self.path_index = 0
            self.conveyor_path_tiles = set()
            self.conveyor_directions = {}
            self.bridge_targets = {}
            self.connection_anchor = None
            self.replan_after_yield = True
            return
        self.scout_target = None
        self.path = []
        self.path_index = 0
        self.conveyor_path_tiles = set()
        self.conveyor_directions = {}
        self.bridge_targets = {}
        self.connection_anchor = None
        self.replan_after_yield = True

    def active_yield_blocked_tiles(self) -> set[Position]:
        """Return tiles still temporarily avoided after yielding to an ally."""
        return {
            pos
            for pos, expires_on_round in self.yield_blocked_until.items()
            if self.current_round <= expires_on_round
        }

    def is_yield_blocked(self, pos: Position) -> bool:
        """Return whether ``pos`` remains temporarily excluded after yielding."""
        return self.yield_blocked_until.get(pos, -1) >= self.current_round

    def mark_staticly_blocked(self, target: Position) -> None:
        """Remember a tile as permanently unusable when its obstacle is static."""
        env = self.known_env.get(target)
        building = self.known_buildings.get(target)
        if env != Environment.WALL and env not in ORE_TYPES:
            if building is None or building[0] in PASSABLE_BUILDINGS:
                return
        self.permanently_blocked.add(target)

    def ensure_tree_conveyor(self, controller: Controller, target: Position) -> bool:
        """Build or safely reuse the planned conveyor or bridge at a branch tile."""
        planned_bridge_target = self.bridge_targets.get(target)
        conveyor_direction = self.get_conveyor_direction(target)
        building_id = self.tile_cache.building_id_at(target)
        building = self.tile_cache.building_at(target)
        if building_id is not None and building is not None:
            if (
                planned_bridge_target is not None
                and building[0] == EntityType.BRIDGE
                and building[1] == self.team
                and self.known_bridge_targets.get(target) == planned_bridge_target
            ):
                return False
            building_type, building_team = building
            if (
                planned_bridge_target is None
                and building_type in {EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR}
                and building_team == self.team
                and self.tile_cache.entity_direction(building_id) == conveyor_direction
            ):
                self.known_conveyor_directions[target] = conveyor_direction
                return False
            if building_type == EntityType.ROAD and building_team != self.team:
                # Enemy logistics can be attacked only from the occupied tile.
                # The movement branch below deliberately steps onto the road;
                # spend as many actions as its remaining HP requires, then lay
                # the saved conveyor direction without abandoning the route.
                if target != self.get_cached_position() or not controller.can_fire(target):
                    return False
                remaining_hp = controller.get_hp(building_id)
                controller.fire(target)
                if remaining_hp <= GameConstants.BUILDER_BOT_ATTACK_DAMAGE:
                    self.tile_cache.forget_building(target)
                self.last_progress_round = self.rounds_alive
                return True
            if building_type == EntityType.MARKER:
                # Order-board markers are deliberately reserved cells.
                self.permanently_blocked.add(target)
                return False
            if (
                building_type == EntityType.ROAD
                and building_team == self.team
                and controller.can_destroy(target)
            ):
                controller.destroy(target)
                self.tile_cache.forget_building(target)
                self.connected_network_cache = None
                return True
            if (
                building_type in {
                    EntityType.CONVEYOR,
                    EntityType.ARMOURED_CONVEYOR,
                    EntityType.BRIDGE,
                }
                and building_team == self.team
                and target in self.unfinished_branch_tiles
                and target not in self.known_connected_network()
                and controller.can_destroy(target)
            ):
                controller.destroy(target)
                self.tile_cache.forget_building(target)
                self.connected_network_cache = None
                return True
            return False

        if planned_bridge_target is not None:
            # Bridges cannot be erected under a builder.  Normally they are
            # built from the preceding path tile; this fallback steps aside if
            # a replan starts while already standing on the bridge source.
            if self.get_cached_position() == target:
                for direction in DIRECTIONS:
                    if controller.can_move(direction):
                        controller.move(direction)
                        return True
                return False
            if not controller.can_build_bridge(target, planned_bridge_target):
                return False
            bridge_id = controller.build_bridge(target, planned_bridge_target)
            self.tile_cache.remember_building(
                target,
                bridge_id,
                EntityType.BRIDGE,
                self.team,
            )
            self.known_bridge_targets[target] = planned_bridge_target
            self.known_bridge_ids[target] = bridge_id
            self.network_blueprint[target] = (
                EntityType.BRIDGE,
                None,
                planned_bridge_target,
            )
            self.owned_network_tiles.add(target)
            self.active_line_tiles.add(target)
            self.unfinished_branch_tiles.add(target)
            self.connected_network_cache = None
            self.last_progress_round = self.rounds_alive
            return True

        if controller.can_build_conveyor(target, conveyor_direction):
            conveyor_id = controller.build_conveyor(target, conveyor_direction)
            self.tile_cache.remember_building(
                target,
                conveyor_id,
                EntityType.CONVEYOR,
                self.team,
                direction=conveyor_direction,
            )
            self.network_blueprint[target] = (
                EntityType.CONVEYOR,
                conveyor_direction,
                None,
            )
            self.owned_network_tiles.add(target)
            self.active_line_tiles.add(target)
            self.unfinished_branch_tiles.add(target)
            self.connected_network_cache = None
            self.last_progress_round = self.rounds_alive
            return True
        return False

    def has_expected_tree_conveyor(self, controller: Controller, target: Position) -> bool:
        """Check that ``target`` contains the allied transport required by the plan."""
        building_id = self.tile_cache.building_id_at(target)
        building = self.tile_cache.building_at(target)
        if building_id is None or building is None or building[1] != self.team:
            return False
        planned_bridge_target = self.bridge_targets.get(target)
        if planned_bridge_target is not None:
            return (
                building[0] == EntityType.BRIDGE
                and self.known_bridge_targets.get(target) == planned_bridge_target
            )
        if building[0] not in {
            EntityType.CONVEYOR,
            EntityType.ARMOURED_CONVEYOR,
        }:
            return False
        return self.tile_cache.entity_direction(building_id) == self.get_conveyor_direction(target)

    def replan_or_wait_for_connection_tile(self, controller: Controller, target: Position) -> None:
        """Never step past a branch tile that was not successfully built."""
        if target not in self.bridge_targets and self.get_conveyor_direction(target) == Direction.CENTRE:
            self.schedule_replan_after_yield()
            return
        if self.tile_cache.building_id_at(target) is None:
            # Usually an action cooldown or a temporary resource shortage.
            # Stay on the branch and try the exact same tile next turn.
            return
        if self.is_enemy_road(target) and target == self.get_cached_position():
            # Firing has action cooldown; retain the exact branch while the
            # builder waits for its next attack or replacement build.
            return
        # A foreign/indestructible building occupies the planned branch tile.
        # Exclude it from the next A* search instead of walking through it and
        # falsely declaring the harvester connected.
        self.permanently_blocked.add(target)
        self.schedule_replan_after_yield()

    def is_enemy_road(self, pos: Position) -> bool:
        """Return whether ``pos`` contains a walkable rival road."""
        building = self.known_buildings.get(pos)
        return (
            building is not None
            and building[0] == EntityType.ROAD
            and building[1] != self.team
        )

    def get_conveyor_direction(self, target: Position) -> Direction:
        """Return the planned outgoing direction for a branch conveyor tile."""
        direction = self.conveyor_directions.get(target)
        if direction is not None:
            return direction
        return Direction.CENTRE
