from collections import deque

from cambc import Controller, Direction, EntityType, Environment, GameConstants, Position

from base import BaseBot
from constants import (
    AXIONITE_TITANIUM_THRESHOLD,
    BUILDER_CODE_DIRECTIONS,
    DIRECTIONS,
    MARKER_KIND_ORE_AX,
    MARKER_KIND_ORE_TI,
    MARKER_KIND_SECTOR_ORE_AX,
    MARKER_KIND_SECTOR_ORE_TI,
    MARKER_KIND_SPAWN_DIRECTION,
    MARKER_KIND_SPAWN_ORE_AX,
    MARKER_KIND_SPAWN_ORE_TI,
    MAX_IDLE_ROUNDS,
    ORE_TYPES,
    ORTHOGONAL_DIRECTIONS,
    PASSABLE_BUILDINGS,
    RESOURCE_AXIONITE,
    RESOURCE_TITANIUM,
    SCOUT_DISTANCE_WEIGHT,
    SCOUT_FORWARD_PROGRESS_WEIGHT,
    SCOUT_FRONTIER_CANDIDATE_LIMIT,
    SCOUT_DEAD_END_AVOID_ROUNDS,
    SCOUT_DEAD_END_PENALTY,
    SCOUT_INWARD_STEP_PENALTY,
    SCOUT_LATERAL_DEVIATION_WEIGHT,
    SCOUT_NEW_VISION_WEIGHT,
    SCOUT_ORE_HINT_PROGRESS_WEIGHT,
    SCOUT_PATH_MAX_EXPANSIONS,
    SCOUT_PERSISTENT_REVISIT_PENALTY,
    SCOUT_REPLAN_STUCK_ROUNDS,
    SCOUT_RETURN_TO_BASE_WEIGHT,
    SCOUT_REVISIT_STEP_PENALTY,
    SCOUT_ROUTE_MEMORY_TILES,
    STUCK_KILL_ROUNDS,
    STEINER_MAX_EXPANSIONS,
    YIELD_ROUTE_AVOID_ROUNDS,
)
from geometry import decode_marker
from navigation import a_star_to_any
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
        self.team = None

        self.known_env: dict[Position, Environment] = {}
        self.known_buildings: dict[Position, tuple[EntityType, object] | None] = {}
        self.known_conveyor_directions: dict[Position, Direction] = {}
        self.known_bridge_targets: dict[Position, Position] = {}
        self.connected_network_cache: set[Position] | None = None
        self.observed_tiles: set[Position] = set()
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
        # Tiles laid by this builder for its current, not-yet-connected mine.
        # They may be safely reused after a newly discovered obstacle forces a
        # replan; conveyors made by any other builder remain immutable.
        self.unfinished_branch_tiles: set[Position] = set()
        self.pending_network_ores: set[Position] = set()
        self.harvester_built = False
        self.harvester_fail_count = 0
        self.skipped_ores: set[Position] = set()
        self.deferred_ores_until: dict[Position, int] = {}
        self.next_select_round = 0
        self.replan_after_yield = False
        self.mode = "scout"
        self.titanium_unlocked = False
        self.scout_heading: Direction | None = None
        self.scout_sweep_direction: Direction | None = None
        self.connection_survey_heading: Direction | None = None
        self.connection_survey_last_pos: Position | None = None

        self.last_pos: Position | None = None
        self.stuck_rounds = 0
        self.rounds_alive = 0
        self.last_progress_round = 0
        self.recent_route: deque[Position] = deque()
        self.recent_route_visits: dict[Position, int] = {}
        self.scout_total_visits: dict[Position, int] = {}
        self.scout_avoid_until: dict[Position, int] = {}
        self.scout_cycle_replan = False

    def run(self, controller: Controller) -> None:
        """Execute one turn of scouting, mining, or conveyor construction."""
        self.rounds_alive += 1
        self.current_round = controller.get_current_round()
        current = controller.get_position()
        if self.last_pos is not None and current == self.last_pos:
            self.stuck_rounds += 1
        else:
            self.stuck_rounds = 0
            self.last_progress_round = self.rounds_alive
            self.remember_route_position(current)
        self.last_pos = current

        # A mine already fitted with a harvester is a committed job.  Do not
        # self-destruct its only worker merely because a competing builder or
        # a temporary lack of titanium delayed one conveyor tile; replan the
        # branch instead and keep the four-builder fleet intact.
        if (
            self.target_ore is not None
            and not self.target_is_connection
            and self.stuck_rounds >= STUCK_KILL_ROUNDS
        ):
            controller.self_destruct()
            return
        if (
            self.target_ore is not None
            and not self.target_is_connection
            and self.rounds_alive - self.last_progress_round > MAX_IDLE_ROUNDS
        ):
            controller.self_destruct()
            return

        self.observe_tiles(controller)
        if self.team is None:
            self.team = controller.get_team()
        if self.core_pos is None:
            self.core_pos = self.find_home_core(controller)
        if self.core_pos is None:
            return
        if self.enemy_estimate is None:
            self.enemy_estimate = Position(
                self.map_width - 1 - self.core_pos.x,
                self.map_height - 1 - self.core_pos.y,
            )
        if self.work_direction is None:
            self.spawn_direction = self.core_pos.direction_to(current)
        self.read_ore_markers(controller)
        if self.work_direction is None:
            self.work_direction = self.spawn_direction
            if self.work_direction not in ORTHOGONAL_DIRECTIONS:
                self.work_direction = Direction.NORTH
        if self.scout_heading is None:
            self.scout_heading = self.work_direction
            self.scout_sweep_direction = self.work_direction.rotate_right().rotate_right()

        # Replan early when the next path is occupied or when returning to a
        # recently visited position closes a loop.  Exploration must never be
        # allowed to wait until the generic self-destruct watchdog fires.
        if (
            self.target_ore is None
            and self.scout_target is not None
            and (self.scout_cycle_replan or self.stuck_rounds >= SCOUT_REPLAN_STUCK_ROUNDS)
        ):
            self.cancel_scout_route()

        self.titanium_unlocked = controller.get_global_resources()[0] > AXIONITE_TITANIUM_THRESHOLD
        if not self.target_is_connection and self.try_build_nearby_harvester(controller, current):
            return
        if self.replan_after_yield:
            self.replan_after_yield = False
            if not self.replan_active_ore_target(controller):
                if self.target_is_connection:
                    self.survey_for_connection(controller)
                    self.replan_after_yield = True
                    return
                self.select_new_target(controller)
        else:
            self.maybe_select_new_target(controller)
        if self.target_ore is None and self.scout_target is None:
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
                controller.build_harvester(self.target_ore)
                self.record_harvester_built(self.target_ore)
                return

            self.harvester_fail_count += 1
            if self.harvester_fail_count >= 5:
                self.skipped_ores.add(self.target_ore)
                self.clear_ore_target()
                self.harvester_fail_count = 0
                self.select_new_target(controller)
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
            self.next_select_round = current_round + (1 if self.scout_retry_pending else 15)

    def has_active_goal(self) -> bool:
        """Return whether the builder currently has an ore or scouting destination."""
        return self.target_ore is not None or self.scout_target is not None

    def observe_tiles(self, controller: Controller) -> None:
        """Refresh known terrain, buildings, conveyor directions, and scout frontier."""
        # Buildings can change on every turn, so any previous transport graph
        # snapshot is no longer authoritative.
        self.connected_network_cache = None
        for pos in controller.get_nearby_tiles():
            env = controller.get_tile_env(pos)
            old_env = self.known_env.get(pos)
            first_observation = pos not in self.observed_tiles
            self.observed_tiles.add(pos)
            self.known_env[pos] = env
            self.inferred_ores.pop(pos, None)
            self.reported_ores.pop(pos, None)
            if first_observation:
                self.update_scout_frontier(pos)
            if old_env != env and env in ORE_TYPES:
                self.infer_symmetric(pos, env)
            building_id = controller.get_tile_building_id(pos)
            if building_id is None:
                self.known_buildings[pos] = None
                self.known_conveyor_directions.pop(pos, None)
                self.known_bridge_targets.pop(pos, None)
            else:
                building_type = controller.get_entity_type(building_id)
                self.known_buildings[pos] = (building_type, controller.get_team(building_id))
                if building_type in {EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR}:
                    self.known_conveyor_directions[pos] = controller.get_direction(building_id)
                else:
                    self.known_conveyor_directions.pop(pos, None)
                if building_type == EntityType.BRIDGE:
                    self.known_bridge_targets[pos] = controller.get_bridge_target(building_id)
                else:
                    self.known_bridge_targets.pop(pos, None)

    def read_ore_markers(self, controller: Controller) -> None:
        """Read core orders and shared ore hints from nearby marker buildings."""
        records: list[tuple[int, Position, int]] = []
        for entity_id in controller.get_nearby_entities():
            if controller.get_entity_type(entity_id) != EntityType.MARKER:
                continue
            try:
                kind, pos, payload = decode_marker(controller.get_marker_value(entity_id))
            except Exception:
                continue
            records.append((kind, pos, payload))

        # A newly created builder has no durable memory yet.  The core writes
        # one handoff marker in the same round as spawn, which is necessary
        # when a map edge forces it to use a non-cardinal core tile.
        if self.work_direction is None:
            for kind, pos, payload in records:
                if kind not in {
                    MARKER_KIND_SPAWN_DIRECTION,
                    MARKER_KIND_SPAWN_ORE_TI,
                    MARKER_KIND_SPAWN_ORE_AX,
                }:
                    continue
                direction = BUILDER_CODE_DIRECTIONS.get(payload)
                if direction is None:
                    continue
                self.work_direction = direction
                if kind == MARKER_KIND_SPAWN_ORE_TI:
                    self.assigned_ores[pos] = Environment.ORE_TITANIUM
                elif kind == MARKER_KIND_SPAWN_ORE_AX:
                    self.assigned_ores[pos] = Environment.ORE_AXIONITE
                break

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
        """Record the mirrored position as an exploration hint for observed ore."""
        mirror = Position(self.map_width - 1 - pos.x, self.map_height - 1 - pos.y)
        if mirror not in self.observed_tiles:
            # Symmetry gives a useful exploration hint, but it is not enough
            # evidence to construct a harvester or a conveyor branch there.
            self.inferred_ores[mirror] = env

    def update_scout_frontier(self, known_pos: Position) -> None:
        """Update unknown boundary cells adjacent to a newly known tile."""
        self.scout_frontier_initialized = True
        self.scout_frontier.discard(known_pos)
        for direction in DIRECTIONS:
            probe = known_pos.add(direction)
            if not self.in_bounds(probe):
                continue
            if probe in self.known_env:
                self.scout_frontier.discard(probe)
            else:
                self.scout_frontier.add(probe)

    def rebuild_scout_frontier(self) -> None:
        """Reconstruct the exploration frontier from all observed tiles."""
        for known_pos in self.observed_tiles:
            self.update_scout_frontier(known_pos)

    def find_home_core(self, controller: Controller) -> Position | None:
        """Locate the friendly core while it is within local vision."""
        for entity_id in controller.get_nearby_entities():
            if controller.get_entity_type(entity_id) == EntityType.CORE:
                return controller.get_position(entity_id)
        current = controller.get_position()
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                probe = Position(current.x + dx, current.y + dy)
                if not self.in_bounds(probe):
                    continue
                building_id = controller.get_tile_building_id(probe)
                if building_id is not None and controller.get_entity_type(building_id) == EntityType.CORE:
                    return controller.get_position(building_id)
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
            controller.build_harvester(ore)
            self.record_harvester_built(ore)
            return True
        return False

    def record_harvester_built(self, ore: Position) -> None:
        """Update local state after successfully placing a harvester on ``ore``."""
        self.known_buildings[ore] = (EntityType.HARVESTER, self.team)
        self.harvester_built = True
        self.last_progress_round = self.rounds_alive
        self.harvester_fail_count = 0
        self.skipped_ores.discard(ore)
        self.assigned_ores.pop(ore, None)
        self.reported_ores.pop(ore, None)
        self.deferred_ores_until.pop(ore, None)
        self.pending_network_ores.add(ore)

    def defer_ore_for_survey(self, ore: Position) -> None:
        """Temporarily resume exploration until a fully known branch exists."""
        self.deferred_ores_until[ore] = self.current_round + 12

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
        self.connection_anchor = None
        self.connection_survey_heading = None
        self.connection_survey_last_pos = None
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
            self.traversable_for_planning,
            movement_directions=DIRECTIONS,
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
            # The harvester already exists, so retain ownership of this job
            # and retry after new terrain/network information arrives.
            self.target_ore = ore
            self.target_is_connection = True
            self.scout_target = None
            self.path = []
            self.path_index = 0
            self.conveyor_path_tiles = set()
            self.conveyor_directions = {}
            self.bridge_targets = {}
            self.connection_anchor = None
            self.mode = "connect"
            self.replan_after_yield = True

    def replan_active_ore_target(self, controller: Controller) -> bool:
        """Recompute the path for the active ore job after a route becomes invalid."""
        if self.target_ore is None:
            return False
        ore = self.target_ore
        connecting = self.target_is_connection
        if self.assign_ore_target(controller, controller.get_position(), ore, connecting):
            return True
        if connecting:
            self.pending_network_ores.add(ore)
            # Do not abandon an already harvested ore if a just-observed
            # obstacle invalidates its old route.  Keeping the target lets a
            # later replan use a newly completed allied branch.
            self.target_ore = ore
            self.target_is_connection = True
            self.scout_target = None
            self.path = []
            self.path_index = 0
            self.conveyor_path_tiles = set()
            self.conveyor_directions = {}
            self.bridge_targets = {}
            self.connection_anchor = None
            self.mode = "connect"
            return False
        self.clear_ore_target()
        return False

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
            if self.is_harvester_on_tile(ore) and self.ore_network_needs_work(ore)
        }

    def ore_action_approaches(self, ore: Position) -> list[Position]:
        """Return movement tiles from which a builder can act on an ore deposit."""
        return [
            ore.add(direction)
            for direction in DIRECTIONS
            if self.traversable_for_planning(None, ore.add(direction))
        ]

    def core_receiver_tiles(self) -> set[Position]:
        """Return in-bounds tiles that deliver resources directly into the core."""
        if self.core_pos is None:
            return set()
        return {
            Position(self.core_pos.x + dx, self.core_pos.y + dy)
            for dx in range(-1, 2)
            for dy in range(-1, 2)
            if self.in_bounds(Position(self.core_pos.x + dx, self.core_pos.y + dy))
        }

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
            if (
                building is None
                or building[1] != self.team
                or building[0] not in {EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR}
            ):
                continue
            direction = self.known_conveyor_directions.get(pos)
            if direction is None:
                continue
            incoming.setdefault(pos.add(direction), []).append(pos)
        for pos, target in self.known_bridge_targets.items():
            building = self.known_buildings.get(pos)
            if building is not None and building == (EntityType.BRIDGE, self.team):
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
        direction = self.known_conveyor_directions.get(receiver)
        return direction is not None and direction != receiver.direction_to(source)

    def harvester_is_connected(self, ore: Position) -> bool:
        """Return whether a harvester has a directed allied conveyor path to core."""
        network = self.known_connected_network()
        for direction in ORTHOGONAL_DIRECTIONS:
            receiver = ore.add(direction)
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
        """Build the cheapest admissible Steiner branch from ore to the tree."""
        network = self.known_connected_network()
        if not network:
            return None
        new_conveyor_cost = controller.get_conveyor_cost()[0]

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
            if building is not None and building[1] == self.team:
                if building[0] in {EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR}:
                    return 0
            return new_conveyor_cost

        conveyor_plan = incremental_steiner_branch(
            starts=self.buildable_approaches(ore),
            tree=network,
            directions=ORTHOGONAL_DIRECTIONS,
            can_use_tile=can_use_tile,
            can_use_edge=can_use_edge,
            receiver_accepts=self.network_receiver_accepts,
            tile_cost=tile_cost,
            max_expansions=STEINER_MAX_EXPANSIONS,
        )
        best = None
        if conveyor_plan is not None:
            approach, tiles, directions, anchor = conveyor_plan
            branch_cost = sum(tile_cost(tile, directions[tile]) for tile in tiles)
            best = (branch_cost, (approach, tiles, directions, {}, anchor))

        # A bridge is deliberately considered only as a terminal edge into the
        # established tree.  This yields the useful wall/gap crossing without
        # making the builder traverse an unreachable landing to finish a
        # partially constructed branch.  Its real titanium price and severe
        # +10% scaling make an ordinary Steiner branch win on open ground.
        bridge_cost = controller.get_bridge_cost()[0]
        for approach in self.buildable_approaches(ore):
            building = self.known_buildings.get(approach)
            if building is not None and building[0] != EntityType.ROAD:
                continue
            for anchor in network:
                if approach.distance_squared(anchor) > 9:
                    continue
                candidate = (
                    bridge_cost,
                    (approach, [approach], {}, {approach: anchor}, anchor),
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
        return None if best is None else best[1]

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
                movement_directions=DIRECTIONS,
            )
            if not path_to_approach and current != approach:
                return False
        path = list(path_to_approach)
        for tile in build_tiles:
            if not path or path[-1] != tile:
                path.append(tile)
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
        self.connection_survey_heading = None
        self.connection_survey_last_pos = None
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
        return building is None or building[0] in PASSABLE_BUILDINGS

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
            return False
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
            ore_pos.add(direction)
            for direction in ORTHOGONAL_DIRECTIONS
            if (
                self.is_core_receiver_tile(ore_pos.add(direction))
                or self.traversable_for_connection(ore_pos.add(direction))
            )
        ]

    def work_direction_progress(self, pos: Position) -> int:
        """Measure signed progress from the core along this builder's sector direction."""
        if self.core_pos is None or self.work_direction is None:
            return 0
        forward = self.core_pos.add(self.work_direction)
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
        forward = self.core_pos.add(self.work_direction)
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
            if len(cycle) >= 3:
                avoid_until = self.current_round + SCOUT_DEAD_END_AVOID_ROUNDS
                for cycle_pos in cycle:
                    self.scout_avoid_until[cycle_pos] = max(
                        avoid_until,
                        self.scout_avoid_until.get(cycle_pos, 0),
                    )
                if self.target_ore is None:
                    self.scout_cycle_replan = True
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
        """Return the additional A* cost for revisiting or moving back toward core."""
        revisit_cost = self.recent_route_visits.get(pos, 0) * SCOUT_REVISIT_STEP_PENALTY
        revisit_cost += min(self.scout_total_visits.get(pos, 0), 6) * SCOUT_PERSISTENT_REVISIT_PENALTY
        if self.scout_avoid_until.get(pos, 0) > self.current_round:
            revisit_cost += SCOUT_DEAD_END_PENALTY
        inward_steps = max(0, self.core_distance(origin) - self.core_distance(pos))
        return revisit_cost + inward_steps * SCOUT_INWARD_STEP_PENALTY

    def newly_visible_tiles(self, centre: Position) -> int:
        """Count unknown tiles that would enter vision from ``centre``."""
        visible = 0
        for dx, dy in SCOUT_VISION_OFFSETS:
            pos = Position(centre.x + dx, centre.y + dy)
            if self.in_bounds(pos) and pos not in self.known_env:
                visible += 1
        return visible

    def ore_hint_progress(self, current: Position, candidate: Position) -> int:
        """Measure how much a candidate step approaches an inferred or reported ore hint."""
        hints = set(self.inferred_ores) | set(self.reported_ores)
        if not hints:
            return 0
        current_distance = min(
            max(abs(current.x - hint.x), abs(current.y - hint.y))
            for hint in hints
        )
        candidate_distance = min(
            max(abs(candidate.x - hint.x), abs(candidate.y - hint.y))
            for hint in hints
        )
        return max(0, current_distance - candidate_distance)

    def scout_frontier_pre_score(self, current: Position, candidate: Position) -> tuple[int, int]:
        """Score a frontier cheaply before calculating expensive visibility details."""
        forward_progress = self.work_direction_progress(candidate)
        in_sector = int(self.work_direction is None or forward_progress > 0)
        returning = max(0, self.core_distance(current) - self.core_distance(candidate))
        distance = max(abs(candidate.x - current.x), abs(candidate.y - current.y))
        return in_sector, (
            forward_progress * SCOUT_FORWARD_PROGRESS_WEIGHT
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
            forward_progress * SCOUT_FORWARD_PROGRESS_WEIGHT
            + self.newly_visible_tiles(candidate) * SCOUT_NEW_VISION_WEIGHT
            - returning * SCOUT_RETURN_TO_BASE_WEIGHT
            - distance * SCOUT_DISTANCE_WEIGHT
            - self.work_direction_lateral_offset(candidate) * SCOUT_LATERAL_DEVIATION_WEIGHT
            + self.ore_hint_progress(current, candidate) * SCOUT_ORE_HINT_PROGRESS_WEIGHT
            + self.snake_sweep_bias(current, candidate)
        )
        return in_sector, score

    def select_new_target(self, controller: Controller) -> None:
        """Choose work in priority order: connections, mineable ore, then scouting."""
        was_idle = not self.has_active_goal()
        self.harvester_built = False
        self.harvester_fail_count = 0
        self.clear_ore_target()
        self.scout_target = None
        self.scout_target_direct = False
        self.scout_retry_pending = False
        current = controller.get_position()
        ore_sort_key = lambda pos: (
            0 if pos in self.assigned_ores else 1,
            self.work_direction_priority(pos),
            current.distance_squared(pos),
        )

        for ore in sorted(self.connection_candidates(), key=ore_sort_key):
            if not self.assign_ore_target(controller, current, ore, connecting=True):
                continue
            if was_idle:
                self.stuck_rounds = 0
                self.last_progress_round = self.rounds_alive
            return

        for ore in sorted(set(self.mineable_ores()), key=ore_sort_key):
            if self.is_harvester_on_tile(ore) or ore in self.skipped_ores:
                continue
            if not self.assign_ore_target(controller, current, ore, connecting=False):
                continue
            if was_idle:
                self.stuck_rounds = 0
                self.last_progress_round = self.rounds_alive
            return

        scout_plan = self.choose_scout_plan(controller)
        if scout_plan is not None:
            self.scout_target, self.path = scout_plan
            self.scout_target_direct = False
            self.path_index = 0
            self.mode = "scout"
            if was_idle:
                self.stuck_rounds = 0
                self.last_progress_round = self.rounds_alive
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
            if was_idle:
                self.stuck_rounds = 0
                self.last_progress_round = self.rounds_alive
            return
        self.scout_retry_pending = True

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

    def scout_a_star_traversable(self, pos: Position, targets: set[Position]) -> bool:
        """Restrict scout A* to known terrain plus selected frontier endpoints."""
        if not self.traversable_for_planning(None, pos):
            return False
        # A* may use the known map and enter its one frontier endpoint, but it
        # must not invent an entire route through terrain it has never seen.
        return pos in targets or pos in self.known_env or self.is_core_receiver_tile(pos)

    def scout_step_is_viable(self, pos: Position) -> bool:
        """Return whether a one-step right-hand scouting move may use ``pos``."""
        return self.traversable_for_planning(None, pos)

    def choose_right_hand_scout_step(self, current: Position) -> Position | None:
        """Choose the least-visited viable escape, permitting true backtracking."""
        if self.work_direction is None:
            return None
        candidates = []
        for direction in DIRECTIONS:
            candidate = current.add(direction)
            if not self.scout_step_is_viable(candidate):
                continue
            active_avoid = int(self.scout_avoid_until.get(candidate, 0) > self.current_round)
            candidates.append((
                active_avoid,
                self.scout_total_visits.get(candidate, 0),
                self.recent_route_visits.get(candidate, 0),
                -self.newly_visible_tiles(candidate),
                -self.work_direction_progress(candidate),
                direction,
                candidate,
            ))
        if not candidates:
            return None
        *_, direction, candidate = min(candidates, key=lambda item: item[:-2])
        self.scout_heading = direction
        return candidate

    def survey_for_connection(self, controller: Controller) -> bool:
        """Reveal a detour when a live branch reaches newly seen terrain.

        A preflighted branch normally uses only the internal known map.  A
        rival can nevertheless occupy one of its future cells while the
        builder is walking there.  In that case keep the harvester job and
        take one safe scouting step around the obstruction, then try A* again
        with the newly visible cells instead of idling beside the break.
        """
        if self.core_pos is None:
            return False
        current = controller.get_position()
        forward = current.direction_to(self.core_pos)
        if forward == Direction.CENTRE:
            return False

        def can_survey_step(direction: Direction, allow_backtrack: bool = False) -> bool:
            """Check whether one detour step can safely reveal new branch terrain."""
            candidate = current.add(direction)
            if (
                not self.in_bounds(candidate)
                or candidate in self.permanently_blocked
                or candidate in self.unfinished_branch_tiles
                or (not allow_backtrack and candidate == self.connection_survey_last_pos)
                or not controller.can_move(direction)
            ):
                return False
            env = self.known_env.get(candidate)
            return env != Environment.WALL and env not in ORE_TYPES

        # First resume the direct line toward the core.  If it is closed,
        # retain a clockwise wall-following heading, so the survey actually
        # walks around an obstruction instead of oscillating between two
        # equally fresh cells.
        if can_survey_step(forward):
            direction = forward
        else:
            direction = self.connection_survey_heading or forward
            for _ in range(7):
                direction = direction.rotate_right()
                if can_survey_step(direction):
                    break
            else:
                # A genuine dead end may require one reverse step, but only
                # after every non-backtracking wall-following option failed.
                direction = self.connection_survey_heading or forward
                for _ in range(7):
                    direction = direction.rotate_right()
                    if can_survey_step(direction, allow_backtrack=True):
                        break
                else:
                    return False
        controller.move(direction)
        self.connection_survey_heading = direction
        self.connection_survey_last_pos = current
        self.stuck_rounds = 0
        self.last_progress_round = self.rounds_alive
        return True

    def choose_scout_plan(self, controller: Controller) -> tuple[Position, list[Position]] | None:
        """Find a bounded-cost path to a reachable, high-value frontier."""
        if not self.scout_frontier and self.known_env and not self.scout_frontier_initialized:
            self.rebuild_scout_frontier()
        current = controller.get_position()
        candidates = [
            pos for pos in self.scout_frontier
            if (
                pos not in self.known_env
                and pos not in self.permanently_blocked
                and pos not in self.unreachable_scout_targets
            )
        ]
        candidates.sort(key=lambda pos: self.scout_frontier_pre_score(current, pos), reverse=True)
        candidates = candidates[:SCOUT_FRONTIER_CANDIDATE_LIMIT]
        candidates.sort(key=lambda pos: self.scout_frontier_score(current, pos), reverse=True)
        # Search small batches so an unreachable, attractive frontier cannot
        # monopolise every turn.  Multi-goal A* returns whichever endpoint is
        # cheapest after revisit/dead-end penalties.
        for index in range(0, len(candidates), 4):
            targets = set(candidates[index:index + 4])
            path = a_star_to_any(
                controller,
                current,
                targets,
                lambda _controller, pos: self.scout_a_star_traversable(pos, targets),
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
        """Drop a cyclic or stalled route and allow an immediate fresh search."""
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
        current = controller.get_position()
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
            if self.ensure_tree_conveyor(controller, next_pos):
                return
            if not self.has_expected_tree_conveyor(controller, next_pos):
                self.replan_or_wait_for_connection_tile(controller, next_pos)
                return
        if not controller.is_tile_passable(next_pos):
            self.try_prepare_tile(controller, next_pos)
        if controller.can_move(direction):
            controller.move(direction)
            self.stuck_rounds = 0
            self.last_progress_round = self.rounds_alive
        elif not controller.is_tile_passable(next_pos):
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
        next_pos = current.add(direction)
        if not controller.is_tile_passable(next_pos):
            self.try_prepare_tile(controller, next_pos)
        if not controller.can_move(direction):
            return False
        controller.move(direction)
        self.scout_heading = direction
        self.stuck_rounds = 0
        self.last_progress_round = self.rounds_alive
        return True

    def try_prepare_tile(self, controller: Controller, target: Position) -> None:
        """Prepare a blocked next tile by building a road or branch conveyor."""
        if self.target_is_connection and target in self.conveyor_path_tiles:
            self.ensure_tree_conveyor(controller, target)
            return
        if controller.get_tile_builder_bot_id(target) is not None:
            return
        if controller.can_build_road(target):
            controller.build_road(target)
            self.last_progress_round = self.rounds_alive
            return
        self.mark_staticly_blocked(target)

    def replan_after_blocked_step(self, controller: Controller, target: Position) -> None:
        """Yield or recompute the active job when the next path tile is blocked."""
        # is_tile_passable() is also false while another builder occupies an
        # otherwise usable road.  For allied builders, entity IDs reflect spawn
        # order: the later builder yields, while the earlier one keeps priority.
        blocking_id = controller.get_tile_builder_bot_id(target)
        if blocking_id is not None:
            if controller.get_team(blocking_id) == self.team:
                if blocking_id < controller.get_id():
                    self.yield_to_higher_priority_builder(controller, target)
                return
            self.schedule_replan_after_yield()
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
        current = controller.get_position()
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
            current.add(direction)
            for direction in DIRECTIONS
            if (
                current.add(direction) != blocking_pos
                and current.add(direction) not in route_history
                and self.in_bounds(current.add(direction))
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
            self.stuck_rounds = 0
            self.last_progress_round = self.rounds_alive
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
        """Build or safely reuse the planned conveyor or terminal bridge."""
        bridge_target = self.bridge_targets.get(target)
        if bridge_target is not None:
            building_id = controller.get_tile_building_id(target)
            if building_id is not None:
                building_type = controller.get_entity_type(building_id)
                if (
                    building_type == EntityType.BRIDGE
                    and controller.get_team(building_id) == self.team
                    and controller.get_bridge_target(building_id) == bridge_target
                ):
                    self.known_bridge_targets[target] = bridge_target
                    return False
                if (
                    building_type == EntityType.ROAD
                    and controller.get_team(building_id) == self.team
                    and controller.can_destroy(target)
                ):
                    controller.destroy(target)
                    self.known_buildings[target] = None
                    self.connected_network_cache = None
                    return True
                return False
            if controller.can_build_bridge(target, bridge_target):
                controller.build_bridge(target, bridge_target)
                self.known_buildings[target] = (EntityType.BRIDGE, self.team)
                self.known_bridge_targets[target] = bridge_target
                self.unfinished_branch_tiles.add(target)
                self.connected_network_cache = None
                self.last_progress_round = self.rounds_alive
                return True
            return False

        conveyor_direction = self.get_conveyor_direction(target)
        building_id = controller.get_tile_building_id(target)
        if building_id is not None:
            building_type = controller.get_entity_type(building_id)
            if (
                building_type in {EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR}
                and controller.get_team(building_id) == self.team
                and controller.get_direction(building_id) == conveyor_direction
            ):
                self.known_conveyor_directions[target] = conveyor_direction
                return False
            if building_type == EntityType.MARKER:
                # Order-board markers are deliberately reserved cells.
                self.permanently_blocked.add(target)
                return False
            if (
                building_type == EntityType.ROAD
                and controller.get_team(building_id) == self.team
                and controller.can_destroy(target)
            ):
                controller.destroy(target)
                self.known_buildings[target] = None
                self.known_conveyor_directions.pop(target, None)
                self.connected_network_cache = None
                return True
            if (
                building_type in {EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR}
                and controller.get_team(building_id) == self.team
                and target in self.unfinished_branch_tiles
                and target not in self.known_connected_network()
                and controller.can_destroy(target)
            ):
                controller.destroy(target)
                self.known_buildings[target] = None
                self.known_conveyor_directions.pop(target, None)
                self.connected_network_cache = None
                return True
            return False

        if controller.can_build_conveyor(target, conveyor_direction):
            controller.build_conveyor(target, conveyor_direction)
            self.known_buildings[target] = (EntityType.CONVEYOR, self.team)
            self.known_conveyor_directions[target] = conveyor_direction
            self.unfinished_branch_tiles.add(target)
            self.connected_network_cache = None
            self.last_progress_round = self.rounds_alive
            return True
        return False

    def has_expected_tree_conveyor(self, controller: Controller, target: Position) -> bool:
        """Check that ``target`` contains the planned allied transport building."""
        building_id = controller.get_tile_building_id(target)
        if building_id is None or controller.get_team(building_id) != self.team:
            return False
        bridge_target = self.bridge_targets.get(target)
        if bridge_target is not None:
            return (
                controller.get_entity_type(building_id) == EntityType.BRIDGE
                and controller.get_bridge_target(building_id) == bridge_target
            )
        if controller.get_entity_type(building_id) not in {
            EntityType.CONVEYOR,
            EntityType.ARMOURED_CONVEYOR,
        }:
            return False
        return controller.get_direction(building_id) == self.get_conveyor_direction(target)

    def replan_or_wait_for_connection_tile(self, controller: Controller, target: Position) -> None:
        """Never step past a branch tile that was not successfully built."""
        if target not in self.bridge_targets and self.get_conveyor_direction(target) == Direction.CENTRE:
            self.schedule_replan_after_yield()
            return
        if controller.get_tile_building_id(target) is None:
            # Usually an action cooldown or a temporary resource shortage.
            # Stay on the branch and try the exact same tile next turn.
            return
        # A foreign/indestructible building occupies the planned branch tile.
        # Exclude it from the next A* search instead of walking through it and
        # falsely declaring the harvester connected.
        self.permanently_blocked.add(target)
        self.schedule_replan_after_yield()

    def get_conveyor_direction(self, target: Position) -> Direction:
        """Return the planned outgoing direction for a branch conveyor tile."""
        direction = self.conveyor_directions.get(target)
        if direction is not None:
            return direction
        return Direction.CENTRE
