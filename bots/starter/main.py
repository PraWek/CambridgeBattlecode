from collections import deque

from cambc import Controller, Direction, EntityType, Environment, GameConstants, Position

from bot.constants import (
    DIRECTIONS,
    MARKER_KIND_ENEMY,
    MARKER_KIND_PHASE,
    MARKER_KIND_ORE_TI,
    MARKER_KIND_ORE_AX,
    MAX_BUILDERS_PHASE_ONE,
    ORE_TYPES,
    ORTHOGONAL_DIRECTIONS,
    PASSABLE_BUILDINGS,
    PHASE_BOOTSTRAP,
    PHASE_EXPAND_AXIONITE,
    PHASE_EXPAND_TITANIUM,
    PHASE_STABILIZE,
    RESOURCE_AXIONITE,
    RESOURCE_TITANIUM,
    STUCK_KILL_ROUNDS,
    MAX_IDLE_ROUNDS,
    TITANIUM_LINE_READY_HARVESTERS,
    GUNNER_RESERVE,
)

from bot.geometry import decode_marker, encode_marker, in_bounds
from bot.navigation import a_star_to_any
from bot.steiner import compute_steiner_tree
from bot.strategy import choose_phase


BUILDER_WORK_DIRECTIONS = (
    Direction.NORTH,
    Direction.EAST,
    Direction.SOUTH,
    Direction.WEST,
)

SCOUT_ROUTE_MEMORY_TILES = 24
SCOUT_REVISIT_STEP_PENALTY = 12
SCOUT_INWARD_STEP_PENALTY = 2
SCOUT_FORWARD_PROGRESS_WEIGHT = 6
SCOUT_NEW_VISION_WEIGHT = 8
SCOUT_RETURN_TO_BASE_WEIGHT = 10
SCOUT_DISTANCE_WEIGHT = 2
SCOUT_LATERAL_DEVIATION_WEIGHT = 2
SCOUT_FRONTIER_CANDIDATE_LIMIT = 12
_BUILDER_VISION_RADIUS = int(GameConstants.BUILDER_BOT_VISION_RADIUS_SQ ** 0.5)
SCOUT_VISION_OFFSETS = tuple(
    (dx, dy)
    for dx in range(-_BUILDER_VISION_RADIUS, _BUILDER_VISION_RADIUS + 1)
    for dy in range(-_BUILDER_VISION_RADIUS, _BUILDER_VISION_RADIUS + 1)
    if dx * dx + dy * dy <= GameConstants.BUILDER_BOT_VISION_RADIUS_SQ
)



class Player:

    def __init__(self) -> None:
        self.initialized = False

        self.core_pos = None
        self.enemy_estimate = None
        self.work_direction: Direction | None = None
        self.enemy_marker_pad = None
        self.phase_marker_pad = None
        self.map_width = 0
        self.map_height = 0
        self.team = None

        self.known_env = {}
        self.known_buildings = {}
        self.scout_frontier: set[Position] = set()
        self.scout_frontier_initialized = False

        self.spawned_builders = 0
        self.spawned_work_directions: set[Direction] = set()

        self.target_ore = None
        self.target_resource = RESOURCE_TITANIUM
        self.scout_target = None
        self.path: list[Position] = []
        self.path_index = 0
        self.harvester_built = False
        self.role = "bootstrap"
        self.titanium_harvesters_built: int = 0
        self.harvester_fail_count: int = 0
        self.skipped_ores: set[Position] = set()
        self.next_select_round: int = 0

        # parent[pos] = parent_pos — дерево Штейнера
        self.steiner_parent: dict[Position, Position] = {}
        self.steiner_ores_key: frozenset = frozenset()

        self.permanently_blocked: set[Position] = set()

        # Трансляция руды через маркеры
        self._ore_broadcast_idx: int = 0

        # Emergency kill: отслеживание прогресса
        self._last_pos: Position | None = None
        self._stuck_rounds: int = 0
        self._rounds_alive: int = 0
        self._last_progress_round: int = 0
        self.recent_route: deque[Position] = deque()
        self.recent_route_visits: dict[Position, int] = {}

    def run(self, ct: Controller) -> None:
        entity_type = ct.get_entity_type()

        if entity_type == EntityType.CORE:
            self.run_core(ct)
        elif entity_type == EntityType.BUILDER_BOT:
            self.run_builder(ct)
        elif entity_type == EntityType.GUNNER:
            self.run_gunner(ct)

    def init_map_state(self, ct: Controller) -> None:
        if self.initialized:
            return
        self.initialized = True
        self.map_width = ct.get_map_width()
        self.map_height = ct.get_map_height()
        self.team = ct.get_team()
        self.observe_tiles(ct)


    def run_core(self, ct: Controller) -> None:
        self.init_map_state(ct)
        self.observe_tiles(ct)
        self.core_pos = ct.get_position()
        self.enemy_estimate = Position(
            ct.get_map_width() - 1 - self.core_pos.x,
            ct.get_map_height() - 1 - self.core_pos.y,
        )

        if self.enemy_marker_pad is None or self.phase_marker_pad is None:
            self.enemy_marker_pad, self.phase_marker_pad = self.find_marker_pads(ct, self.core_pos)

        titanium_harvesters = self.count_harvesters(self.known_titanium_ores(), allied_only=True, limit=4)
        axionite_harvesters = self.count_harvesters(self.known_axionite_ores(), allied_only=True, limit=2)
        phase = choose_phase(ct, titanium_harvesters, axionite_harvesters)

        self.place_core_markers(ct, phase)
        self.broadcast_ore(ct)
        self.try_spawn_builder(ct)


    def run_builder(self, ct: Controller) -> None:
        self.init_map_state(ct)
        self._rounds_alive += 1

        cur_pos = ct.get_position()

        if self._last_pos is not None and cur_pos == self._last_pos:
            self._stuck_rounds += 1
        else:
            self._stuck_rounds = 0
            self._last_progress_round = self._rounds_alive
            self.remember_route_position(cur_pos)
        self._last_pos = cur_pos

        if self._stuck_rounds >= STUCK_KILL_ROUNDS:
            ct.self_destruct()
            return

        if self._rounds_alive - self._last_progress_round > MAX_IDLE_ROUNDS:
            ct.self_destruct()
            return

        self.observe_tiles(ct)
        self.read_ore_markers(ct)

        if self.core_pos is None:
            self.core_pos = self.find_home_core(ct)
        if self.core_pos is None:
            return
        if self.enemy_estimate is None:
            self.enemy_estimate = Position(
                ct.get_map_width() - 1 - self.core_pos.x,
                ct.get_map_height() - 1 - self.core_pos.y,
            )
        if self.work_direction is None:
            self.work_direction = self.core_pos.direction_to(cur_pos)

        self._ensure_steiner_tree(ct)
        self.update_role_from_phase_marker(ct)

        current_round = ct.get_current_round()
        need_new_target = False
        if self.harvester_built:
            self._last_progress_round = self._rounds_alive
            need_new_target = True
        elif self.target_ore is not None and self.is_harvester_on_tile(self.target_ore):
            need_new_target = True
        elif self.target_ore is None and self.scout_target is None and current_round >= self.next_select_round:
            need_new_target = True
        elif self.target_ore is None and self.scout_target is not None and self.mineable_ores_for_role():
            need_new_target = True

        if need_new_target:
            self.select_new_target(ct)
            if self.target_ore is None and self.scout_target is None:
                self.next_select_round = current_round + 15

        if self.target_ore is None and self.scout_target is None:
            return

        if self.target_ore is not None and ct.get_position().distance_squared(
                self.target_ore) <= GameConstants.ACTION_RADIUS_SQ:
            if ct.can_build_harvester(self.target_ore):
                ct.build_harvester(self.target_ore)
                self.harvester_built = True
                self._last_progress_round = self._rounds_alive
                self.harvester_fail_count = 0
                self.skipped_ores.discard(self.target_ore)
                if self.target_resource == RESOURCE_TITANIUM:
                    self.titanium_harvesters_built += 1
                self.target_ore = None
                self.scout_target = None
                self.path = []
                self.path_index = 0
                return
            else:
                self.harvester_fail_count += 1
                if self.harvester_fail_count >= 5:
                    self.skipped_ores.add(self.target_ore)
                    self.harvester_fail_count = 0
                    self.target_ore = None
                    self.path = []
                    self.path_index = 0
                    self.select_new_target(ct)
                    return

        self.follow_path_and_build(ct)


    def run_gunner(self, ct: Controller) -> None:
        self.observe_tiles(ct)
        target = ct.get_gunner_target()
        if target is not None and ct.can_fire(target):
            ct.fire(target)
            return

        marker_target = self.read_enemy_marker_target(ct)
        if marker_target is None:
            return

        desired = ct.get_position().direction_to(marker_target)
        if desired != Direction.CENTRE and desired != ct.get_direction() and ct.can_rotate(desired):
            ct.rotate(desired)


    def _ensure_steiner_tree(self, ct: Controller) -> None:
        if self.core_pos is None:
            return
        ores = self.known_titanium_ores() + self.known_axionite_ores()
        key = frozenset(ores)
        if key == self.steiner_ores_key:
            return

        self.steiner_ores_key = key
        self.steiner_parent = compute_steiner_tree(
            self.core_pos,
            ores,
            self.known_env,
            self.map_width,
            self.map_height,
            blocked=self.permanently_blocked,
        )

    def _get_conveyor_direction(self, target: Position) -> Direction:
        steiner_p = self.steiner_parent.get(target)
        if steiner_p is not None:
            d = target.direction_to(steiner_p)
            if d != Direction.CENTRE:
                return d
        return Direction.CENTRE


    def find_marker_pads(self, ct: Controller, core_pos: Position) -> tuple[Position | None, Position | None]:
        pads = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                pos = Position(core_pos.x + dx, core_pos.y + dy)
                if not in_bounds(ct, pos):
                    continue
                if pos.distance_squared(core_pos) > GameConstants.CORE_ACTION_RADIUS_SQ:
                    continue
                if ct.can_place_marker(pos):
                    pads.append(pos)
        if not pads:
            return None, None
        if len(pads) == 1:
            return pads[0], pads[0]
        return pads[0], pads[1]

    def place_core_markers(self, ct: Controller, phase: int) -> None:
        if self.enemy_estimate is None:
            return
        if self.enemy_marker_pad is not None and ct.can_place_marker(self.enemy_marker_pad):
            ct.place_marker(self.enemy_marker_pad, encode_marker(MARKER_KIND_ENEMY, self.enemy_estimate, phase))
        if self.phase_marker_pad is not None and ct.can_place_marker(self.phase_marker_pad):
            ct.place_marker(self.phase_marker_pad, encode_marker(MARKER_KIND_PHASE, self.core_pos, phase))


    def broadcast_ore(self, ct: Controller) -> None:
        """Ядро циклически транслирует известные руды через свободные маркерные тайлы."""
        ti_ores = self.known_titanium_ores()
        ax_ores = self.known_axionite_ores()
        all_ores = [(o, MARKER_KIND_ORE_TI) for o in ti_ores] + [(o, MARKER_KIND_ORE_AX) for o in ax_ores]
        if not all_ores:
            return

        self._ore_broadcast_idx = self._ore_broadcast_idx % len(all_ores)
        ore_pos, kind = all_ores[self._ore_broadcast_idx]
        self._ore_broadcast_idx += 1

        core_pos = ct.get_position()
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                pad = Position(core_pos.x + dx, core_pos.y + dy)
                if pad == self.enemy_marker_pad or pad == self.phase_marker_pad:
                    continue
                if not in_bounds(ct, pad):
                    continue
                if pad.distance_squared(core_pos) > GameConstants.CORE_ACTION_RADIUS_SQ:
                    continue
                if ct.can_place_marker(pad):
                    ct.place_marker(pad, encode_marker(kind, ore_pos, 0))
                    return

    def read_ore_markers(self, ct: Controller) -> None:
        """Билдер читает маркеры руды, добавляет их в known_env и выводит зеркальные"""
        for entity_id in ct.get_nearby_entities():
            if ct.get_entity_type(entity_id) != EntityType.MARKER:
                continue
            try:
                kind, pos, _ = decode_marker(ct.get_marker_value(entity_id))
            except Exception:
                continue
            if kind == MARKER_KIND_ORE_TI:
                if pos not in self.known_env:
                    self.known_env[pos] = Environment.ORE_TITANIUM
                    self.update_scout_frontier(pos)
                    self._infer_symmetric(pos, Environment.ORE_TITANIUM)
            elif kind == MARKER_KIND_ORE_AX:
                if pos not in self.known_env:
                    self.known_env[pos] = Environment.ORE_AXIONITE
                    self.update_scout_frontier(pos)
                    self._infer_symmetric(pos, Environment.ORE_AXIONITE)

    def _infer_symmetric(self, pos: Position, env: Environment) -> None:
        """Вывод симметричной позиции руды (карта гарантированно симметрична)"""
        if self.map_width == 0 or self.map_height == 0:
            return
        mirror = Position(self.map_width - 1 - pos.x, self.map_height - 1 - pos.y)
        if mirror not in self.known_env:
            self.known_env[mirror] = env
            self.update_scout_frontier(mirror)


    def try_spawn_builder(self, ct: Controller) -> None:
        if ct.get_unit_count() >= GameConstants.MAX_TEAM_UNITS:
            return
        if self.spawned_builders >= MAX_BUILDERS_PHASE_ONE:
            return

        for direction in BUILDER_WORK_DIRECTIONS:
            if direction in self.spawned_work_directions:
                continue
            if self.spawn_in_direction(ct, direction):
                self.spawned_work_directions.add(direction)
                return

    def spawn_in_direction(self, ct: Controller, preferred: Direction) -> bool:
        core_pos = ct.get_position()
        spawn_pos = core_pos.add(preferred)
        if not ct.can_spawn(spawn_pos):
            return False
        ct.spawn_builder(spawn_pos)
        self.spawned_builders += 1
        return True

    def count_harvesters(self, ores: list[Position], allied_only: bool, limit: int | None = None) -> int:
        count = 0
        for ore in ores:
            if self.is_harvester_on_tile(ore, allied_only=allied_only):
                count += 1
                if limit is not None and count >= limit:
                    return count
        return count

    def find_home_core(self, ct: Controller) -> Position | None:
        for entity_id in ct.get_nearby_entities():
            if ct.get_entity_type(entity_id) == EntityType.CORE:
                return ct.get_position(entity_id)
        pos = ct.get_position()
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                probe = Position(pos.x + dx, pos.y + dy)
                if not in_bounds(ct, probe):
                    continue
                building_id = ct.get_tile_building_id(probe)
                if building_id is not None and ct.get_entity_type(building_id) == EntityType.CORE:
                    return ct.get_position(building_id)
        return None


    def update_role_from_phase_marker(self, ct: Controller) -> None:
        old_role = self.role
        phase = self.read_phase_marker(ct)
        if phase is None:
            return
        if phase == PHASE_BOOTSTRAP:
            self.role = "bootstrap"
        elif phase == PHASE_EXPAND_TITANIUM:
            self.role = "expand_titanium"
        elif phase == PHASE_EXPAND_AXIONITE:
            self.role = "expand_axionite"
        else:
            if self.role == "expand_axionite":
                self.role = "stabilize"
            elif self.role != "stabilize":
                self.role = "expand_titanium"

    def read_phase_marker(self, ct: Controller) -> int | None:
        for entity_id in ct.get_nearby_entities():
            if ct.get_entity_type(entity_id) != EntityType.MARKER:
                continue
            try:
                kind, _, payload = decode_marker(ct.get_marker_value(entity_id))
            except Exception:
                continue
            if kind == MARKER_KIND_PHASE:
                return payload
        return None

    def known_ores_for_role(self) -> list[Position]:
        if self.role == "expand_axionite":
            return self.known_axionite_ores()
        return self.known_titanium_ores()

    def mineable_ores_for_role(self) -> list[Position]:
        return [ore for ore in self.known_ores_for_role() if not self.is_harvester_on_tile(ore)]

    def work_direction_priority(self, pos: Position) -> int:
        if self.core_pos is None or self.work_direction is None:
            return 0
        return 0 if self.work_direction_progress(pos) > 0 else 1

    def work_direction_progress(self, pos: Position) -> int:
        if self.core_pos is None or self.work_direction is None:
            return 0

        forward = self.core_pos.add(self.work_direction)
        direction_x = forward.x - self.core_pos.x
        direction_y = forward.y - self.core_pos.y
        offset_x = pos.x - self.core_pos.x
        offset_y = pos.y - self.core_pos.y
        return offset_x * direction_x + offset_y * direction_y

    def core_distance(self, pos: Position) -> int:
        if self.core_pos is None:
            return 0
        return max(abs(pos.x - self.core_pos.x), abs(pos.y - self.core_pos.y))

    def work_direction_lateral_offset(self, pos: Position) -> int:
        if self.core_pos is None or self.work_direction is None:
            return 0

        forward = self.core_pos.add(self.work_direction)
        direction_x = forward.x - self.core_pos.x
        direction_y = forward.y - self.core_pos.y
        offset_x = pos.x - self.core_pos.x
        offset_y = pos.y - self.core_pos.y
        return abs(offset_x * direction_y - offset_y * direction_x)

    def remember_route_position(self, pos: Position) -> None:
        if self.recent_route and self.recent_route[-1] == pos:
            return

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
        revisit_cost = self.recent_route_visits.get(pos, 0) * SCOUT_REVISIT_STEP_PENALTY
        inward_steps = max(0, self.core_distance(origin) - self.core_distance(pos))
        return revisit_cost + inward_steps * SCOUT_INWARD_STEP_PENALTY

    def newly_visible_tiles(self, centre: Position) -> int:
        visible = 0
        for dx, dy in SCOUT_VISION_OFFSETS:
            x = centre.x + dx
            y = centre.y + dy
            if 0 <= x < self.map_width and 0 <= y < self.map_height:
                if Position(x, y) not in self.known_env:
                    visible += 1
        return visible

    def scout_frontier_score(self, current: Position, candidate: Position) -> tuple[int, int]:
        forward_progress = self.work_direction_progress(candidate)
        in_work_sector = int(self.work_direction is None or forward_progress > 0)
        newly_visible = self.newly_visible_tiles(candidate)
        return_to_base = max(0, self.core_distance(current) - self.core_distance(candidate))
        travel_distance = max(abs(candidate.x - current.x), abs(candidate.y - current.y))
        score = (
            forward_progress * SCOUT_FORWARD_PROGRESS_WEIGHT
            + newly_visible * SCOUT_NEW_VISION_WEIGHT
            - return_to_base * SCOUT_RETURN_TO_BASE_WEIGHT
            - travel_distance * SCOUT_DISTANCE_WEIGHT
            - self.work_direction_lateral_offset(candidate) * SCOUT_LATERAL_DEVIATION_WEIGHT
        )
        return in_work_sector, score

    def scout_frontier_pre_score(self, current: Position, candidate: Position) -> tuple[int, int]:
        forward_progress = self.work_direction_progress(candidate)
        in_work_sector = int(self.work_direction is None or forward_progress > 0)
        return_to_base = max(0, self.core_distance(current) - self.core_distance(candidate))
        travel_distance = max(abs(candidate.x - current.x), abs(candidate.y - current.y))
        score = (
            forward_progress * SCOUT_FORWARD_PROGRESS_WEIGHT
            - return_to_base * SCOUT_RETURN_TO_BASE_WEIGHT
            - travel_distance * SCOUT_DISTANCE_WEIGHT
        )
        return in_work_sector, score


    def select_new_target(self, ct: Controller) -> None:
        self.harvester_built = False
        self.harvester_fail_count = 0
        self.target_ore = None
        self.scout_target = None
        self.path = []
        self.path_index = 0

        if self.role != "expand_axionite" and self.titanium_harvesters_built >= TITANIUM_LINE_READY_HARVESTERS:
            self.role = "expand_axionite"

        self.target_resource = RESOURCE_AXIONITE if self.role == "expand_axionite" else RESOURCE_TITANIUM
        candidates = self.known_axionite_ores() if self.target_resource == RESOURCE_AXIONITE else self.known_titanium_ores()

        current = ct.get_position()
        for ore in sorted(
                candidates,
                key=lambda pos: (self.work_direction_priority(pos), current.distance_squared(pos)),
        ):
            if self.is_harvester_on_tile(ore):
                continue
            if ore in self.skipped_ores:
                continue
            goals = set(self.buildable_approaches(ore))
            if not goals:
                continue
            path = self.path_to_ore_network(ct, current, ore)
            if path is not None:
                self.target_ore = ore
                self.path = path
                self.path_index = 0
                return

        self.scout_target = self.choose_scout_target(ct)
        if self.scout_target is not None:
            path = a_star_to_any(
                ct,
                current,
                {self.scout_target},
                self.traversable_for_planning,
                movement_directions=DIRECTIONS,
                extra_step_cost_fn=lambda pos: self.scout_path_step_cost(current, pos),
            )
            self.path = path
            self.path_index = 0

    def path_to_ore_network(self, ct: Controller, current: Position, ore: Position) -> list[Position] | None:
        if self.core_pos is None:
            return None
        branch = self.steiner_branch_to_ore(ore)
        if branch is None:
            return None
        if current == self.core_pos:
            return branch

        return_path = a_star_to_any(ct, current, {self.core_pos}, self.traversable_for_planning)
        if not return_path:
            return None
        return return_path + branch

    def steiner_branch_to_ore(self, ore: Position) -> list[Position] | None:
        if self.core_pos is None:
            return None

        branches = []
        for approach in self.buildable_approaches(ore):
            if approach == self.core_pos:
                branches.append([])
                continue
            if approach not in self.steiner_parent:
                continue

            branch = []
            pos = approach
            while pos != self.core_pos:
                parent = self.steiner_parent.get(pos)
                if parent is None:
                    break
                branch.append(pos)
                pos = parent
            if pos == self.core_pos:
                branch.reverse()
                branches.append(branch)

        if not branches:
            return None
        return min(branches, key=len)


    def follow_path_and_build(self, ct: Controller) -> None:
        current = ct.get_position()
        while self.path_index < len(self.path) and current == self.path[self.path_index]:
            self.path_index += 1

        if self.path_index >= len(self.path):
            if self.scout_target is not None and current == self.scout_target:
                self.scout_target = None
            return

        next_pos = self.path[self.path_index]
        move_dir = current.direction_to(next_pos)
        if move_dir == Direction.CENTRE:
            self.path_index += 1
            return

        if current.distance_squared(next_pos) > 2:
            self.select_new_target(ct)
            return

        if next_pos in self.steiner_parent:
            if self.ensure_tree_conveyor(ct, next_pos):
                return
            if not ct.is_tile_passable(next_pos):
                return

        if not ct.is_tile_passable(next_pos):
            self.try_prepare_tile(ct, next_pos)

        if ct.can_move(move_dir):
            ct.move(move_dir)
            self._stuck_rounds = 0
            self._last_progress_round = self._rounds_alive
            return

        if not ct.is_tile_passable(next_pos):
            self.select_new_target(ct)

    def try_prepare_tile(self, ct: Controller, target: Position) -> None:
        if target in self.steiner_parent:
            self.ensure_tree_conveyor(ct, target)
            return

        if ct.can_build_road(target):
            ct.build_road(target)
            self._last_progress_round = self._rounds_alive
            return

        if not ct.is_tile_passable(target):
            if target not in self.permanently_blocked:
                self.permanently_blocked.add(target)
                self.steiner_ores_key = frozenset()

    def ensure_tree_conveyor(self, ct: Controller, target: Position) -> bool:
        conveyor_direction = self._get_conveyor_direction(target)
        building_id = ct.get_tile_building_id(target)
        if building_id is not None:
            building_type = ct.get_entity_type(building_id)
            if building_type == EntityType.CONVEYOR and ct.get_direction(building_id) == conveyor_direction:
                return False
            if ct.get_team(building_id) == self.team and ct.can_destroy(target):
                ct.destroy(target)
                self.known_buildings[target] = None
                return True
            return False

        if ct.can_build_conveyor(target, conveyor_direction):
            ct.build_conveyor(target, conveyor_direction)
            self._last_progress_round = self._rounds_alive
            return True
        return False

    def should_build_splitter(self, ct: Controller, target: Position, conveyor_direction: Direction) -> bool:
        if self.target_resource != RESOURCE_TITANIUM:
            return False
        if self.core_pos is None:
            return False
        if target.distance_squared(self.core_pos) != 1:
            return False
        if not ct.can_build_splitter(target, conveyor_direction):
            return False
        left = target.add(conveyor_direction.rotate_left())
        right = target.add(conveyor_direction.rotate_right())
        return in_bounds(ct, left) and in_bounds(ct, right)

    def find_bridge_target(self, bridge_pos: Position) -> Position | None:
        if self.path_index + 3 >= len(self.path):
            return None
        best_target = None
        best_gain = 0
        for offset in range(3, min(7, len(self.path) - self.path_index)):
            candidate = self.path[self.path_index + offset]
            if bridge_pos.distance_squared(candidate) > GameConstants.BRIDGE_TARGET_RADIUS_SQ:
                continue
            gain = offset - 1
            if gain <= best_gain:
                continue
            best_target = candidate
            best_gain = gain
        return best_target

    def try_build_gunner(self, ct: Controller) -> bool:
        titanium, _ = ct.get_global_resources()
        gunner_cost, _ = ct.get_gunner_cost()
        if titanium < gunner_cost + GUNNER_RESERVE:
            return False
        if self.core_pos is None:
            return False

        enemy_hint = self.read_enemy_marker_target(ct)
        if enemy_hint is None:
            enemy_hint = Position(
                ct.get_map_width() - 1 - self.core_pos.x,
                ct.get_map_height() - 1 - self.core_pos.y,
            )

        facing = self.core_pos.direction_to(enemy_hint)
        for direction in DIRECTIONS:
            build_pos = ct.get_position().add(direction)
            if not in_bounds(ct, build_pos):
                continue
            if build_pos.distance_squared(self.core_pos) > 8:
                continue
            if ct.can_build_gunner(build_pos, facing):
                ct.build_gunner(build_pos, facing)
                return True
        return False

    def read_enemy_marker_target(self, ct: Controller) -> Position | None:
        for entity_id in ct.get_nearby_entities():
            if ct.get_entity_type(entity_id) != EntityType.MARKER:
                continue
            try:
                kind, pos, _ = decode_marker(ct.get_marker_value(entity_id))
            except Exception:
                continue
            if kind == MARKER_KIND_ENEMY:
                return pos
        return None


    def observe_tiles(self, ct: Controller) -> None:
        for pos in ct.get_nearby_tiles():
            env = ct.get_tile_env(pos)
            old_env = self.known_env.get(pos)
            self.known_env[pos] = env
            if old_env is None:
                self.update_scout_frontier(pos)

            # При обнаружении новой руды вычисляем зеркальную позицию
            if old_env != env and env in ORE_TYPES:
                self._infer_symmetric(pos, env)

            # Если была стена там, где Штейнер планировал путь — перестроить
            if old_env is None and env == Environment.WALL and pos in self.steiner_parent:
                self.permanently_blocked.add(pos)
                self.steiner_ores_key = frozenset()

            building_info = None
            building_id = ct.get_tile_building_id(pos)
            if building_id is not None:
                building_info = (ct.get_entity_type(building_id), ct.get_team(building_id))
            self.known_buildings[pos] = building_info

    def update_scout_frontier(self, known_pos: Position) -> None:
        if self.map_width == 0 or self.map_height == 0:
            return

        self.scout_frontier_initialized = True
        self.scout_frontier.discard(known_pos)
        for direction in DIRECTIONS:
            probe = known_pos.add(direction)
            if not (0 <= probe.x < self.map_width and 0 <= probe.y < self.map_height):
                continue
            if probe in self.known_env:
                self.scout_frontier.discard(probe)
            else:
                self.scout_frontier.add(probe)

    def rebuild_scout_frontier(self) -> None:
        for known_pos in self.known_env:
            self.update_scout_frontier(known_pos)

    def known_titanium_ores(self) -> list[Position]:
        return [pos for pos, env in self.known_env.items() if env == Environment.ORE_TITANIUM]

    def known_axionite_ores(self) -> list[Position]:
        return [pos for pos, env in self.known_env.items() if env == Environment.ORE_AXIONITE]

    def is_harvester_on_tile(self, pos: Position, allied_only: bool = False) -> bool:
        building_info = self.known_buildings.get(pos)
        if building_info is None:
            return False
        building_type, team = building_info
        if building_type != EntityType.HARVESTER:
            return False
        return not allied_only or team == self.team

    def traversable_for_planning(self, _ct: Controller | None, pos: Position) -> bool:
        if not (0 <= pos.x < self.map_width and 0 <= pos.y < self.map_height):
            return False
        if pos in self.permanently_blocked:
            return False
        env = self.known_env.get(pos)
        if env == Environment.WALL or env in ORE_TYPES:
            return False
        building_info = self.known_buildings.get(pos)
        if building_info is None:
            return True
        building_type, _ = building_info
        return building_type in PASSABLE_BUILDINGS

    def buildable_approaches(self, ore_pos: Position) -> list[Position]:
        candidates = []
        for direction in ORTHOGONAL_DIRECTIONS:
            pos = ore_pos.add(direction)
            if self.traversable_for_planning(None, pos):
                candidates.append(pos)
        return candidates


    def choose_scout_target(self, ct: Controller) -> Position | None:
        current = ct.get_position()
        if not self.scout_frontier and self.known_env and not self.scout_frontier_initialized:
            self.rebuild_scout_frontier()

        candidates = [
            pos for pos in self.scout_frontier
            if pos not in self.known_env and pos not in self.permanently_blocked
        ]
        candidates.sort(key=lambda pos: self.scout_frontier_pre_score(current, pos), reverse=True)
        candidates = candidates[:SCOUT_FRONTIER_CANDIDATE_LIMIT]

        best = None
        best_score = None
        for probe in candidates:
            score = self.scout_frontier_score(current, probe)
            if best_score is None or score > best_score:
                best = probe
                best_score = score
        return best
