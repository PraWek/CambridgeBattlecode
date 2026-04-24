"""Bootstrap bot for Cambridge Battlecode.

Реализовано:
  - кэшированный маршрут к ближайшему месту добычи титановой руды
  - строительство конвейера/дороги по мере продвижения строителя
  - размещение комбайна по прибытии на рудник
  - упрощенная координация действий на основе маркеров вблизи активной зоны
  - простая смена стрелка на основе маркера с координатами цели
"""

from __future__ import annotations

from heapq import heappop, heappush

from cambc import Controller, Direction, EntityType, Environment, GameConstants, Position, Team


DIRECTIONS = [d for d in Direction if d != Direction.CENTRE]
ORTHOGONAL_DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
PASSABLE_BUILDINGS = {
    EntityType.CORE,
    EntityType.ROAD,
    EntityType.CONVEYOR,
    EntityType.ARMOURED_CONVEYOR,
    EntityType.BRIDGE,
    EntityType.SPLITTER,
}
ORE_TYPES = {Environment.ORE_TITANIUM, Environment.ORE_AXIONITE}

MARKER_KIND_ENEMY = 1
MARKER_KIND_PHASE = 2

PHASE_BOOTSTRAP = 1
PHASE_EXPAND_TITANIUM = 2
PHASE_DEFEND = 3

TI_BUILDER_RESERVE = 30
SECOND_BUILDER_RESERVE = 90
GUNNER_RESERVE = 80
LARGE_NUMBER = 10**9

MARKER_KIND_BASE = 1_000_000
MARKER_X_BASE = 10_000
MARKER_Y_BASE = 100


def chebyshev(a: Position, b: Position) -> int:
    return max(abs(a.x - b.x), abs(a.y - b.y))


def in_bounds(ct: Controller, pos: Position) -> bool:
    return 0 <= pos.x < ct.get_map_width() and 0 <= pos.y < ct.get_map_height()


def encode_marker(kind: int, pos: Position, payload: int = 0) -> int:
    return kind * MARKER_KIND_BASE + pos.x * MARKER_X_BASE + pos.y * MARKER_Y_BASE + payload


def decode_marker(value: int) -> tuple[int, Position, int]:
    kind = value // MARKER_KIND_BASE
    value %= MARKER_KIND_BASE
    x = value // MARKER_X_BASE
    value %= MARKER_X_BASE
    y = value // MARKER_Y_BASE
    payload = value % MARKER_Y_BASE
    return kind, Position(x, y), payload


def a_star_to_any(ct: Controller, start: Position, goals: set[Position], traversable_fn) -> list[Position]:
    if start in goals:
        return []

    queue = []
    came_from = {start}
    g_score = {start}
    heappush(queue, (0, 0, start))

    while queue:
        _, cost, current = heappop(queue)
        if current in goals:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        if cost != g_score[current]:
            continue

        for direction in ORTHOGONAL_DIRECTIONS:
            nxt = current.add(direction)
            if not traversable_fn(ct, nxt):
                continue
            new_cost = cost + 1
            if new_cost >= g_score.get(nxt, LARGE_NUMBER):
                continue
            g_score[nxt] = new_cost
            came_from[nxt] = current
            heuristic = min(chebyshev(nxt, goal) for goal in goals)
            heappush(queue, (new_cost + heuristic, new_cost, nxt))

    return []


class Player:
    def __init__(self) -> None:
        self.initialized = False

        self.core_pos = None
        self.enemy_estimate = None
        self.enemy_marker_pad = None
        self.phase_marker_pad = None
        self.map_width = 0
        self.map_height = 0
        self.team = None

        self.known_env = {}
        self.known_buildings = {}

        self.spawned_builders = 0

        self.target_ore = None
        self.scout_target = None
        self.path: list[Position] = []
        self.path_index = 0
        self.harvester_built = False
        self.role = "bootstrap"

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

        ti_harvesters = self.count_harvesters(self.known_titanium_ores(), allied_only=True, limit=2)
        phase = PHASE_BOOTSTRAP
        if ti_harvesters >= 1:
            phase = PHASE_EXPAND_TITANIUM
        if ti_harvesters >= 2:
            phase = PHASE_DEFEND

        self.place_core_markers(ct, phase)
        self.try_spawn_builder(ct, ti_harvesters)

    def run_builder(self, ct: Controller) -> None:
        self.init_map_state(ct)
        self.observe_tiles(ct)
        if self.core_pos is None:
            self.core_pos = self.find_home_core(ct)
        if self.core_pos is None:
            return

        titanium_harvesters = self.count_harvesters(self.known_titanium_ores(), allied_only=True, limit=3)
        if titanium_harvesters == 0:
            self.role = "bootstrap"
        elif titanium_harvesters < 2:
            self.role = "expand_titanium"
        else:
            self.role = "defend"

        if self.role == "defend" and self.try_build_gunner(ct):
            return

        need_new_target = False
        if self.harvester_built:
            need_new_target = True
        elif self.target_ore is not None and self.is_harvester_on_tile(self.target_ore):
            need_new_target = True
        elif self.target_ore is None and self.scout_target is None:
            need_new_target = True
        elif self.target_ore is None and self.scout_target is not None and self.known_titanium_ores():
            need_new_target = True

        if need_new_target:
            self.select_new_target(ct)

        if self.target_ore is None and self.scout_target is None:
            return

        if self.target_ore is not None and ct.get_position().distance_squared(self.target_ore) <= GameConstants.ACTION_RADIUS_SQ:
            if ct.can_build_harvester(self.target_ore):
                ct.build_harvester(self.target_ore)
                self.harvester_built = True
                self.target_ore = None
                self.scout_target = None
                self.path = []
                self.path_index = 0
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

    def try_spawn_builder(self, ct: Controller, ti_harvesters: int) -> None:
        if ct.get_unit_count() >= GameConstants.MAX_TEAM_UNITS:
            return

        titanium, _ = ct.get_global_resources()
        builder_cost, _ = ct.get_builder_bot_cost()

        if self.spawned_builders == 0:
            self.spawn_in_direction(ct, self.direction_towards_best_titanium(ct))
            return

        if self.spawned_builders >= 2 or (ti_harvesters == 0 and ct.get_current_round() < 60):
            return

        if titanium < builder_cost + SECOND_BUILDER_RESERVE:
            return
        self.spawn_in_direction(ct, self.direction_towards_best_titanium(ct).rotate_right())

    def spawn_in_direction(self, ct: Controller, preferred: Direction) -> None:
        core_pos = ct.get_position()
        ordered = [preferred]
        left = preferred
        right = preferred
        for _ in range(3):
            left = left.rotate_left()
            right = right.rotate_right()
            ordered.extend([left, right])
        ordered.extend(d for d in DIRECTIONS if d not in ordered)

        for direction in ordered:
            spawn_pos = core_pos.add(direction)
            if ct.can_spawn(spawn_pos):
                ct.spawn_builder(spawn_pos)
                self.spawned_builders += 1
                return

    def direction_towards_best_titanium(self, ct: Controller) -> Direction:
        origin = ct.get_position()
        for ore in sorted(self.known_titanium_ores(), key=lambda pos: origin.distance_squared(pos)):
            return origin.direction_to(ore)
        if self.enemy_estimate is not None:
            return origin.direction_to(self.enemy_estimate)
        return Direction.NORTH

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

    def select_new_target(self, ct: Controller) -> None:
        self.harvester_built = False
        self.target_ore = None
        self.scout_target = None
        self.path = []
        self.path_index = 0

        if self.role in {"bootstrap", "expand_titanium"}:
            candidates = self.known_titanium_ores()
        else:
            candidates = self.known_titanium_ores()

        current = ct.get_position()
        for ore in sorted(candidates, key=lambda pos: current.distance_squared(pos)):
            if self.is_harvester_on_tile(ore):
                continue
            goals = set(self.buildable_approaches(ore))
            if not goals:
                continue
            path = a_star_to_any(ct, current, goals, self.traversable_for_planning)
            if current in goals or path:
                self.target_ore = ore
                self.path = path
                self.path_index = 0
                return

        self.scout_target = self.choose_scout_target(ct)
        if self.scout_target is not None:
            path = a_star_to_any(ct, current, {self.scout_target}, self.traversable_for_planning)
            self.path = path
            self.path_index = 0

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

        if not ct.is_tile_passable(next_pos):
            self.try_prepare_tile(ct, next_pos)

        if ct.can_move(move_dir):
            ct.move(move_dir)
            return

        if not ct.is_tile_passable(next_pos):
            self.select_new_target(ct)

    def try_prepare_tile(self, ct: Controller, target: Position) -> None:
        if self.path_index == 0:
            previous = ct.get_position()
        else:
            previous = self.path[self.path_index - 1]
        conveyor_direction = target.direction_to(previous)

        if ct.can_build_conveyor(target, conveyor_direction):
            ct.build_conveyor(target, conveyor_direction)
            return
        if ct.can_build_road(target):
            ct.build_road(target)

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
            self.known_env[pos] = ct.get_tile_env(pos)
            building_info = None
            building_id = ct.get_tile_building_id(pos)
            if building_id is not None:
                building_info = (ct.get_entity_type(building_id), ct.get_team(building_id))
            self.known_buildings[pos] = building_info

    def known_titanium_ores(self) -> list[Position]:
        return [pos for pos, env in self.known_env.items() if env == Environment.ORE_TITANIUM]

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
        preferred = self.enemy_estimate if self.enemy_estimate is not None else current
        forward = current.direction_to(preferred)
        if forward not in ORTHOGONAL_DIRECTIONS:
            forward = Direction.EAST if abs(preferred.x - current.x) >= abs(preferred.y - current.y) and preferred.x >= current.x else (
                Direction.WEST if abs(preferred.x - current.x) >= abs(preferred.y - current.y) else (
                    Direction.SOUTH if preferred.y >= current.y else Direction.NORTH
                )
            )
        probe = current
        for _ in range(4):
            probe = probe.add(forward)
            if not in_bounds(ct, probe):
                break
            if probe not in self.known_env:
                return probe

        best = None
        best_score = LARGE_NUMBER
        for known_pos in self.known_env:
            for direction in ORTHOGONAL_DIRECTIONS:
                probe = known_pos.add(direction)
                if not in_bounds(ct, probe) or probe in self.known_env:
                    continue
                score = current.distance_squared(probe)
                if self.enemy_estimate is not None:
                    score += probe.distance_squared(self.enemy_estimate)
                if score < best_score:
                    best = probe
                    best_score = score
        return best
