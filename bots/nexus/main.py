from cambc import Controller, Direction, EntityType, Environment, GameConstants, Position, Team

from bot.constants import (
    DIRECTIONS,
    FOURTH_BUILDER_RESERVE,
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
    SECOND_BUILDER_RESERVE,
    STUCK_KILL_ROUNDS,
    MAX_IDLE_ROUNDS,
    TITANIUM_LINE_READY_HARVESTERS,
    THIRD_BUILDER_RESERVE,
    GUNNER_RESERVE,
)

from bot.geometry import decode_marker, encode_marker, in_bounds
from bot.navigation import a_star_to_any
from bot.steiner import compute_steiner_tree
from bot.strategy import choose_phase



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
        self.try_spawn_builder(ct, phase, titanium_harvesters, axionite_harvesters)


    def run_builder(self, ct: Controller) -> None:
        self.init_map_state(ct)
        self._rounds_alive += 1

        cur_pos = ct.get_position()

        if self._last_pos is not None and cur_pos == self._last_pos:
            self._stuck_rounds += 1
        else:
            self._stuck_rounds = 0
            self._last_progress_round = self._rounds_alive
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
        ti_ores = self.known_titanium_ores()
        key = frozenset(ti_ores)
        if key == self.steiner_ores_key:
            return

        self.steiner_ores_key = key
        self.steiner_parent = compute_steiner_tree(
            self.core_pos,
            ti_ores,
            self.known_env,
            self.map_width,
            self.map_height,
            blocked=self.permanently_blocked,
        )

    def _get_conveyor_direction(self, ct: Controller, target: Position) -> Direction:
        steiner_p = self.steiner_parent.get(target)
        if steiner_p is not None:
            d = target.direction_to(steiner_p)
            if d != Direction.CENTRE:
                return d
        return self.get_line_direction(ct, target)


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
                    self._infer_symmetric(pos, Environment.ORE_TITANIUM)
            elif kind == MARKER_KIND_ORE_AX:
                if pos not in self.known_env:
                    self.known_env[pos] = Environment.ORE_AXIONITE
                    self._infer_symmetric(pos, Environment.ORE_AXIONITE)

    def _infer_symmetric(self, pos: Position, env: Environment) -> None:
        """Вывод симметричной позиции руды (карта гарантированно симметрична)"""
        if self.map_width == 0 or self.map_height == 0:
            return
        mirror = Position(self.map_width - 1 - pos.x, self.map_height - 1 - pos.y)
        if mirror not in self.known_env:
            self.known_env[mirror] = env


    def try_spawn_builder(
            self,
            ct: Controller,
            phase: int,
            titanium_harvesters: int,
            axionite_harvesters: int,
    ) -> None:
        if ct.get_unit_count() >= GameConstants.MAX_TEAM_UNITS:
            return
        if self.spawned_builders >= MAX_BUILDERS_PHASE_ONE:
            return

        titanium, _ = ct.get_global_resources()
        builder_cost, _ = ct.get_builder_bot_cost()

        if self.spawned_builders == 0:
            self.spawn_in_direction(ct, self.direction_towards_best_ore(ct, RESOURCE_TITANIUM))
            return

        if self.spawned_builders == 1:
            if titanium_harvesters == 0 and ct.get_current_round() < 60:
                return
            if titanium < builder_cost + SECOND_BUILDER_RESERVE:
                return
            self.spawn_in_direction(ct, self.direction_towards_best_ore(ct, RESOURCE_TITANIUM).rotate_right())
            return

        if self.spawned_builders == 2:
            if phase < PHASE_EXPAND_AXIONITE:
                return
            if titanium < builder_cost + THIRD_BUILDER_RESERVE:
                return
            self.spawn_in_direction(ct, self.direction_towards_best_ore(ct, RESOURCE_AXIONITE))
            return

        if self.spawned_builders == 3:
            if phase != PHASE_STABILIZE or axionite_harvesters == 0:
                return
            if titanium < builder_cost + FOURTH_BUILDER_RESERVE:
                return
            self.spawn_in_direction(ct, self.direction_towards_best_ore(ct, RESOURCE_TITANIUM).rotate_left())

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
                new_builder_id = ct.spawn_builder(spawn_pos)
                self.spawned_builders += 1
                return

    def direction_towards_best_ore(self, ct: Controller, resource_kind: str) -> Direction:
        origin = ct.get_position()
        ores = self.known_titanium_ores() if resource_kind == RESOURCE_TITANIUM else self.known_axionite_ores()
        for ore in sorted(ores, key=lambda pos: origin.distance_squared(pos)):
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

        preferred = set(self.steiner_parent.keys()) if self.target_resource == RESOURCE_TITANIUM else set()

        current = ct.get_position()
        for ore in sorted(candidates, key=lambda pos: current.distance_squared(pos)):
            if self.is_harvester_on_tile(ore):
                continue
            if ore in self.skipped_ores:
                continue
            goals = set(self.buildable_approaches(ore))
            if not goals:
                continue
            path = a_star_to_any(ct, current, goals, self.traversable_for_planning, preferred)
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
            self._stuck_rounds = 0
            self._last_progress_round = self._rounds_alive
            return

        if not ct.is_tile_passable(next_pos):
            self.select_new_target(ct)

    def try_prepare_tile(self, ct: Controller, target: Position) -> None:
        conveyor_built = False
        conveyor_direction = self._get_conveyor_direction(ct, target)
        if ct.can_build_conveyor(target, conveyor_direction):
            ct.build_conveyor(target, conveyor_direction)
            conveyor_built = True
            self._last_progress_round = self._rounds_alive

        if conveyor_built:
            return

        if ct.can_build_road(target):
            ct.build_road(target)
            self._last_progress_round = self._rounds_alive
            return

        if not ct.is_tile_passable(target):
            if target not in self.permanently_blocked:
                self.permanently_blocked.add(target)
                self.steiner_ores_key = frozenset()

    def get_line_direction(self, ct: Controller, target: Position) -> Direction:
        if self.path_index == 0:
            previous = ct.get_position()
        else:
            previous = self.path[self.path_index - 1]
        return target.direction_to(previous)

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
        preferred = self.enemy_estimate if self.enemy_estimate is not None else current
        forward = current.direction_to(preferred)
        if forward not in ORTHOGONAL_DIRECTIONS:
            if abs(preferred.x - current.x) >= abs(preferred.y - current.y):
                forward = Direction.EAST if preferred.x >= current.x else Direction.WEST
            else:
                forward = Direction.SOUTH if preferred.y >= current.y else Direction.NORTH

        probe = current
        for _ in range(4):
            probe = probe.add(forward)
            if not in_bounds(ct, probe):
                break
            if probe not in self.known_env:
                return probe

        best = None
        best_score = 10 ** 9
        for known_pos in self.known_env:
            for direction in ORTHOGONAL_DIRECTIONS:
                probe = known_pos.add(direction)
                if not in_bounds(ct, probe) or probe in self.known_env:
                    continue
                score = current.distance_squared(probe)
                if self.target_resource == RESOURCE_AXIONITE:
                    score = score * 2
                if self.enemy_estimate is not None:
                    score += probe.distance_squared(self.enemy_estimate)
                if score < best_score:
                    best = probe
                    best_score = score
        return best
