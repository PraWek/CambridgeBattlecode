from __future__ import annotations

from cambc import Controller, Direction, EntityType, Environment, GameConstants, Position, ResourceType

from bot.constants import (
    BARRIER_RESERVE,
    BREACH_RESERVE,
    CONVERT_AXIONITE_MIN,
    CONVERT_TITANIUM_FLOOR,
    DIRECTIONS,
    FOUNDRY_AXIONITE_KEEP,
    FOUNDRY_TITANIUM_BUFFER,
    FOURTH_BUILDER_RESERVE,
    GUNNER_RESERVE,
    LAUNCHER_RESERVE,
    MARKER_KIND_ENEMY,
    MARKER_KIND_PHASE,
    MAX_BREACHES,
    MAX_GUNNERS,
    MAX_LAUNCHERS,
    MAX_BUILDERS_PHASE_ONE,
    MAX_SENTINELS,
    ORE_TYPES,
    ORTHOGONAL_DIRECTIONS,
    PASSABLE_BUILDINGS,
    PHASE_BOOTSTRAP,
    PHASE_EXPAND_AXIONITE,
    PHASE_EXPAND_TITANIUM,
    PHASE_REFINE_AXIONITE,
    PHASE_STABILIZE,
    RESOURCE_AXIONITE,
    RESOURCE_TITANIUM,
    SECOND_BUILDER_RESERVE,
    SENTINEL_RESERVE,
    THIRD_BUILDER_RESERVE,
    TITANIUM_LINE_READY_SCALE,
)
from bot.geometry import decode_marker, encode_marker, in_bounds
from bot.navigation import a_star_to_any
from bot.strategy import choose_phase


TRANSPORT_BUILDINGS = {
    EntityType.CONVEYOR,
    EntityType.SPLITTER,
    EntityType.BRIDGE,
    EntityType.ARMOURED_CONVEYOR,
}


class FoundryPlan:
    def __init__(
        self,
        foundry_pos: Position,
        titanium_input: Position,
        axionite_input: Position,
        output_pos: Position | None,
        output_target: Position | None,
    ) -> None:
        self.foundry_pos = foundry_pos
        self.titanium_input = titanium_input
        self.axionite_input = axionite_input
        self.output_pos = output_pos
        self.output_target = output_target


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
        self.foundry_plan = None

    def run(self, ct: Controller) -> None:
        entity_type = ct.get_entity_type()

        if entity_type == EntityType.CORE:
            self.run_core(ct)
        elif entity_type == EntityType.BUILDER_BOT:
            self.run_builder(ct)
        elif entity_type == EntityType.GUNNER:
            self.run_gunner(ct)
        elif entity_type == EntityType.SENTINEL:
            self.run_sentinel(ct)
        elif entity_type == EntityType.BREACH:
            self.run_breach(ct)
        elif entity_type == EntityType.LAUNCHER:
            self.run_launcher(ct)

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
        foundry_count = self.count_known_buildings(EntityType.FOUNDRY, allied_only=True, limit=1)
        phase = choose_phase(ct, titanium_harvesters, axionite_harvesters, foundry_count)

        self.place_core_markers(ct, phase)
        self.try_convert_axionite(ct, foundry_count)
        self.try_spawn_builder(ct, phase, titanium_harvesters, axionite_harvesters, foundry_count)

    def run_builder(self, ct: Controller) -> None:
        self.init_map_state(ct)
        self.observe_tiles(ct)
        if self.core_pos is None:
            self.core_pos = self.find_home_core(ct)
        if self.core_pos is None:
            return

        self.update_role_from_phase_marker(ct)

        if self.role == "refine" and self.run_refinery_builder(ct):
            return
        if self.role == "stabilize" and self.run_combat_builder(ct):
            return

        need_new_target = False
        if self.harvester_built:
            need_new_target = True
        elif self.target_ore is not None and self.is_harvester_on_tile(self.target_ore):
            need_new_target = True
        elif self.target_ore is None and self.scout_target is None:
            need_new_target = True
        elif self.target_ore is None and self.scout_target is not None and self.known_ores_for_role():
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
        if self.try_fire_best_target(ct):
            return
        self.rotate_towards_enemy(ct)

    def run_sentinel(self, ct: Controller) -> None:
        self.observe_tiles(ct)
        self.try_fire_best_target(ct)

    def run_breach(self, ct: Controller) -> None:
        self.observe_tiles(ct)
        self.try_fire_best_target(ct)

    def run_launcher(self, ct: Controller) -> None:
        self.observe_tiles(ct)
        allied_builder_positions = []
        for unit_id in ct.get_nearby_units():
            if ct.get_team(unit_id) != self.team:
                continue
            if ct.get_entity_type(unit_id) != EntityType.BUILDER_BOT:
                continue
            allied_builder_positions.append(ct.get_position(unit_id))

        target = self.find_enemy_core_position(ct)
        if target is None:
            target = self.enemy_estimate
        if target is None:
            return

        for builder_pos in sorted(allied_builder_positions, key=lambda pos: ct.get_position().distance_squared(pos)):
            if ct.can_launch(builder_pos, target):
                ct.launch(builder_pos, target)
                return

    def run_refinery_builder(self, ct: Controller) -> bool:
        if self.core_pos is None:
            return False

        plan = self.find_foundry_plan(ct)
        self.foundry_plan = plan
        if plan is None:
            return self.try_idle_near_core(ct)

        if not self.is_allied_foundry_at(plan.foundry_pos):
            if ct.get_position().distance_squared(plan.foundry_pos) <= GameConstants.ACTION_RADIUS_SQ:
                if ct.can_build_foundry(plan.foundry_pos):
                    ct.build_foundry(plan.foundry_pos)
                    return True
            return self.move_towards_target_tile(ct, plan.foundry_pos)

        if self.ensure_foundry_links(ct, plan):
            return True

        self.role = "stabilize"
        return False

    def run_combat_builder(self, ct: Controller) -> bool:
        if self.try_builder_attack_on_enemy_core(ct):
            return True
        if self.try_build_barrier(ct):
            return True
        if self.try_build_perimeter_gunner(ct):
            return True
        if self.try_build_sentinel(ct):
            return True
        if self.try_build_breach(ct):
            return True
        if self.try_build_launcher(ct):
            return True
        return False

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

    def try_convert_axionite(self, ct: Controller, foundry_count: int) -> None:
        titanium, axionite = ct.get_global_resources()
        if foundry_count == 0:
            return
        if axionite < CONVERT_AXIONITE_MIN or titanium >= CONVERT_TITANIUM_FLOOR:
            return
        available = axionite - FOUNDRY_AXIONITE_KEEP
        if available <= 0:
            return
        need = max(1, (CONVERT_TITANIUM_FLOOR - titanium + 3) // 4)
        amount = min(available, need)
        if amount > 0:
            ct.convert(amount)

    def try_spawn_builder(
        self,
        ct: Controller,
        phase: int,
        titanium_harvesters: int,
        axionite_harvesters: int,
        foundry_count: int,
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
            if titanium_harvesters == 0 and ct.get_current_round() < 45:
                return
            if titanium < builder_cost + SECOND_BUILDER_RESERVE:
                return
            self.spawn_in_direction(ct, self.direction_towards_best_ore(ct, RESOURCE_TITANIUM).rotate_right())
            return

        if self.spawned_builders == 2:
            if phase < PHASE_EXPAND_AXIONITE and ct.get_current_round() < 220:
                return
            reserve = THIRD_BUILDER_RESERVE if phase >= PHASE_EXPAND_AXIONITE else 60
            if titanium < builder_cost + reserve:
                return
            self.spawn_in_direction(ct, self.direction_towards_best_ore(ct, RESOURCE_AXIONITE))
            return

        if self.spawned_builders == 3:
            if phase != PHASE_REFINE_AXIONITE or axionite_harvesters == 0 or foundry_count > 0:
                return
            foundry_cost, _ = ct.get_foundry_cost()
            if titanium < builder_cost + foundry_cost + FOUNDRY_TITANIUM_BUFFER + FOURTH_BUILDER_RESERVE:
                return
            self.spawn_in_direction(ct, self.direction_towards_best_ore(ct, RESOURCE_AXIONITE).rotate_left())

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

    def count_known_buildings(self, entity_type: EntityType, allied_only: bool, limit: int | None = None) -> int:
        count = 0
        for building_info in self.known_buildings.values():
            if building_info is None:
                continue
            building_type, team = building_info
            if building_type != entity_type:
                continue
            if allied_only and team != self.team:
                continue
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
        phase = self.read_phase_marker(ct)
        if phase is None:
            return
        if phase == PHASE_BOOTSTRAP:
            self.role = "bootstrap"
        elif phase == PHASE_EXPAND_TITANIUM:
            self.role = "expand_titanium"
        elif phase == PHASE_EXPAND_AXIONITE:
            self.role = "expand_axionite"
        elif phase == PHASE_REFINE_AXIONITE:
            self.role = "refine"
        else:
            if self.role == "refine":
                self.role = "stabilize"
            elif self.role == "expand_axionite":
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

    def select_new_target(self, ct: Controller) -> None:
        self.harvester_built = False
        self.target_ore = None
        self.scout_target = None
        self.path = []
        self.path_index = 0

        if self.role not in {"expand_axionite", "refine"} and (
            ct.get_scale_percent() >= TITANIUM_LINE_READY_SCALE
            or (ct.get_current_round() >= 220 and self.known_axionite_ores())
        ):
            self.role = "expand_axionite"

        self.target_resource = RESOURCE_AXIONITE if self.role == "expand_axionite" else RESOURCE_TITANIUM
        candidates = self.known_axionite_ores() if self.target_resource == RESOURCE_AXIONITE else self.known_titanium_ores()

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
        conveyor_direction = self.get_line_direction(ct, target)
        if ct.can_build_conveyor(target, conveyor_direction):
            ct.build_conveyor(target, conveyor_direction)
            return
        if ct.can_build_road(target):
            ct.build_road(target)

    def get_line_direction(self, ct: Controller, target: Position) -> Direction:
        if self.path_index == 0:
            previous = ct.get_position()
        else:
            previous = self.path[self.path_index - 1]
        return target.direction_to(previous)

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

    def try_build_perimeter_gunner(self, ct: Controller) -> bool:
        if self.count_known_buildings(EntityType.GUNNER, allied_only=True, limit=MAX_GUNNERS) >= MAX_GUNNERS:
            return False
        titanium, _ = ct.get_global_resources()
        gunner_cost, _ = ct.get_gunner_cost()
        if titanium < gunner_cost + GUNNER_RESERVE:
            return False
        build_pos = self.find_gunner_position(ct)
        if build_pos is None:
            return False
        if ct.get_position().distance_squared(build_pos) > GameConstants.ACTION_RADIUS_SQ:
            return self.move_towards_target_tile(ct, build_pos)
        facing = self.direction_from_pos_to_enemy(build_pos)
        if ct.can_build_gunner(build_pos, facing):
            ct.build_gunner(build_pos, facing)
            return True
        return False

    def try_build_sentinel(self, ct: Controller) -> bool:
        if self.count_known_buildings(EntityType.SENTINEL, allied_only=True, limit=MAX_SENTINELS) >= MAX_SENTINELS:
            return False
        titanium, axionite = ct.get_global_resources()
        sentinel_cost_ti, sentinel_cost_ax = ct.get_sentinel_cost()
        if titanium < sentinel_cost_ti + SENTINEL_RESERVE or axionite < sentinel_cost_ax + 10:
            return False
        build_pos = self.find_ring_build_position(ct, prefer_enemy=True)
        if build_pos is None:
            return False
        if ct.get_position().distance_squared(build_pos) > GameConstants.ACTION_RADIUS_SQ:
            return self.move_towards_target_tile(ct, build_pos)
        facing = self.direction_from_pos_to_enemy(build_pos)
        if ct.can_build_sentinel(build_pos, facing):
            ct.build_sentinel(build_pos, facing)
            return True
        return False

    def try_build_breach(self, ct: Controller) -> bool:
        if self.count_known_buildings(EntityType.BREACH, allied_only=True, limit=MAX_BREACHES) >= MAX_BREACHES:
            return False
        titanium, axionite = ct.get_global_resources()
        breach_cost_ti, breach_cost_ax = ct.get_breach_cost()
        if titanium < breach_cost_ti + BREACH_RESERVE or axionite < breach_cost_ax + 15:
            return False
        build_pos = self.find_forward_offense_position(ct)
        if build_pos is None:
            return False
        if ct.get_position().distance_squared(build_pos) > GameConstants.ACTION_RADIUS_SQ:
            return self.move_towards_target_tile(ct, build_pos)
        facing = self.direction_from_pos_to_enemy(build_pos)
        if ct.can_build_breach(build_pos, facing):
            ct.build_breach(build_pos, facing)
            return True
        return False

    def try_build_launcher(self, ct: Controller) -> bool:
        if self.count_known_buildings(EntityType.LAUNCHER, allied_only=True, limit=MAX_LAUNCHERS) >= MAX_LAUNCHERS:
            return False
        titanium, _ = ct.get_global_resources()
        launcher_cost, _ = ct.get_launcher_cost()
        if titanium < launcher_cost + LAUNCHER_RESERVE:
            return False
        build_pos = self.find_ring_build_position(ct, prefer_enemy=True)
        if build_pos is None:
            return False
        if ct.get_position().distance_squared(build_pos) > GameConstants.ACTION_RADIUS_SQ:
            return self.move_towards_target_tile(ct, build_pos)
        if ct.can_build_launcher(build_pos):
            ct.build_launcher(build_pos)
            return True
        return False

    def try_build_barrier(self, ct: Controller) -> bool:
        titanium, _ = ct.get_global_resources()
        barrier_cost, _ = ct.get_barrier_cost()
        if titanium < barrier_cost + BARRIER_RESERVE:
            return False
        build_pos = self.find_barrier_position()
        if build_pos is None:
            return False
        if ct.get_position().distance_squared(build_pos) > GameConstants.ACTION_RADIUS_SQ:
            return self.move_towards_target_tile(ct, build_pos)
        if ct.can_build_barrier(build_pos):
            ct.build_barrier(build_pos)
            return True
        return False

    def find_foundry_plan(self, ct: Controller) -> FoundryPlan | None:
        if self.core_pos is None:
            return None

        titanium_tiles = self.find_resource_line_tiles(ct, ResourceType.TITANIUM, self.known_titanium_ores())
        axionite_tiles = self.find_resource_line_tiles(ct, ResourceType.RAW_AXIONITE, self.known_axionite_ores())

        best_plan = None
        best_score = 10**9
        for titanium_pos in titanium_tiles[:6]:
            for axionite_pos in axionite_tiles[:6]:
                if titanium_pos == axionite_pos:
                    continue
                common_neighbors = set(self.adjacent_tiles(titanium_pos)) & set(self.adjacent_tiles(axionite_pos))
                for foundry_pos in common_neighbors:
                    if not self.can_place_foundry_on(foundry_pos):
                        continue
                    output_pos, output_target = self.find_foundry_output(foundry_pos, titanium_pos, axionite_pos)
                    if output_target is None and not self.is_adjacent_to_core(foundry_pos):
                        continue
                    score = foundry_pos.distance_squared(self.core_pos)
                    score += titanium_pos.distance_squared(self.core_pos)
                    score += axionite_pos.distance_squared(self.core_pos)
                    if self.is_adjacent_to_core(foundry_pos):
                        score -= 10
                    if score < best_score:
                        best_score = score
                        best_plan = FoundryPlan(
                            foundry_pos=foundry_pos,
                            titanium_input=titanium_pos,
                            axionite_input=axionite_pos,
                            output_pos=output_pos,
                            output_target=output_target,
                        )
        return best_plan

    def find_resource_line_tiles(
        self,
        ct: Controller,
        resource_type: ResourceType,
        ores: list[Position],
    ) -> list[Position]:
        ranked = []
        for pos, building_info in self.known_buildings.items():
            if building_info is None:
                continue
            building_type, team = building_info
            if team != self.team or building_type not in TRANSPORT_BUILDINGS:
                continue
            if self.core_pos is None or pos.distance_squared(self.core_pos) > 25:
                continue
            building_id = ct.get_tile_building_id(pos)
            if building_id is None:
                continue
            score = pos.distance_squared(self.core_pos)
            stored = self.safe_get_stored_resource(ct, building_id)
            if stored == resource_type:
                score -= 1_000
            elif stored is not None:
                score += 300
            if ores:
                score += min(pos.distance_squared(ore) for ore in ores)
            if self.is_core_contact_tile(pos):
                score -= 150
            ranked.append((score, pos))

        if ranked:
            ranked.sort(key=lambda item: item[0])
            return [pos for _, pos in ranked]

        if self.core_pos is None or not ores:
            return []

        predicted = []
        for ore in sorted(ores, key=lambda pos: self.core_pos.distance_squared(pos)):
            best_contact = min(self.core_contact_tiles(), key=lambda pos: pos.distance_squared(ore))
            predicted.append(best_contact)
        return predicted

    def can_place_foundry_on(self, pos: Position) -> bool:
        if self.core_pos is None:
            return False
        if not (0 <= pos.x < self.map_width and 0 <= pos.y < self.map_height):
            return False
        env = self.known_env.get(pos)
        if env == Environment.WALL or env in ORE_TYPES:
            return False
        return self.known_buildings.get(pos) is None

    def find_foundry_output(
        self,
        foundry_pos: Position,
        titanium_pos: Position,
        axionite_pos: Position,
    ) -> tuple[Position | None, Position | None]:
        if self.is_adjacent_to_core(foundry_pos):
            return None, None

        for contact in sorted(self.core_contact_tiles(), key=lambda pos: foundry_pos.distance_squared(pos)):
            if contact in {titanium_pos, axionite_pos}:
                continue
            if foundry_pos.distance_squared(contact) != 1:
                continue
            building_info = self.known_buildings.get(contact)
            if building_info is not None and building_info[1] != self.team:
                continue
            return contact, self.core_input_target(contact)
        return None, None

    def ensure_foundry_links(self, ct: Controller, plan: FoundryPlan) -> bool:
        titanium_target = plan.foundry_pos
        if self.is_core_contact_tile(plan.titanium_input):
            core_target = self.core_input_target(plan.titanium_input)
            splitter_direction = plan.titanium_input.direction_to(core_target)
            left_target = plan.titanium_input.add(splitter_direction.rotate_left())
            right_target = plan.titanium_input.add(splitter_direction.rotate_right())
            if plan.foundry_pos == left_target or plan.foundry_pos == right_target:
                if self.ensure_transport_building(ct, plan.titanium_input, splitter_direction, use_splitter=True):
                    return True
                titanium_target = plan.foundry_pos

        if self.ensure_transport_building(
            ct,
            plan.axionite_input,
            plan.axionite_input.direction_to(plan.foundry_pos),
            use_splitter=False,
        ):
            return True

        if titanium_target == plan.foundry_pos:
            if self.ensure_transport_building(
                ct,
                plan.titanium_input,
                plan.titanium_input.direction_to(plan.foundry_pos),
                use_splitter=False,
            ):
                return True

        if plan.output_pos is not None and plan.output_target is not None:
            if self.ensure_transport_building(
                ct,
                plan.output_pos,
                plan.output_pos.direction_to(plan.output_target),
                use_splitter=False,
            ):
                return True

        return False

    def ensure_transport_building(
        self,
        ct: Controller,
        pos: Position,
        direction: Direction,
        use_splitter: bool,
    ) -> bool:
        if ct.get_position().distance_squared(pos) > GameConstants.ACTION_RADIUS_SQ:
            return self.move_towards_target_tile(ct, pos)

        building_id = ct.get_tile_building_id(pos)
        if building_id is not None:
            if ct.get_team(building_id) != self.team:
                return False
            building_type = ct.get_entity_type(building_id)
            if use_splitter and building_type == EntityType.SPLITTER and self.safe_get_direction(ct, building_id) == direction:
                return False
            if not use_splitter and building_type == EntityType.CONVEYOR and self.safe_get_direction(ct, building_id) == direction:
                return False
            if building_type == EntityType.FOUNDRY:
                return False
            if ct.can_destroy(pos):
                ct.destroy(pos)

        if use_splitter:
            if ct.can_build_splitter(pos, direction):
                ct.build_splitter(pos, direction)
                return True
            return False

        if ct.can_build_conveyor(pos, direction):
            ct.build_conveyor(pos, direction)
            return True
        return False

    def move_towards_target_tile(self, ct: Controller, target: Position) -> bool:
        goals = {pos for pos in self.adjacent_tiles(target) if self.traversable_for_planning(None, pos)}
        if not goals:
            return False
        current = ct.get_position()
        path = a_star_to_any(ct, current, goals, self.traversable_for_planning)
        if not path:
            return False
        self.path = path
        self.path_index = 0
        self.follow_path_and_build(ct)
        return True

    def try_idle_near_core(self, ct: Controller) -> bool:
        if self.core_pos is None:
            return False
        if ct.get_position().distance_squared(self.core_pos) <= 8:
            return False
        return self.move_towards_target_tile(ct, self.core_pos)

    def try_fire_best_target(self, ct: Controller) -> bool:
        for target in self.prioritized_enemy_targets(ct):
            if ct.can_fire(target):
                ct.fire(target)
                return True
        return False

    def prioritized_enemy_targets(self, ct: Controller) -> list[Position]:
        enemy_core = self.find_enemy_core_position(ct)
        if enemy_core is not None:
            return [enemy_core] + self.visible_enemy_unit_positions(ct) + self.visible_enemy_building_positions(ct)
        return self.visible_enemy_unit_positions(ct) + self.visible_enemy_building_positions(ct)

    def visible_enemy_unit_positions(self, ct: Controller) -> list[Position]:
        positions = []
        for unit_id in ct.get_nearby_units():
            if ct.get_team(unit_id) == self.team:
                continue
            positions.append(ct.get_position(unit_id))
        positions.sort(key=lambda pos: ct.get_position().distance_squared(pos))
        return positions

    def visible_enemy_building_positions(self, ct: Controller) -> list[Position]:
        positions = []
        for building_id in ct.get_nearby_buildings():
            if ct.get_team(building_id) == self.team:
                continue
            positions.append(ct.get_position(building_id))
        positions.sort(key=lambda pos: ct.get_position().distance_squared(pos))
        return positions

    def rotate_towards_enemy(self, ct: Controller) -> None:
        marker_target = self.read_enemy_marker_target(ct)
        if marker_target is None:
            marker_target = self.enemy_estimate
        if marker_target is None:
            return
        desired = ct.get_position().direction_to(marker_target)
        if desired != Direction.CENTRE and desired != ct.get_direction() and ct.can_rotate(desired):
            ct.rotate(desired)

    def adjacent_tiles(self, pos: Position) -> list[Position]:
        return [pos.add(direction) for direction in ORTHOGONAL_DIRECTIONS]

    def find_gunner_position(self, ct: Controller) -> Position | None:
        candidates = []
        for pos, building_info in self.known_buildings.items():
            if building_info is None or building_info[1] != self.team:
                continue
            if building_info[0] not in TRANSPORT_BUILDINGS:
                continue
            for probe in self.adjacent_tiles(pos):
                if not self.can_place_combat_building_on(probe):
                    continue
                score = probe.distance_squared(self.core_pos) if self.core_pos is not None else 0
                if self.enemy_estimate is not None:
                    score += probe.distance_squared(self.enemy_estimate) // 2
                candidates.append((score, probe))
        if not candidates:
            return self.find_ring_build_position(ct, prefer_enemy=False)
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def find_ring_build_position(self, ct: Controller, prefer_enemy: bool) -> Position | None:
        candidates = []
        for pos in self.core_contact_tiles():
            for probe in self.adjacent_tiles(pos):
                if not self.can_place_combat_building_on(probe):
                    continue
                score = probe.distance_squared(self.core_pos) if self.core_pos is not None else 0
                if self.enemy_estimate is not None:
                    enemy_score = probe.distance_squared(self.enemy_estimate)
                    score += enemy_score if not prefer_enemy else -enemy_score
                candidates.append((score, probe))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def find_forward_offense_position(self, ct: Controller) -> Position | None:
        candidates = []
        for pos, building_info in self.known_buildings.items():
            if building_info is None or building_info[1] != self.team:
                continue
            if building_info[0] not in TRANSPORT_BUILDINGS and building_info[0] != EntityType.FOUNDRY:
                continue
            for probe in self.adjacent_tiles(pos):
                if not self.can_place_combat_building_on(probe):
                    continue
                score = probe.distance_squared(self.enemy_estimate) if self.enemy_estimate is not None else 0
                candidates.append((score, probe))
        if not candidates:
            return self.find_ring_build_position(ct, prefer_enemy=True)
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def find_barrier_position(self) -> Position | None:
        if self.enemy_estimate is None:
            return None
        candidates = []
        for harvester_pos in self.allied_harvester_positions():
            direction = harvester_pos.direction_to(self.enemy_estimate)
            probe = harvester_pos.add(direction)
            if self.can_place_combat_building_on(probe):
                score = probe.distance_squared(self.core_pos) if self.core_pos is not None else 0
                candidates.append((score, probe))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def allied_harvester_positions(self) -> list[Position]:
        positions = []
        for pos, building_info in self.known_buildings.items():
            if building_info is None:
                continue
            if building_info[0] == EntityType.HARVESTER and building_info[1] == self.team:
                positions.append(pos)
        return positions

    def can_place_combat_building_on(self, pos: Position) -> bool:
        if not (0 <= pos.x < self.map_width and 0 <= pos.y < self.map_height):
            return False
        env = self.known_env.get(pos)
        if env == Environment.WALL or env in ORE_TYPES:
            return False
        return self.known_buildings.get(pos) is None

    def core_contact_tiles(self) -> list[Position]:
        if self.core_pos is None:
            return []
        positions = []
        for dy in range(-1, 2):
            positions.append(Position(self.core_pos.x - 2, self.core_pos.y + dy))
            positions.append(Position(self.core_pos.x + 2, self.core_pos.y + dy))
        for dx in range(-1, 2):
            positions.append(Position(self.core_pos.x + dx, self.core_pos.y - 2))
            positions.append(Position(self.core_pos.x + dx, self.core_pos.y + 2))
        unique = []
        seen = set()
        for pos in positions:
            if (pos.x, pos.y) in seen:
                continue
            if 0 <= pos.x < self.map_width and 0 <= pos.y < self.map_height:
                seen.add((pos.x, pos.y))
                unique.append(pos)
        return unique

    def is_core_contact_tile(self, pos: Position) -> bool:
        if self.core_pos is None:
            return False
        dx = abs(pos.x - self.core_pos.x)
        dy = abs(pos.y - self.core_pos.y)
        return (dx == 2 and dy <= 1) or (dy == 2 and dx <= 1)

    def is_adjacent_to_core(self, pos: Position) -> bool:
        if self.core_pos is None:
            return False
        for contact in self.core_contact_tiles():
            if pos.distance_squared(contact) == 1:
                return True
        return False

    def core_input_target(self, contact: Position) -> Position:
        if self.core_pos is None:
            return contact
        step_x = 0 if contact.x == self.core_pos.x else (1 if contact.x > self.core_pos.x else -1)
        step_y = 0 if contact.y == self.core_pos.y else (1 if contact.y > self.core_pos.y else -1)
        return Position(contact.x - step_x, contact.y - step_y)

    def direction_from_pos_to_enemy(self, pos: Position) -> Direction:
        target = self.find_enemy_core_position_from_known() or self.enemy_estimate
        if target is None:
            return Direction.NORTH
        direction = pos.direction_to(target)
        if direction == Direction.CENTRE:
            return Direction.NORTH
        return direction

    def find_enemy_core_position(self, ct: Controller) -> Position | None:
        for building_id in ct.get_nearby_buildings():
            if ct.get_entity_type(building_id) == EntityType.CORE and ct.get_team(building_id) != self.team:
                return ct.get_position(building_id)
        return self.find_enemy_core_position_from_known()

    def find_enemy_core_position_from_known(self) -> Position | None:
        for pos, building_info in self.known_buildings.items():
            if building_info is None:
                continue
            if building_info[0] == EntityType.CORE and building_info[1] != self.team:
                return pos
        return None

    def try_builder_attack_on_enemy_core(self, ct: Controller) -> bool:
        enemy_core = self.find_enemy_core_position(ct)
        if enemy_core is None:
            return False
        if ct.get_position() != enemy_core:
            return False
        if ct.can_fire(enemy_core):
            ct.fire(enemy_core)
            return True
        return False

    def is_allied_foundry_at(self, pos: Position) -> bool:
        building_info = self.known_buildings.get(pos)
        return building_info is not None and building_info[0] == EntityType.FOUNDRY and building_info[1] == self.team

    def safe_get_stored_resource(self, ct: Controller, building_id: int) -> ResourceType | None:
        try:
            return ct.get_stored_resource(building_id)
        except Exception:
            return None

    def safe_get_direction(self, ct: Controller, building_id: int) -> Direction | None:
        try:
            return ct.get_direction(building_id)
        except Exception:
            return None

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
        best_score = 10**9
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


player = Player()


def run(c):
    player.run(c)
