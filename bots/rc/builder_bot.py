from __future__ import annotations

from collections import deque

from cambc import Controller, Direction, EntityType, Environment, GameConstants, Position

from base import BaseBot
from constants import ASSIGNMENT_DIRECTIONS, CARDINALS, DIRECTIONS, PASSABLE_BUILDINGS
from geometry import direction_index, direction_to_vector, rotate_left


PHASE_EXPLORE_TITANIUM = 1
PHASE_BUILD_TITANIUM = 2
PHASE_EXPLORE_EXPANDED = 3
PHASE_BUILD_AXIONITE = 4

ROADLIKE_BUILDINGS = PASSABLE_BUILDINGS | {EntityType.HARVESTER}


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


class BuilderBot(BaseBot):
    def __init__(self, map_width: int, map_height: int) -> None:
        super().__init__(map_width, map_height)
        self.core_pos: Position | None = None
        self.assigned_direction: Direction | None = None
        self.assigned_index: int | None = None
        self.team = None
        self.phase = PHASE_EXPLORE_TITANIUM

        self.known_tiles: set[Position] = set()
        self.known_walls: set[Position] = set()
        self.known_titanium: set[Position] = set()
        self.known_axionite: set[Position] = set()
        self.known_buildings: dict[Position, tuple[EntityType, Direction | None] | None] = {}
        self.known_building_teams: dict[Position, object | None] = {}

        self.scan_depth = 1
        self.max_distance_reached = 0

        self.titanium_plan: dict[Position, tuple[str, Direction | None]] = {}
        self.axionite_plan: dict[Position, tuple[str, Direction | None]] = {}
        self.foundry_plan: FoundryPlan | None = None
        self.target_tile: Position | None = None

    def run(self, c: Controller) -> None:
        super().run(c)
        self.observe(c)
        if self.team is None:
            self.team = c.get_team()

        if self.core_pos is None:
            self.core_pos = self.find_home_core(c)
        if self.core_pos is None:
            return

        if self.assigned_direction is None:
            self.read_assignment(c)
        if self.assigned_direction is None:
            return

        self.max_distance_reached = max(self.max_distance_reached, self.core_distance(c.get_position()))
        self.update_phase(c)

        if self.phase in {PHASE_EXPLORE_TITANIUM, PHASE_EXPLORE_EXPANDED}:
            self.run_exploration_phase(c)
            return

        if self.phase == PHASE_BUILD_TITANIUM:
            self.run_titanium_network_phase(c)
            return

        self.run_axionite_network_phase(c)

    def update_phase(self, c: Controller) -> None:
        if self.phase == PHASE_EXPLORE_TITANIUM:
            if self.is_titanium_exploration_complete():
                self.phase = PHASE_BUILD_TITANIUM
                self.titanium_plan = {}
                self.target_tile = None
            return

        if self.phase == PHASE_BUILD_TITANIUM:
            if self.foundry_is_affordable(c):
                self.phase = PHASE_BUILD_AXIONITE
                self.axionite_plan = {}
                self.foundry_plan = None
                self.target_tile = None
                return
            if not self.has_pending_titanium_network():
                self.phase = PHASE_EXPLORE_EXPANDED
                self.target_tile = None
            return

        if self.phase == PHASE_EXPLORE_EXPANDED:
            if self.foundry_is_affordable(c):
                self.phase = PHASE_BUILD_AXIONITE
                self.axionite_plan = {}
                self.foundry_plan = None
                self.target_tile = None
                return
            if self.has_pending_titanium_network():
                self.phase = PHASE_BUILD_TITANIUM
                self.titanium_plan = {}
                self.target_tile = None
            return

        if self.phase == PHASE_BUILD_AXIONITE and not self.known_axionite_ores():
            self.target_tile = None

    def is_titanium_exploration_complete(self) -> bool:
        ore_count = len(self.sector_titanium_ores())
        return ore_count >= 1 and self.max_distance_reached > 5

    def foundry_is_affordable(self, c: Controller) -> bool:
        titanium, axionite = c.get_global_resources()
        foundry_titanium, foundry_axionite = c.get_foundry_cost()
        return titanium >= foundry_titanium and axionite >= foundry_axionite

    def run_exploration_phase(self, c: Controller) -> None:
        if self.try_build_nearby_titanium_harvester(c):
            return
        if self.try_build_road_underfoot(c):
            return
        target = self.choose_scan_target()
        if target is None:
            return
        self.target_tile = target
        self.move_towards_any(c, {target}, allow_ore=False, pave_roads=True)

    def try_build_nearby_titanium_harvester(self, c: Controller) -> bool:
        for ore in sorted(self.sector_titanium_ores(), key=lambda pos: c.get_position().distance_squared(pos)):
            if self.is_harvester_built(ore):
                continue
            if c.get_position().distance_squared(ore) <= GameConstants.ACTION_RADIUS_SQ and c.can_build_harvester(ore):
                c.build_harvester(ore)
                self.known_buildings[ore] = (EntityType.HARVESTER, None)
                self.known_building_teams[ore] = self.team
                return True
        return False

    def try_build_road_underfoot(self, c: Controller) -> bool:
        pos = c.get_position()
        if not self.should_build_road(pos):
            return False
        if c.can_build_road(pos):
            c.build_road(pos)
            self.known_buildings[pos] = (EntityType.ROAD, None)
            self.known_building_teams[pos] = self.team
            return True
        return False

    def run_titanium_network_phase(self, c: Controller) -> None:
        if not self.has_pending_titanium_network():
            self.phase = PHASE_EXPLORE_EXPANDED
            self.target_tile = None
            return

        if not self.titanium_plan:
            self.titanium_plan = self.plan_conveyor_network(self.sector_titanium_ores())
            if not self.titanium_plan:
                self.phase = PHASE_EXPLORE_EXPANDED
                return

        build_target = self.choose_next_build_target(c, self.titanium_plan)
        if build_target is None:
            self.titanium_plan = {}
            if not self.has_pending_titanium_network():
                self.phase = PHASE_EXPLORE_EXPANDED
            return

        task_type, direction = self.titanium_plan[build_target]
        if c.get_position().distance_squared(build_target) <= GameConstants.ACTION_RADIUS_SQ:
            if self.try_execute_titanium_task(c, build_target, task_type, direction):
                return

        if task_type == "harvester":
            goals = self.harvester_stand_positions(build_target)
            if goals:
                self.move_towards_any(c, goals, allow_ore=False, pave_roads=True)
                return

        self.move_towards_any(c, {build_target}, allow_ore=False, pave_roads=True)

    def run_axionite_network_phase(self, c: Controller) -> None:
        if self.foundry_plan is None:
            self.foundry_plan = self.find_foundry_plan()

        if self.foundry_plan is not None:
            if not self.is_allied_foundry_at(self.foundry_plan.foundry_pos):
                if self.try_build_foundry(c, self.foundry_plan.foundry_pos):
                    return
            elif self.ensure_foundry_links(c, self.foundry_plan):
                return

        if not self.known_axionite_ores():
            self.run_exploration_phase(c)
            return

        if not self.axionite_plan:
            self.axionite_plan = self.plan_road_network(self.known_axionite_ores())
            if not self.axionite_plan:
                return

        build_target = self.choose_next_build_target(c, self.axionite_plan)
        if build_target is None:
            self.axionite_plan = self.plan_road_network(self.known_axionite_ores())
            return

        task_type, _ = self.axionite_plan[build_target]
        if c.get_position().distance_squared(build_target) <= GameConstants.ACTION_RADIUS_SQ:
            if self.try_execute_axionite_task(c, build_target, task_type):
                return

        if task_type == "harvester":
            goals = self.harvester_stand_positions(build_target)
            if goals:
                self.move_towards_any(c, goals, allow_ore=False, pave_roads=True)
                return

        self.move_towards_any(c, {build_target}, allow_ore=False, pave_roads=True)

    def read_assignment(self, c: Controller) -> None:
        if self.core_pos is None:
            return

        spawn_lane = self.core_pos.direction_to(c.get_position())
        if spawn_lane == Direction.CENTRE or spawn_lane not in ASSIGNMENT_DIRECTIONS:
            return

        self.assigned_index = direction_index(spawn_lane)
        self.assigned_direction = ASSIGNMENT_DIRECTIONS[self.assigned_index]

    def choose_scan_target(self) -> Position | None:
        if self.core_pos is None or self.assigned_direction is None:
            return None

        forward_x, forward_y = direction_to_vector(self.assigned_direction)
        side_x, side_y = direction_to_vector(rotate_left(self.assigned_direction))

        max_depth = max(self.map_width, self.map_height)
        while self.scan_depth <= max_depth:
            depth = self.scan_depth
            width = depth * 2 + 1
            offsets = self.row_offsets(width)

            for lateral in offsets:
                target = Position(
                    self.core_pos.x + forward_x * depth + side_x * lateral,
                    self.core_pos.y + forward_y * depth + side_y * lateral,
                )
                if not self.in_bounds(target) or not self.is_in_sector(target):
                    continue
                if self.is_row_tile_complete(target):
                    continue
                return target

            self.scan_depth += 1

        return self.edge_target()

    def row_offsets(self, width: int) -> list[int]:
        offsets = [0]
        half = width // 2
        for delta in range(1, half + 1):
            offsets.append(delta)
            offsets.append(-delta)
        return offsets

    def is_row_tile_complete(self, pos: Position) -> bool:
        if pos in self.known_walls or pos in self.known_titanium or pos in self.known_axionite:
            return True

        building = self.known_buildings.get(pos)
        if building is None:
            return False
        return building[0] in ROADLIKE_BUILDINGS

    def edge_target(self) -> Position | None:
        if self.core_pos is None or self.assigned_direction is None:
            return None
        dx, dy = direction_to_vector(self.assigned_direction)
        pos = self.core_pos
        while True:
            nxt = Position(pos.x + dx, pos.y + dy)
            if not self.in_bounds(nxt):
                return pos
            pos = nxt

    def move_towards_any(self, c: Controller, goals: set[Position], allow_ore: bool, pave_roads: bool) -> None:
        current = c.get_position()
        path = self.find_path(current, goals, allow_ore=allow_ore)
        if path:
            next_pos = path[0]
            if self.try_prepare_or_move(c, next_pos, pave_roads):
                return

        detour = self.choose_right_hand_step(c, goals, allow_ore)
        if detour is not None:
            self.try_prepare_or_move(c, detour, pave_roads)

    def choose_right_hand_step(self, c: Controller, goals: set[Position], allow_ore: bool) -> Position | None:
        current = c.get_position()
        primary = self.assigned_direction or Direction.NORTH
        if self.target_tile is not None:
            primary = current.direction_to(self.target_tile)
            if primary == Direction.CENTRE:
                primary = self.assigned_direction or Direction.NORTH

        ordered = []
        direction = primary
        for _ in range(8):
            ordered.append(direction)
            direction = direction.rotate_right()

        for direction in ordered:
            if direction == Direction.CENTRE:
                continue
            nxt = current.add(direction)
            if not self.in_bounds(nxt):
                continue
            if not self.can_travel_through(nxt, allow_ore, goals):
                continue
            return nxt
        return None

    def try_prepare_or_move(self, c: Controller, next_pos: Position, pave_roads: bool) -> bool:
        current = c.get_position()
        if current == next_pos:
            return False

        step = current.direction_to(next_pos)
        if step == Direction.CENTRE:
            return False

        if pave_roads and self.should_build_road(next_pos):
            if c.can_build_road(next_pos):
                c.build_road(next_pos)
                self.known_buildings[next_pos] = (EntityType.ROAD, None)
                return True

        if c.can_move(step):
            c.move(step)
            return True

        return False

    def should_build_road(self, pos: Position) -> bool:
        if pos in self.known_titanium or pos in self.known_axionite or pos in self.known_walls:
            return False
        building = self.known_buildings.get(pos)
        if building is None:
            return True
        return building[0] not in ROADLIKE_BUILDINGS

    def choose_next_build_target(
        self,
        c: Controller,
        plan: dict[Position, tuple[str, Direction | None]],
    ) -> Position | None:
        current = c.get_position()
        best = None
        best_score = 10**9
        for pos, task in plan.items():
            if self.is_task_done(pos, task[0], task[1]):
                continue
            goals = self.harvester_stand_positions(pos) if task[0] == "harvester" else {pos}
            path = self.find_path(current, goals, allow_ore=False)
            score = len(path) if path else current.distance_squared(pos)
            if score < best_score:
                best_score = score
                best = pos
        return best

    def is_task_done(self, pos: Position, task_type: str, direction: Direction | None) -> bool:
        building_info = self.known_buildings.get(pos)
        if task_type == "harvester":
            return building_info is not None and building_info[0] == EntityType.HARVESTER
        if task_type == "conveyor":
            return building_info is not None and building_info[0] == EntityType.CONVEYOR and building_info[1] == direction
        if task_type == "road":
            return building_info is not None and building_info[0] in ROADLIKE_BUILDINGS
        return False

    def try_execute_titanium_task(
        self,
        c: Controller,
        pos: Position,
        task_type: str,
        direction: Direction | None,
    ) -> bool:
        building_id = c.get_tile_building_id(pos)

        if task_type == "harvester":
            if c.can_build_harvester(pos):
                c.build_harvester(pos)
                self.known_buildings[pos] = (EntityType.HARVESTER, None)
                return True
            return False

        if building_id is not None:
            entity_type = c.get_entity_type(building_id)
            if entity_type == EntityType.CONVEYOR and c.get_direction(building_id) == direction:
                self.known_buildings[pos] = (EntityType.CONVEYOR, direction)
                return False
            if entity_type in (EntityType.ROAD, EntityType.CONVEYOR):
                if c.can_destroy(pos):
                    c.destroy(pos)
                    self.known_buildings[pos] = None
                    return True
                return False
            return False

        if direction is not None and c.can_build_conveyor(pos, direction):
            c.build_conveyor(pos, direction)
            self.known_buildings[pos] = (EntityType.CONVEYOR, direction)
            return True

        if c.can_build_road(pos):
            c.build_road(pos)
            self.known_buildings[pos] = (EntityType.ROAD, None)
            return True

        return False

    def try_execute_axionite_task(self, c: Controller, pos: Position, task_type: str) -> bool:
        if task_type == "harvester":
            if c.can_build_harvester(pos):
                c.build_harvester(pos)
                self.known_buildings[pos] = (EntityType.HARVESTER, None)
                return True
            return False

        if c.can_build_road(pos):
            c.build_road(pos)
            self.known_buildings[pos] = (EntityType.ROAD, None)
            return True
        return False

    def try_build_foundry(self, c: Controller, pos: Position) -> bool:
        if c.get_position().distance_squared(pos) > GameConstants.ACTION_RADIUS_SQ:
            return self.move_towards_build_tile(c, pos)
        building_id = c.get_tile_building_id(pos)
        if building_id is not None:
            if c.get_entity_type(building_id) == EntityType.FOUNDRY and c.get_team(building_id) == self.team:
                self.known_buildings[pos] = (EntityType.FOUNDRY, None)
                self.known_building_teams[pos] = self.team
                return False
            if c.can_destroy(pos):
                c.destroy(pos)
                self.known_buildings[pos] = None
                self.known_building_teams[pos] = None
                return True
            return False
        if c.can_build_foundry(pos):
            c.build_foundry(pos)
            self.known_buildings[pos] = (EntityType.FOUNDRY, None)
            self.known_building_teams[pos] = self.team
            return True
        return False

    def harvester_stand_positions(self, ore: Position) -> set[Position]:
        goals = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                pos = Position(ore.x + dx, ore.y + dy)
                if not self.in_bounds(pos):
                    continue
                if pos in self.known_walls or pos in self.known_titanium or pos in self.known_axionite:
                    continue
                building = self.known_buildings.get(pos)
                if building is not None and building[0] not in PASSABLE_BUILDINGS:
                    continue
                goals.add(pos)
        return goals

    def plan_conveyor_network(self, ores: list[Position]) -> dict[Position, tuple[str, Direction | None]]:
        if self.core_pos is None:
            return {}

        targets = sorted(
            [ore for ore in ores if not self.is_harvester_built(ore)],
            key=lambda pos: (self.core_distance(pos), pos.x, pos.y),
        )
        if not targets:
            return {}

        network_anchors = set(self.core_tiles())
        plan: dict[Position, tuple[str, Direction | None]] = {}
        remaining = list(targets)

        while remaining:
            best_ore = None
            best_path = None
            best_anchor = None
            best_approach = None
            best_score = 10**9

            for ore in remaining:
                for approach in self.ore_approaches(ore):
                    path = self.find_cardinal_path(approach, network_anchors, ore)
                    if path is None:
                        continue
                    anchor = path[1] if len(path) > 1 else self.closest_anchor(approach, network_anchors)
                    if anchor is None:
                        continue
                    if approach.direction_to(anchor) == approach.direction_to(ore):
                        continue
                    if len(path) < best_score:
                        best_score = len(path)
                        best_ore = ore
                        best_path = path
                        best_anchor = anchor
                        best_approach = approach

            if best_ore is None or best_path is None or best_approach is None or best_anchor is None:
                break

            plan[best_ore] = ("harvester", None)
            parent = best_anchor
            for tile in reversed(best_path[:-1]):
                plan[tile] = ("conveyor", tile.direction_to(parent))
                network_anchors.add(tile)
                parent = tile
            network_anchors.add(best_approach)
            remaining.remove(best_ore)

        return plan

    def plan_road_network(self, ores: list[Position]) -> dict[Position, tuple[str, Direction | None]]:
        if self.core_pos is None:
            return {}

        targets = sorted(
            [ore for ore in ores if not self.is_harvester_built(ore)],
            key=lambda pos: (self.core_distance(pos), pos.x, pos.y),
        )
        if not targets:
            return {}

        anchors = set(self.core_tiles())
        plan: dict[Position, tuple[str, Direction | None]] = {}
        remaining = list(targets)

        while remaining:
            best_ore = None
            best_path = None
            best_score = 10**9

            for ore in remaining:
                for approach in self.ore_approaches(ore):
                    path = self.find_cardinal_path(approach, anchors, ore)
                    if path is None:
                        continue
                    if len(path) < best_score:
                        best_score = len(path)
                        best_ore = ore
                        best_path = path

            if best_ore is None or best_path is None:
                break

            plan[best_ore] = ("harvester", None)
            for tile in best_path[:-1]:
                plan[tile] = ("road", None)
                anchors.add(tile)
            remaining.remove(best_ore)

        return plan

    def find_foundry_plan(self) -> FoundryPlan | None:
        if self.core_pos is None:
            return None

        titanium_tiles = self.find_resource_input_tiles(self.sector_titanium_ores())
        axionite_tiles = self.find_resource_input_tiles(self.known_axionite_ores())
        if not titanium_tiles or not axionite_tiles:
            return None

        best_plan = None
        best_score = 10**9
        for titanium_pos in titanium_tiles[:6]:
            for axionite_pos in axionite_tiles[:6]:
                if titanium_pos == axionite_pos:
                    continue
                common_neighbors = set(self.cardinal_adjacent_tiles(titanium_pos)) & set(self.cardinal_adjacent_tiles(axionite_pos))
                for foundry_pos in common_neighbors:
                    if not self.can_place_foundry_on(foundry_pos):
                        continue
                    output_pos, output_target = self.find_foundry_output(foundry_pos, titanium_pos, axionite_pos)
                    if output_pos is None or output_target is None:
                        continue
                    score = self.core_distance(foundry_pos)
                    score += self.core_distance(titanium_pos)
                    score += self.core_distance(axionite_pos)
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

    def find_resource_input_tiles(self, ores: list[Position]) -> list[Position]:
        candidates = []
        core_tiles = self.core_adjacent_tiles()
        for pos in core_tiles:
            if pos in self.known_walls or pos in self.known_titanium or pos in self.known_axionite:
                continue
            building = self.known_buildings.get(pos)
            team = self.known_building_teams.get(pos)
            if building is not None and team is not None and team != self.team:
                continue
            score = min((pos.distance_squared(ore) for ore in ores), default=10**6)
            if building is not None and building[0] in {EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.ROAD}:
                score -= 20
            candidates.append((score, pos))
        candidates.sort(key=lambda item: item[0])
        return [pos for _, pos in candidates]

    def can_place_foundry_on(self, pos: Position) -> bool:
        if not self.in_bounds(pos):
            return False
        if pos in self.known_walls or pos in self.known_titanium or pos in self.known_axionite:
            return False
        building = self.known_buildings.get(pos)
        team = self.known_building_teams.get(pos)
        return building is None or team == self.team and building[0] in PASSABLE_BUILDINGS

    def find_foundry_output(
        self,
        foundry_pos: Position,
        titanium_pos: Position,
        axionite_pos: Position,
    ) -> tuple[Position | None, Position | None]:
        for output_pos in self.core_adjacent_tiles():
            if output_pos in {titanium_pos, axionite_pos}:
                continue
            if output_pos.distance_squared(foundry_pos) != 1:
                continue
            return output_pos, self.core_pos
        return None, None

    def ensure_foundry_links(self, c: Controller, plan: FoundryPlan) -> bool:
        splitter_direction = plan.titanium_input.direction_to(self.core_pos)
        left_target = plan.titanium_input.add(splitter_direction.rotate_left())
        right_target = plan.titanium_input.add(splitter_direction.rotate_right())
        if plan.foundry_pos in {left_target, right_target}:
            if self.ensure_transport_building(c, plan.titanium_input, splitter_direction, use_splitter=True):
                return True
        else:
            if self.ensure_transport_building(
                c,
                plan.titanium_input,
                plan.titanium_input.direction_to(plan.foundry_pos),
                use_splitter=False,
            ):
                return True

        if self.ensure_transport_building(
            c,
            plan.axionite_input,
            plan.axionite_input.direction_to(plan.foundry_pos),
            use_splitter=False,
        ):
            return True

        if plan.output_pos is not None and plan.output_target is not None:
            if self.ensure_transport_building(
                c,
                plan.output_pos,
                plan.output_pos.direction_to(plan.output_target),
                use_splitter=False,
            ):
                return True

        return False

    def ensure_transport_building(
        self,
        c: Controller,
        pos: Position,
        direction: Direction,
        use_splitter: bool,
    ) -> bool:
        if c.get_position().distance_squared(pos) > GameConstants.ACTION_RADIUS_SQ:
            return self.move_towards_build_tile(c, pos)

        building_id = c.get_tile_building_id(pos)
        if building_id is not None:
            entity_type = c.get_entity_type(building_id)
            team = c.get_team(building_id)
            existing_direction = None
            if entity_type in {EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.ARMOURED_CONVEYOR}:
                existing_direction = c.get_direction(building_id)
            if team == self.team:
                if use_splitter and entity_type == EntityType.SPLITTER and existing_direction == direction:
                    return False
                if not use_splitter and entity_type == EntityType.CONVEYOR and existing_direction == direction:
                    return False
            if entity_type == EntityType.FOUNDRY:
                return False
            if c.can_destroy(pos):
                c.destroy(pos)
                self.known_buildings[pos] = None
                self.known_building_teams[pos] = None
                return True
            return False

        if use_splitter:
            if c.can_build_splitter(pos, direction):
                c.build_splitter(pos, direction)
                self.known_buildings[pos] = (EntityType.SPLITTER, direction)
                self.known_building_teams[pos] = self.team
                return True
            return False

        if c.can_build_conveyor(pos, direction):
            c.build_conveyor(pos, direction)
            self.known_buildings[pos] = (EntityType.CONVEYOR, direction)
            self.known_building_teams[pos] = self.team
            return True
        return False

    def core_tiles(self) -> list[Position]:
        if self.core_pos is None:
            return []
        tiles = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                pos = Position(self.core_pos.x + dx, self.core_pos.y + dy)
                if self.in_bounds(pos):
                    tiles.append(pos)
        return tiles

    def core_adjacent_tiles(self) -> list[Position]:
        if self.core_pos is None:
            return []
        tiles = []
        for direction in DIRECTIONS:
            pos = self.core_pos.add(direction)
            if self.in_bounds(pos):
                tiles.append(pos)
        return tiles

    def cardinal_adjacent_tiles(self, pos: Position) -> list[Position]:
        return [pos.add(direction) for direction in CARDINALS if self.in_bounds(pos.add(direction))]

    def closest_anchor(self, origin: Position, anchors: set[Position]) -> Position | None:
        best = None
        best_score = 10**9
        for anchor in anchors:
            score = origin.distance_squared(anchor)
            if score < best_score:
                best_score = score
                best = anchor
        return best

    def ore_approaches(self, ore: Position) -> list[Position]:
        result = []
        for direction in CARDINALS:
            pos = ore.add(direction)
            if not self.in_bounds(pos):
                continue
            if pos in self.known_walls or pos in self.known_titanium or pos in self.known_axionite:
                continue
            building = self.known_buildings.get(pos)
            if building is not None and building[0] not in PASSABLE_BUILDINGS:
                continue
            result.append(pos)
        result.sort(key=lambda pos: (self.core_distance(pos), pos.x, pos.y))
        return result

    def find_cardinal_path(self, start: Position, goals: set[Position], blocked_ore: Position) -> list[Position] | None:
        if start in goals:
            return [start]

        queue = deque([start])
        came_from = {start: start}

        while queue:
            current = queue.popleft()
            for direction in CARDINALS:
                nxt = current.add(direction)
                if not self.in_bounds(nxt) or nxt in came_from:
                    continue
                if nxt != blocked_ore and (nxt in self.known_titanium or nxt in self.known_axionite):
                    continue
                if nxt in self.known_walls:
                    continue
                building = self.known_buildings.get(nxt)
                if building is not None and building[0] not in PASSABLE_BUILDINGS and nxt not in goals:
                    continue
                came_from[nxt] = current
                if nxt in goals:
                    path = [nxt]
                    while path[-1] != start:
                        path.append(came_from[path[-1]])
                    path.reverse()
                    return path
                queue.append(nxt)
        return None

    def find_path(self, start: Position, goals: set[Position], allow_ore: bool) -> list[Position]:
        queue = deque([start])
        came_from = {start: start}

        while queue:
            current = queue.popleft()
            if current in goals:
                path = []
                while current != start:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for direction in DIRECTIONS:
                nxt = current.add(direction)
                if not self.in_bounds(nxt) or nxt in came_from:
                    continue
                if not self.can_travel_through(nxt, allow_ore, goals):
                    continue
                came_from[nxt] = current
                queue.append(nxt)
        return []

    def move_towards_build_tile(self, c: Controller, target: Position) -> bool:
        goals = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                pos = Position(target.x + dx, target.y + dy)
                if pos == target or not self.in_bounds(pos):
                    continue
                if self.can_travel_through(pos, allow_ore=False, goals={pos}):
                    goals.add(pos)
        if not goals:
            return False
        self.move_towards_any(c, goals, allow_ore=False, pave_roads=True)
        return True

    def can_travel_through(self, pos: Position, allow_ore: bool, goals: set[Position]) -> bool:
        if pos in self.known_walls:
            return False
        if pos in goals:
            return True
        if pos in self.known_titanium or pos in self.known_axionite:
            return allow_ore
        building = self.known_buildings.get(pos)
        if building is None:
            return True
        return building[0] in PASSABLE_BUILDINGS

    def sector_titanium_ores(self) -> list[Position]:
        return [pos for pos in self.known_titanium if self.is_in_sector(pos)]

    def known_axionite_ores(self) -> list[Position]:
        return [pos for pos in self.known_axionite if self.is_in_sector(pos)]

    def has_unconnected_titanium_ores(self) -> bool:
        for ore in self.sector_titanium_ores():
            if not self.is_harvester_built(ore):
                return True
        return False

    def has_pending_titanium_network(self) -> bool:
        if self.titanium_plan:
            for pos, task in self.titanium_plan.items():
                if not self.is_task_done(pos, task[0], task[1]):
                    return True
        return bool(self.plan_conveyor_network(self.sector_titanium_ores()))

    def is_harvester_built(self, pos: Position) -> bool:
        building = self.known_buildings.get(pos)
        return building is not None and building[0] == EntityType.HARVESTER

    def is_in_sector(self, pos: Position) -> bool:
        if self.core_pos is None or self.assigned_direction is None:
            return False
        fx, fy = direction_to_vector(self.assigned_direction)
        sx, sy = direction_to_vector(rotate_left(self.assigned_direction))
        rel_x = pos.x - self.core_pos.x
        rel_y = pos.y - self.core_pos.y
        forward = rel_x * fx + rel_y * fy
        side = rel_x * sx + rel_y * sy
        return forward > 0 and abs(side) <= forward + 1

    def core_distance(self, pos: Position) -> int:
        if self.core_pos is None:
            return 10**9
        return abs(pos.x - self.core_pos.x) + abs(pos.y - self.core_pos.y)

    def observe(self, c: Controller) -> None:
        for pos in c.get_nearby_tiles():
            self.known_tiles.add(pos)
            env = c.get_tile_env(pos)
            if env == Environment.WALL:
                self.known_walls.add(pos)
            elif env == Environment.ORE_TITANIUM:
                self.known_titanium.add(pos)
            elif env == Environment.ORE_AXIONITE:
                self.known_axionite.add(pos)

            building_id = c.get_tile_building_id(pos)
            if building_id is None:
                self.known_buildings[pos] = None
                self.known_building_teams[pos] = None
                continue

            entity_type = c.get_entity_type(building_id)
            team = c.get_team(building_id)
            direction = None
            if entity_type in {
                EntityType.CONVEYOR,
                EntityType.ARMOURED_CONVEYOR,
                EntityType.SPLITTER,
                EntityType.GUNNER,
                EntityType.SENTINEL,
                EntityType.BREACH,
            }:
                direction = c.get_direction(building_id)
            self.known_buildings[pos] = (entity_type, direction)
            self.known_building_teams[pos] = team

    def find_home_core(self, c: Controller) -> Position | None:
        for entity_id in c.get_nearby_entities():
            if c.get_entity_type(entity_id) == EntityType.CORE:
                return c.get_position(entity_id)
        return None

    def is_allied_foundry_at(self, pos: Position) -> bool:
        building = self.known_buildings.get(pos)
        team = self.known_building_teams.get(pos)
        return building is not None and building[0] == EntityType.FOUNDRY and team == self.team
