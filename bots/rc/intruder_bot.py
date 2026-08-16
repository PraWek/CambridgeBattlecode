"""A fifth-turn BuilderBot role that infiltrates and supplies a forward gunner."""

from cambc import Controller, Direction, EntityType, Environment, GameConstants, Position

from base import BaseBot
from constants import (
    DIRECTIONS,
    MARKER_KIND_INTRUDER_LAUNCH,
    MARKER_KIND_SPAWN_INTRUDER,
    ORTHOGONAL_DIRECTIONS,
    PASSABLE_BUILDINGS,
)
from geometry import decode_marker, decode_marker_coordinates, encode_marker
from navigation import a_star_from_any_with_bridges, a_star_to_any


_CLEARABLE_WALKABLE_BUILDINGS = {EntityType.ROAD, EntityType.CONVEYOR}
# A BuilderBot's attack is restricted to the building on its own tile.  Do not
# spend many turns chewing through an enemy logistics line: two 2-damage shots
# are the most this opportunistic cleanup may require.
_CHEAP_ENEMY_WALKABLE_BUILDINGS = {
    EntityType.CONVEYOR,
    EntityType.SPLITTER,
    EntityType.BRIDGE,
}
_CHEAP_ENEMY_BUILDING_MAX_HP = GameConstants.BUILDER_BOT_ATTACK_DAMAGE * 2
_SUPPLY_SEARCH_EXPANSIONS = 96
_SUPPLY_BRIDGE_SEARCH_EXPANSIONS = 96
_SUPPLY_CONVEYOR_STEP_COST = 3
_SUPPLY_BRIDGE_STEP_COST = 20
_RETURN_TO_CORE_A_STAR_MAX_EXPANSIONS = 192
# ``can_fire_from`` crosses into the engine and is unusually costly compared
# with the local geometry work.  Validate a prepared Gunner candidate one at
# a time so an unsuccessful search cannot repeatedly consume the full turn.
_GUNNER_SITE_VALIDATIONS_PER_TURN = 1
# Preparing a Gunner candidate list also performs enough local geometry and
# allocation to exhaust a 2 ms turn after a symmetry has just been confirmed.
# Keep its cursor in the bot state and inspect only a small fixed batch each
# turn, so an interruption cannot restart the full enumeration forever.
_GUNNER_CANDIDATE_PREPARATIONS_PER_TURN = 48
# A bridge-aware A* search is intentionally fairly thorough, but running one
# for every visible titanium deposit in a single 2 ms turn is not viable on a
# fully explored map.  Evaluate the nearest candidates incrementally instead.
_SUPPLY_ROUTE_PLANS_PER_TURN = 1


class IntruderBot(BaseBot):
    """Cross the map, locate the opposing core, and establish a supplied gunner."""

    def __init__(self, map_width: int, map_height: int) -> None:
        """Initialize infiltration, launcher, and supply-line state."""
        super().__init__(map_width, map_height)
        self.known_env = self.tile_cache.environments
        self.known_buildings = self.tile_cache.buildings
        self.core_pos: Position | None = None
        # One target position throughout the role: first a symmetry guess,
        # then the confirmed opposing Core.  Only its certainty is separate.
        self.destination: Position | None = None
        self.destination_is_confirmed_core = False

        self.mode = "infiltrate"
        self.heading: Direction | None = None
        self.wall_following = False
        self.exploration_wall_heading: Direction | None = None
        self.exploration_wall_side: str | None = None
        self.last_position: Position | None = None
        self.previous_position: Position | None = None
        # Exploration never revisits a tile.  The only exception is the
        # explicit A* return route to the friendly Core after a dead end.
        self.visited_tiles: set[Position] = set()
        self.returning_to_core = False
        self.return_path: list[Position] = []
        self.waiting_launcher_origin: Position | None = None
        self.waiting_launcher_round: int | None = None

        self.gunner_site: Position | None = None
        self.gunner_direction: Direction | None = None
        self.rejected_gunner_sites: set[Position] = set()
        self.gunner_id: int | None = None
        self.gunner_site_candidates: list[
            tuple[Position, Direction, Position, bool]
        ] | None = None
        self.gunner_candidate_cursor = 0
        self.gunner_candidate_targets: tuple[Position, ...] | None = None
        self.gunner_candidate_destination: Position | None = None
        self.gunner_candidate_target_index = 0
        self.gunner_candidate_facing_index = 0
        self.gunner_candidate_distance = 0
        self.gunner_candidate_seen_sites: set[tuple[Position, Direction]] = set()
        self.gunner_candidate_builds: list[
            tuple[int, int, int, Position, Direction, Position, bool]
        ] = []

        self.supply_ore: Position | None = None
        self.supply_path: list[Position] = []
        self.supply_travel_path: list[Position] = []
        self.supply_travel_bridges: dict[Position, Position] = {}
        self.supply_travel_index = 0
        self.pending_supply_bridge: Position | None = None
        self.supply_directions: dict[Position, Direction] = {}
        self.supply_bridge_targets: dict[Position, Position] = {}
        self.supply_index = 0
        self.unavailable_titanium: set[Position] = set()
        self.supply_plan_candidates: list[Position] | None = None
        self.supply_plan_cursor = 0
        self.deferred_supply_candidates: list[Position] = []

    @staticmethod
    def claims_spawn(controller: Controller) -> bool:
        """Return whether the nearby core marker designates this newborn intruder."""
        current = controller.get_position()
        for entity_id in controller.get_nearby_buildings():
            if controller.get_entity_type(entity_id) != EntityType.MARKER:
                continue
            try:
                kind, x, y, _ = decode_marker_coordinates(
                    controller.get_marker_value(entity_id)
                )
            except Exception:
                continue
            if (
                kind == MARKER_KIND_SPAWN_INTRUDER
                and x == current.x
                and y == current.y
            ):
                return True
        return False

    def run(self, controller: Controller) -> None:
        """Execute one cached turn of infiltration, gunner construction, or supply."""
        if self._scan_turn(controller):
            return
        # Confirming a symmetry schedules the historical terrain backfill and
        # infers the opposite Core.  Do not immediately follow it with Gunner
        # planning, whose engine validation calls can otherwise turn this
        # already expensive transition into a TLE.
        current = self.get_cached_position()
        self.visited_tiles.update((current,))
        self.draw_visited_tile_indicators(controller, current)
        self.remember_position(current)
        if self.core_pos is None:
            self.core_pos = self.find_friendly_core()
        if self.core_pos is None:
            return

        self.update_enemy_knowledge()
        if self.clear_cheap_enemy_building(controller, current):
            self.draw_goal_indicator(controller, current)
            return
        if self.wait_for_launcher(controller, current):
            self.draw_goal_indicator(controller, current)
            return
        if self.returning_to_core:
            self.mode = "return_to_core"
            self.return_to_core(controller, current)
            self.draw_goal_indicator(controller, current)
            return
        if self.gunner_id is None:
            self.mode = "infiltrate"
            self.infiltrate(controller, current)
            self.draw_goal_indicator(controller, current)
            return

        self.mode = "supply_gunner"
        self.supply_gunner(controller, current)
        self.draw_goal_indicator(controller, current)

    def infiltrate(self, controller: Controller, current: Position) -> None:
        """Explore toward a Core hypothesis, then establish a forward Gunner."""
        if not self.destination_is_confirmed_core:
            self.advance_towards_unvisited_target(controller, self.destination)
            return
        self.build_forward_gunner(controller, current)

    def draw_goal_indicator(self, controller: Controller, current: Position) -> None:
        """Draw a red replay line from this intruder to its current destination."""
        if self.returning_to_core:
            target = self.core_pos
        elif self.gunner_id is not None and self.supply_ore is not None:
            target = self.supply_ore
        else:
            target = self.destination
        if target is not None:
            controller.draw_indicator_line(current, target, 255, 0, 0)

    def remember_position(self, current: Position) -> None:
        """Keep one previous tile for retreating from a launcher dead end."""
        if self.last_position is not None and current != self.last_position:
            self.previous_position = self.last_position
        self.last_position = current

    def draw_visited_tile_indicators(
            self,
            controller: Controller,
            current: Position,
    ) -> None:
        """Draw only this turn's position without making debug work grow forever.

        Replay indicators are cosmetic, while a BuilderBot has a 2 ms turn
        budget.  Re-emitting one dot for every historical step became more
        expensive every round and was enough to starve the supply planner just
        after it had built a forward Gunner.
        """
        controller.draw_indicator_dot(current, 255, 0, 0)

    def has_unvisited_exploration_exit(self, current: Position) -> bool:
        """Return whether an unvisited neighbouring tile can still be explored.

        This deliberately ignores transient restrictions such as a builder on
        the tile, an action cooldown, or an unavailable road build.  Those
        make a step impossible *this turn*, but do not prove that the Intruder
        is in a map dead end and therefore must return to its Core.
        """
        for direction in DIRECTIONS:
            candidate = self.tile_cache.neighbor(current, direction)
            if (
                candidate is None
                or candidate in self.visited_tiles
            ):
                continue
            environment = self.known_env.get(candidate)
            # The unscanned half of a newborn Intruder's first vision is not
            # evidence of a wall.  Wait for its next scan rather than turn
            # around before the tile can be classified.
            if environment in {
                Environment.WALL,
                Environment.ORE_TITANIUM,
                Environment.ORE_AXIONITE,
            }:
                continue
            building = self.known_buildings.get(candidate)
            if building is None:
                return True
            building_type, building_team = building
            if building_type in PASSABLE_BUILDINGS and (
                building_type != EntityType.CORE or building_team == self.team
            ):
                return True
        return False

    def advance_towards_unvisited_target(
            self,
            controller: Controller,
            target: Position | None,
    ) -> bool:
        """Explore toward ``target`` through new cells, then return from a dead end."""
        if target is None:
            return False
        current = self.get_cached_position()
        if self.exploration_wall_side is not None:
            # A wall-following mode ends only when direct forward progress is
            # again available, or when its chosen wall side has no unvisited
            # continuation.  Do not restart A*, switch sides, or take a wide
            # fallback while the wall can still be followed.
            if self.move_towards(
                controller,
                target,
                avoid_visited=True,
                forward_sector_only=True,
                require_closer=True,
            ):
                self.stop_exploration_wall_following()
                return True
            if self.continue_exploration_wall_following(
                controller,
                avoid_visited=True,
            ):
                return True
            self.stop_exploration_wall_following()
            return self.finish_unvisited_movement(controller, target, current)

        # Exploration deliberately stays local even after the enemy Core is
        # confirmed.  A Core tile itself is not traversable, so an A* whose
        # sole goal is that tile can never succeed; retrying its bounded
        # failure each turn exhausts the BuilderBot's CPU budget before the
        # wall-following fallback can act.
        # First probe only the forward sector.
        if self.move_towards(
            controller,
            target,
            avoid_visited=True,
            forward_sector_only=True,
            require_closer=True,
        ):
            return True
        # Choose one side of the obstacle.  Once a side can advance, it is
        # retained across turns until forward progress resumes or it ends.
        if self.start_exploration_wall_following(
            controller,
            target,
            side="left",
            avoid_visited=True,
        ):
            return True
        if self.start_exploration_wall_following(
            controller,
            target,
            side="right",
            avoid_visited=True,
        ):
            return True
        return self.finish_unvisited_movement(controller, target, current)

    def finish_unvisited_movement(
            self,
            controller: Controller,
            target: Position,
            current: Position,
    ) -> bool:
        """Use the broad fallback only after no wall-following route remains."""
        if self.move_towards(
            controller,
            target,
            avoid_visited=True,
            remaining_directional_spectrum=True,
        ):
            return True
        if not self.has_unvisited_exploration_exit(current):
            self.begin_return_to_core(current)
        return False

    def advance_towards_revisitable_target(
            self,
            controller: Controller,
            target: Position | None,
    ) -> bool:
        """Reach a selected target, allowing known tiles to be used again."""
        if target is None:
            return False
        current = self.get_cached_position()
        path = a_star_to_any(
            None,
            current,
            {target},
            lambda _controller, pos: self.is_roadable_position(pos),
            self.tile_cache.neighbor,
            movement_directions=DIRECTIONS,
            max_expansions=_SUPPLY_SEARCH_EXPANSIONS,
        )
        if path and self.try_move_step(
                controller,
                current.direction_to(path[0]),
                avoid_visited=False,
        ):
            return True
        return self.move_towards(controller, target)

    def begin_return_to_core(self, current: Position) -> None:
        """Plan a known-terrain A* route back to the friendly Core after a dead end."""
        if self.core_pos is None or current == self.core_pos:
            return
        path = a_star_to_any(
            None,
            current,
            {self.core_pos},
            lambda _controller, pos: self.return_path_traversable(pos),
            self.tile_cache.neighbor,
            movement_directions=DIRECTIONS,
            max_expansions=_RETURN_TO_CORE_A_STAR_MAX_EXPANSIONS,
        )
        if not path:
            return
        self.returning_to_core = True
        self.return_path = path
        self.heading = None

    def return_path_traversable(self, pos: Position) -> bool:
        """Allow A* to use known walkable ground while retracing a route to Core."""
        if not self.in_bounds(pos):
            return False
        env = self.known_env.get(pos)
        if env is None or env in {
            Environment.WALL,
            Environment.ORE_TITANIUM,
            Environment.ORE_AXIONITE,
        }:
            return False
        building = self.known_buildings.get(pos)
        return building is None or (
            building[0] in PASSABLE_BUILDINGS
            and (building[0] != EntityType.CORE or building[1] == self.team)
        )

    def return_to_core(self, controller: Controller, current: Position) -> None:
        """Follow the A* route to Core, permitting only this deliberate revisit."""
        if self.core_pos is None:
            self.returning_to_core = False
            self.return_path = []
            return
        if current == self.core_pos:
            self.returning_to_core = False
            self.return_path = []
            self.heading = None
            self.previous_position = None
            return
        while self.return_path and current == self.return_path[0]:
            self.return_path.pop(0)
        if not self.return_path:
            self.returning_to_core = False
            self.begin_return_to_core(current)
            return
        next_pos = self.return_path[0]
        direction = current.direction_to(next_pos)
        if current.distance_squared(next_pos) > 2 or direction == Direction.CENTRE:
            self.returning_to_core = False
            self.begin_return_to_core(current)
            return
        if self.try_move_step(controller, direction, avoid_visited=False):
            self.return_path.pop(0)
            self.heading = direction
            return
        # A moving builder or fresh obstacle may invalidate a cached segment.
        self.returning_to_core = False
        self.begin_return_to_core(current)

    def find_friendly_core(self) -> Position | None:
        """Locate the allied core solely from the tile-cache entity index."""
        for entity_id in self.tile_cache.visible_entity_ids:
            if (
                self.tile_cache.entity_type(entity_id) == EntityType.CORE
                and self.tile_cache.entity_team(entity_id) == self.team
            ):
                return self.tile_cache.entity_position(entity_id)
        return None

    def update_enemy_knowledge(self) -> None:
        """Read the observed or symmetry-inferred opposing Core from the cache."""
        for entity_id in self.tile_cache.visible_entity_ids:
            if (
                self.tile_cache.entity_type(entity_id) == EntityType.CORE
                and self.tile_cache.entity_team(entity_id) != self.team
            ):
                self.destination = self.tile_cache.entity_position(entity_id)
                self.destination_is_confirmed_core = True
                self.wall_following = False
                self.stop_exploration_wall_following()
                return

        inferred_core = self.tile_cache.enemy_core_position(self.team)
        if inferred_core is not None:
            self.destination = inferred_core
            self.destination_is_confirmed_core = True
            self.wall_following = False
            self.stop_exploration_wall_following()
            return

        if self.destination is None:
            # The first hypothesis is the 180-degree counterpart.  It is
            # replaced immediately once TileCache confirms the real symmetry.
            self.destination = self.tile_cache.mirrored_position(
                self.core_pos,
                "rotational",
            )

    def wait_for_launcher(self, controller: Controller, current: Position) -> bool:
        """Stay adjacent to a new launcher until it has one turn to throw this bot."""
        if self.waiting_launcher_origin is None or self.waiting_launcher_round is None:
            return False
        if current != self.waiting_launcher_origin:
            self.waiting_launcher_origin = None
            self.waiting_launcher_round = None
            self.wall_following = False
            return False
        if controller.get_current_round() <= self.waiting_launcher_round + 1:
            return True
        # The launcher acts after this older builder on the requested turn.
        # If it still could not throw us, stop waiting and bypass the wall.
        self.waiting_launcher_origin = None
        self.waiting_launcher_round = None
        self.wall_following = True
        return False

    def move_towards(
            self,
            controller: Controller,
            target: Position | None,
            avoid_visited: bool = False,
            forward_sector_only: bool = False,
            remaining_directional_spectrum: bool = False,
            require_closer: bool = False,
    ) -> bool:
        """Advance greedily, optionally avoiding explored cells, or follow a wall."""
        if target is None:
            return False
        current = self.get_cached_position()
        desired = current.direction_to(target)
        if desired == Direction.CENTRE:
            return False
        if forward_sector_only:
            directions = self.forward_sector_directions(desired)
        elif remaining_directional_spectrum:
            directions = self.remaining_exploration_directions(desired)
        else:
            directions = (desired,)
        for direction in directions:
            candidate = self.tile_cache.neighbor(current, direction)
            if candidate is None:
                continue
            if (
                require_closer
                and candidate.distance_squared(target) >= current.distance_squared(target)
            ):
                continue
            if self.try_move_step(controller, direction, avoid_visited=avoid_visited):
                self.heading = direction
                self.wall_following = False
                return True

        # Explicit exploration probes do not spill into a different movement
        # policy.  The caller decides whether to try wall following or the
        # remaining direction spectrum next.
        if (
            forward_sector_only
            or remaining_directional_spectrum
        ):
            return False

        self.wall_following = True
        heading = self.heading or desired
        retreat = (
            None
            if self.previous_position is None
            else current.direction_to(self.previous_position)
        )
        right_hand = self.right_hand_directions(heading)
        attempted = set(right_hand)
        for direction in right_hand:
            if self.try_move_step(controller, direction, avoid_visited=avoid_visited):
                self.heading = direction
                return True

        # A wall follower must exhaust every side exit before it walks back to
        # the tile it just left.  Returning first creates a two-tile loop at
        # a wall corner: on the next turn the exact same priority order brings
        # the bot straight back.  Keep the previous tile as a true last resort.
        for direction in DIRECTIONS:
            if direction in attempted or direction == retreat:
                continue
            if self.try_move_step(controller, direction, avoid_visited=avoid_visited):
                self.heading = direction
                return True
        if (
            retreat is not None
            and retreat != Direction.CENTRE
            and self.try_move_step(controller, retreat, avoid_visited=avoid_visited)
        ):
            self.heading = retreat
            return True
        return False

    def forward_sector_directions(self, desired: Direction) -> tuple[Direction, ...]:
        """Return the direct heading and its two adjacent forward diagonals."""
        return (
            desired,
            desired.rotate_left(),
            desired.rotate_right(),
        )

    def remaining_exploration_directions(
            self,
            desired: Direction,
    ) -> tuple[Direction, ...]:
        """Return the five directions deferred until wall following fails.

        For NORTH the order is EAST, WEST, SOUTHEAST, SOUTHWEST, SOUTH.
        """
        left = desired.rotate_left()
        right = desired.rotate_right()
        return (
            right.rotate_right(),
            left.rotate_left(),
            right.rotate_right().rotate_right(),
            left.rotate_left().rotate_left(),
            desired.opposite(),
        )

    def start_exploration_wall_following(
            self,
            controller: Controller,
            target: Position | None,
            side: str,
            avoid_visited: bool,
    ) -> bool:
        """Choose a wall side and make the first unvisited bypass step."""
        if target is None:
            return False
        current = self.get_cached_position()
        desired = current.direction_to(target)
        if desired == Direction.CENTRE:
            return False
        self.stop_exploration_wall_following()
        self.exploration_wall_side = side
        self.exploration_wall_heading = (
            desired.rotate_right().rotate_right()
            if side == "left"
            else desired.rotate_left().rotate_left()
        )
        self.wall_following = True
        if self.continue_exploration_wall_following(controller, avoid_visited):
            return True
        self.stop_exploration_wall_following()
        return False

    def continue_exploration_wall_following(
            self,
            controller: Controller,
            avoid_visited: bool,
    ) -> bool:
        """Continue on the selected side; never switch sides implicitly."""
        side = self.exploration_wall_side
        heading = self.exploration_wall_heading
        if side is None or heading is None:
            return False
        directions = (
            self.left_wall_directions(heading)
            if side == "left"
            else self.right_wall_directions(heading)
        )
        for direction in directions:
            if self.try_move_step(controller, direction, avoid_visited=avoid_visited):
                self.exploration_wall_heading = direction
                self.heading = direction
                return True
        return False

    def stop_exploration_wall_following(self) -> None:
        """Forget the selected exploration-wall side and its heading."""
        self.exploration_wall_heading = None
        self.exploration_wall_side = None
        self.wall_following = False

    def left_wall_directions(self, heading: Direction) -> tuple[Direction, ...]:
        """Return the left-hand wall-following choices from a current heading."""
        return (
            heading.rotate_left(),
            heading,
            heading.rotate_right(),
        )

    def right_wall_directions(self, heading: Direction) -> tuple[Direction, ...]:
        """Return the right-hand wall-following choices from a current heading."""
        return (
            heading.rotate_right(),
            heading,
            heading.rotate_left(),
        )

    def right_hand_directions(self, heading: Direction) -> tuple[Direction, ...]:
        """Return the three clockwise forward wall-following choices."""
        candidates = (
            heading.rotate_right(),
            heading,
            heading.rotate_left(),
        )
        return tuple(dict.fromkeys(candidates))

    def try_move_step(
            self,
            controller: Controller,
            direction: Direction,
            avoid_visited: bool = False,
            build_road: bool = True,
    ) -> bool:
        """Optionally lay a road on an empty tile, then take one legal step."""
        if direction == Direction.CENTRE:
            return False
        current = self.get_cached_position()
        target = self.tile_cache.neighbor(current, direction)
        if target is None:
            return False
        if avoid_visited and target in self.visited_tiles:
            return False
        env = self.known_env.get(target)
        if env is None or env in {Environment.WALL, Environment.ORE_TITANIUM, Environment.ORE_AXIONITE}:
            return False
        if self.tile_cache.builder_id_at(target) is not None:
            return False
        building = self.known_buildings.get(target)
        if building is None:
            if not build_road:
                return False
            if not controller.can_build_road(target):
                return False
            road_id = controller.build_road(target)
            self.tile_cache.remember_building(target, road_id, EntityType.ROAD, self.team)
        elif building[0] == EntityType.MARKER and building[1] == self.team:
            # A persistent Core order can block the sole exit from a narrow
            # spawn corridor.  Friendly markers are free to destroy from the
            # adjacent BuilderBot tile; open it now and step through next turn.
            if not controller.can_destroy(target):
                return False
            controller.destroy(target)
            self.tile_cache.forget_building(target)
            return True
        elif building[0] not in PASSABLE_BUILDINGS:
            return False
        elif building[0] == EntityType.CORE and building[1] != self.team:
            return False
        if not self.is_cached_tile_passable(target) or not controller.can_move(direction):
            return False
        controller.move(direction)
        return True

    def start_launcher_crossing(
            self,
            controller: Controller,
            current: Position,
            direction: Direction,
            destination: Position,
            exact_landing: bool = False,
    ) -> bool:
        """Build a launcher for a known landing tile beyond a blocking wall.

        Supply-route crossings use an exact A* bridge endpoint.  Letting the
        generic exploration launcher choose a farther visible cell would put
        the Intruder outside its saved path and make it rejoin from the wrong
        side of the barrier.
        """
        landing = (
            destination
            if exact_landing and self.is_cached_tile_passable(destination)
            else self.find_launcher_landing(current, direction, destination)
        )
        if exact_landing and landing != destination:
            return False
        if landing is None:
            return False
        for launcher_pos in self.launcher_build_sites(current, landing):
            marker_pos = self.find_launch_marker_site(current, launcher_pos)
            if marker_pos is None:
                continue
            if not controller.can_place_marker(marker_pos):
                continue
            if not controller.can_build_launcher(launcher_pos):
                continue
            marker_value = encode_marker(MARKER_KIND_INTRUDER_LAUNCH, landing)
            marker_id = controller.place_marker(marker_pos, marker_value)
            self.tile_cache.remember_building(
                marker_pos,
                marker_id,
                EntityType.MARKER,
                self.team,
                marker_value=marker_value,
            )
            launcher_id = controller.build_launcher(launcher_pos)
            self.tile_cache.remember_building(
                launcher_pos,
                launcher_id,
                EntityType.LAUNCHER,
                self.team,
            )
            self.waiting_launcher_origin = current
            self.waiting_launcher_round = controller.get_current_round()
            return True
        return False

    def find_launcher_landing(
            self,
            current: Position,
            direction: Direction,
            destination: Position,
    ) -> Position | None:
        """Choose a visible passable landing tile strictly beyond the blocking wall.

        A positive dot product is not enough for a diagonal heading: it would
        accept a tile south-east of a south-west wall merely because it also
        has a southern component.  Every non-zero axis of the heading must
        advance by at least two tiles, putting the landing on the far side of
        the adjacent wall rather than beside it or in another direction.
        """
        dx, dy = direction.delta()
        candidates: list[tuple[int, int, Position]] = []
        for pos in self.tile_cache.visible_tiles:
            if pos in self.visited_tiles:
                continue
            if not self.is_cached_tile_passable(pos):
                continue
            horizontal_progress = (pos.x - current.x) * dx
            vertical_progress = (pos.y - current.y) * dy
            axis_progress = []
            if dx != 0:
                axis_progress.append(horizontal_progress)
            if dy != 0:
                axis_progress.append(vertical_progress)
            if not axis_progress or min(axis_progress) <= 1:
                continue
            forward_progress = min(axis_progress)
            candidates.append((forward_progress, -pos.distance_squared(destination), pos))
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (item[0], item[1], item[2].x, item[2].y),
            reverse=True,
        )
        # Launcher range is measured from its eventual adjacent build site;
        # keep a conservative one-tile allowance while choosing the landing.
        for _, _, landing in candidates:
            if current.distance_squared(landing) <= GameConstants.LAUNCHER_VISION_RADIUS_SQ + 10:
                return landing
        return None

    def launcher_build_sites(self, current: Position, landing: Position) -> list[Position]:
        """Return empty action-radius tiles where a launcher could reach ``landing``."""
        sites = []
        for direction in DIRECTIONS:
            pos = self.tile_cache.neighbor(current, direction)
            if pos is None:
                continue
            env = self.known_env.get(pos)
            if (
                env is None
                or env in {Environment.WALL, Environment.ORE_TITANIUM, Environment.ORE_AXIONITE}
                or self.known_buildings.get(pos) is not None
                or self.tile_cache.builder_id_at(pos) is not None
                or pos.distance_squared(landing) > GameConstants.LAUNCHER_VISION_RADIUS_SQ
            ):
                continue
            sites.append(pos)
        sites.sort(key=lambda pos: pos.distance_squared(landing))
        return sites

    def find_launch_marker_site(self, current: Position, launcher_pos: Position) -> Position | None:
        """Find an empty nearby cell visible to both the intruder and its launcher."""
        for direction in DIRECTIONS:
            pos = self.tile_cache.neighbor(current, direction)
            if pos is None:
                continue
            if pos == launcher_pos:
                continue
            env = self.known_env.get(pos)
            if (
                env is not None
                and env not in {Environment.WALL, Environment.ORE_TITANIUM, Environment.ORE_AXIONITE}
                and self.known_buildings.get(pos) is None
                and self.tile_cache.builder_id_at(pos) is None
                and launcher_pos.distance_squared(pos) <= GameConstants.LAUNCHER_VISION_RADIUS_SQ
            ):
                return pos
        return None

    def build_forward_gunner(self, controller: Controller, current: Position) -> None:
        """Reach a maximum-range core shot, clear it if needed, and build the gunner."""
        if self.gunner_site is None:
            site_data = self.choose_gunner_site(controller)
            if site_data is None:
                self.advance_towards_unvisited_target(controller, self.destination)
                return
            self.gunner_site, self.gunner_direction = site_data

        site = self.gunner_site
        building = self.known_buildings.get(site)
        if building is not None:
            if building[0] not in _CLEARABLE_WALKABLE_BUILDINGS:
                self.rejected_gunner_sites.update((site,))
                self.gunner_site = None
                self.gunner_direction = None
                return
            if current != site:
                self.advance_towards_revisitable_target(controller, site)
                return
            self.clear_walkable_tile(controller, site, allow_enemy_road=True)
            return

        if current == site:
            self.vacate_gunner_site(controller, site)
            return
        if current.distance_squared(site) > GameConstants.ACTION_RADIUS_SQ:
            approach = self.construction_approach(site)
            self.advance_towards_revisitable_target(
                controller,
                approach or site,
            )
            return
        if self.gunner_direction is None:
            self.rejected_gunner_sites.update((site,))
            self.gunner_site = None
            self.gunner_direction = None
            return
        if not controller.can_build_gunner(site, self.gunner_direction):
            # A road was just removed or the bot has just moved here.  In
            # both cases its action cooldown may still be one for this turn;
            # retain the geometrically validated site and retry next round.
            return
        self.gunner_id = controller.build_gunner(site, self.gunner_direction)
        self.tile_cache.remember_building(
            site,
            self.gunner_id,
            EntityType.GUNNER,
            self.team,
            direction=self.gunner_direction,
        )
        self.reset_supply_plan()

    def vacate_gunner_site(self, controller: Controller, site: Position) -> bool:
        """Move off an empty Gunner site so the stationary unit can be built there.

        A BuilderBot may clear a road while standing on it, but cannot create
        a non-passable Gunner on the tile it occupies.  It may build a short
        escape road and move in the same turn; the following turn builds the
        Gunner from the adjacent action-radius tile.
        """
        current = self.get_cached_position()
        for direction in DIRECTIONS:
            target = self.tile_cache.neighbor(current, direction)
            if target is None:
                continue
            if target == site or not self.is_roadable_position(target):
                continue
            if self.try_move_step(controller, direction):
                return True
        return False

    def choose_gunner_site(self, controller: Controller) -> tuple[Position, Direction] | None:
        """Choose the farthest clearable site whose ray reaches a Core edge tile.

        A Core is a 3x3 building.  A gunner firing at its centre would first
        hit the nearer edge of that same Core, so the Controller correctly
        rejects the centre as an obstructed target.  Aim at each perimeter
        tile instead: that is the first Core tile on a valid attack ray.
        """
        if not self.destination_is_confirmed_core or self.destination is None:
            self.clear_gunner_site_candidates()
            return None
        if self.gunner_site_candidates is None:
            self.continue_gunner_site_candidate_preparation()
            # Candidate enumeration is deliberately its own turn.  The
            # persistent preparation state keeps it bounded even if this
            # turn is interrupted by the engine.
            return None

        validations = 0
        while (
            self.gunner_candidate_cursor < len(self.gunner_site_candidates)
            and validations < _GUNNER_SITE_VALIDATIONS_PER_TURN
        ):
            site, facing, target, clearable_adjacent_site = (
                self.gunner_site_candidates[self.gunner_candidate_cursor]
            )
            self.gunner_candidate_cursor += 1
            validations += 1
            if (
                clearable_adjacent_site
                or controller.can_fire_from(
                    site,
                    facing,
                    EntityType.GUNNER,
                    target,
                )
            ):
                self.clear_gunner_site_candidates()
                return site, facing
        if self.gunner_candidate_cursor >= len(self.gunner_site_candidates):
            # All retained candidates have been checked against the current
            # engine state.  Allow a fresh local enumeration next turn after
            # additional terrain may have entered the cache.
            self.clear_gunner_site_candidates()
        return None

    def continue_gunner_site_candidate_preparation(self) -> None:
        """Incrementally enumerate and rank candidate sites without engine calls."""
        enemy_core = self.destination
        if not self.destination_is_confirmed_core or enemy_core is None:
            self.clear_gunner_site_candidates()
            return
        if (
            self.gunner_candidate_targets is None
            or self.gunner_candidate_destination != enemy_core
        ):
            self.gunner_candidate_targets = self.enemy_core_edge_tiles()
            self.gunner_candidate_destination = enemy_core
            self.gunner_candidate_target_index = 0
            self.gunner_candidate_facing_index = 0
            self.gunner_candidate_distance = 0
            self.gunner_candidate_seen_sites = set()
            self.gunner_candidate_builds = []

        preparations = 0
        while (
            preparations < _GUNNER_CANDIDATE_PREPARATIONS_PER_TURN
            and self.gunner_candidate_targets is not None
            and self.gunner_candidate_target_index < len(self.gunner_candidate_targets)
        ):
            target = self.gunner_candidate_targets[self.gunner_candidate_target_index]
            facing = DIRECTIONS[self.gunner_candidate_facing_index]
            max_steps = 3 if facing in ORTHOGONAL_DIRECTIONS else 2
            if self.gunner_candidate_distance == 0:
                self.gunner_candidate_distance = max_steps
            distance = self.gunner_candidate_distance
            self.advance_gunner_candidate_cursor()
            preparations += 1

            site = target
            for _ in range(distance):
                site = self.tile_cache.neighbor(site, facing.opposite())
                if site is None:
                    break
            if site is None:
                continue
            site_key = (site, facing)
            if site_key in self.gunner_candidate_seen_sites:
                continue
            self.gunner_candidate_seen_sites.update((site_key,))
            if (
                site in self.rejected_gunner_sites
                or self.known_env.get(site) in {
                    None,
                    Environment.WALL,
                    Environment.ORE_TITANIUM,
                    Environment.ORE_AXIONITE,
                }
            ):
                continue
            building = self.known_buildings.get(site)
            if (
                building is not None
                and building[0] not in _CLEARABLE_WALKABLE_BUILDINGS
            ):
                continue
            # ``can_fire_from`` rejects a prospective Gunner sitting on an
            # enemy road.  Adjacent to a Core edge there is no intervening
            # tile, so clearing that road is sufficient.
            clearable_adjacent_site = (
                building is not None
                and building[0] in _CLEARABLE_WALKABLE_BUILDINGS
                and distance == 1
            )
            self.gunner_candidate_builds.append((
                site.distance_squared(target),
                site.distance_squared(enemy_core),
                int(building is None),
                site,
                facing,
                target,
                clearable_adjacent_site,
            ))

        if (
            self.gunner_candidate_targets is not None
            and self.gunner_candidate_target_index
            < len(self.gunner_candidate_targets)
        ):
            return
        self.gunner_candidate_builds.sort(
            key=lambda item: (item[0], item[1], item[2], item[3].x, item[3].y),
            reverse=True,
        )
        self.gunner_site_candidates = [
            (site, facing, target, clearable_adjacent_site)
            for _, _, _, site, facing, target, clearable_adjacent_site
            in self.gunner_candidate_builds
        ]
        self.gunner_candidate_cursor = 0
        self.gunner_candidate_targets = None
        self.gunner_candidate_seen_sites = set()
        self.gunner_candidate_builds = []

    def advance_gunner_candidate_cursor(self) -> None:
        """Advance the persistent target/facing/distance candidate cursor."""
        self.gunner_candidate_distance -= 1
        if self.gunner_candidate_distance > 0:
            return
        self.gunner_candidate_facing_index += 1
        if self.gunner_candidate_facing_index < len(DIRECTIONS):
            return
        self.gunner_candidate_facing_index = 0
        self.gunner_candidate_target_index += 1

    def clear_gunner_site_candidates(self) -> None:
        """Forget a completed or obsolete prepared Gunner-site ranking."""
        self.gunner_site_candidates = None
        self.gunner_candidate_cursor = 0
        self.gunner_candidate_targets = None
        self.gunner_candidate_destination = None
        self.gunner_candidate_target_index = 0
        self.gunner_candidate_facing_index = 0
        self.gunner_candidate_distance = 0
        self.gunner_candidate_seen_sites = set()
        self.gunner_candidate_builds = []

    def enemy_core_edge_tiles(self) -> tuple[Position, ...]:
        """Return all in-bounds perimeter cells of the known 3x3 enemy Core."""
        enemy_core = self.destination
        if not self.destination_is_confirmed_core or enemy_core is None:
            return ()
        return tuple(
            pos
            for dx in range(-1, 2)
            for dy in range(-1, 2)
            if (
                (dx != 0 or dy != 0)
                and (pos := self.tile_cache.offset(enemy_core, dx, dy)) is not None
            )
        )

    def construction_approach(self, site: Position) -> Position | None:
        """Pick a roadable tile from which the builder can act on ``site``."""
        current = self.get_cached_position()
        candidates = [
            pos
            for direction in DIRECTIONS
            if (pos := self.tile_cache.neighbor(site, direction)) is not None
            and self.is_roadable_position(pos)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda pos: current.distance_squared(pos))

    def is_roadable_position(self, pos: Position) -> bool:
        """Return whether an empty or walkable cached tile can support an intruder step."""
        env = self.known_env.get(pos)
        if env is None or env in {Environment.WALL, Environment.ORE_TITANIUM, Environment.ORE_AXIONITE}:
            return False
        building = self.known_buildings.get(pos)
        return building is None or (
            building[0] in PASSABLE_BUILDINGS
            and (building[0] != EntityType.CORE or building[1] == self.team)
        )

    def unvisited_path_traversable(self, pos: Position) -> bool:
        """Restrict exploratory A* to known roadable cells not visited before."""
        return pos not in self.visited_tiles and self.is_roadable_position(pos)

    def clear_walkable_tile(
            self,
            controller: Controller,
            pos: Position,
            allow_enemy_road: bool = False,
    ) -> bool:
        """Remove a replaceable tile, allowing enemy roads only for supply routes.

        A BuilderBot can attack an enemy building only while standing on it.
        Consequently, an enemy road is never cleared during ordinary
        movement, but a supply branch may clear one after deliberately
        stepping onto that planned route cell.
        """
        building = self.known_buildings.get(pos)
        if building is None:
            return True
        building_type, building_team = building
        if building_type not in _CLEARABLE_WALKABLE_BUILDINGS:
            return False
        if building_team == self.team:
            if not controller.can_destroy(pos):
                return False
            controller.destroy(pos)
            self.tile_cache.forget_building(pos)
            return True
        if building_type == EntityType.ROAD and not allow_enemy_road:
            return False
        if pos != self.get_cached_position() or not controller.can_fire(pos):
            return False
        building_id = self.tile_cache.building_id_at(pos)
        if building_id is None:
            return False
        remaining_hp = controller.get_hp(building_id)
        controller.fire(pos)
        if remaining_hp <= GameConstants.BUILDER_BOT_ATTACK_DAMAGE:
            self.tile_cache.forget_building(pos)
        return False

    def clear_cheap_enemy_building(self, controller: Controller, current: Position) -> bool:
        """Fire at a cheap enemy logistics tile underfoot and spend this turn on it.

        BuilderBots cannot fire at an arbitrary adjacent enemy structure: the
        rules permit damage only to the building on their own tile.  Therefore
        this is intentionally limited to enemy walkable logistics buildings,
        never to a Core, Harvester, or armoured conveyor.  A target is cheap
        only when its remaining HP fits in at most two BuilderBot shots.
        """
        building = self.known_buildings.get(current)
        if building is None:
            return False
        building_type, building_team = building
        if (
            building_team == self.team
            or building_type not in _CHEAP_ENEMY_WALKABLE_BUILDINGS
            or not controller.can_fire(current)
        ):
            return False

        building_id = self.tile_cache.building_id_at(current)
        if building_id is None:
            return False
        remaining_hp = controller.get_hp(building_id)
        if remaining_hp > _CHEAP_ENEMY_BUILDING_MAX_HP:
            return False

        controller.fire(current)
        if remaining_hp <= GameConstants.BUILDER_BOT_ATTACK_DAMAGE:
            self.tile_cache.forget_building(current)
        return True

    def supply_gunner(self, controller: Controller, current: Position) -> None:
        """Connect the gunner to the nearest usable titanium harvester or ore deposit."""
        if self.supply_ore is None:
            if not self.select_supply_plan():
                if self.supply_plan_candidates is not None:
                    # An expensive route search is deliberately spread across
                    # turns.  Do not wander away while the next candidate is
                    # still awaiting its bounded A* attempt.
                    return
                self.search_for_titanium(controller)
                return

        if not self.ensure_supply_harvester(controller, current):
            return
        if not self.supply_path:
            plan = self.plan_supply_route(self.supply_ore)
            if plan is None:
                self.explore_supply_route(controller, current)
                return
            self.store_supply_plan(self.supply_ore, plan)
        if self.supply_index >= len(self.supply_path):
            self.mode = "complete"
            return

        tile = self.supply_path[self.supply_index]
        if self.supply_tile_complete(tile):
            self.supply_index += 1
            return
        if (
            tile in self.supply_bridge_targets
            and current != tile
            and current.distance_squared(tile) <= GameConstants.ACTION_RADIUS_SQ
        ):
            if self.build_supply_tile(controller, tile):
                self.supply_index += 1
            return
        if current != tile:
            self.move_towards(controller, tile)
            return
        if tile in self.supply_bridge_targets:
            self.step_off_supply_bridge(controller)
            return
        if self.build_supply_tile(controller, tile):
            self.supply_index += 1

    def select_supply_plan(self) -> bool:
        """Select the closest usable Ti deposit without unbounded route work."""
        if self.gunner_site is None or self.gunner_direction is None:
            return False
        if self.supply_plan_candidates is None:
            self.supply_plan_candidates = sorted(
                (
                    pos
                    for pos, env in self.known_env.items()
                    if env == Environment.ORE_TITANIUM
                    and pos not in self.unavailable_titanium
                    and (
                        self.known_buildings.get(pos) is None
                        or self.known_buildings[pos][0] == EntityType.HARVESTER
                    )
                ),
                key=lambda pos: (pos.distance_squared(self.gunner_site), pos.y, pos.x),
            )
            self.supply_plan_cursor = 0
            self.deferred_supply_candidates = []

        attempts = 0
        while (
            self.supply_plan_cursor < len(self.supply_plan_candidates)
            and attempts < _SUPPLY_ROUTE_PLANS_PER_TURN
        ):
            ore = self.supply_plan_candidates[self.supply_plan_cursor]
            self.supply_plan_cursor += 1
            attempts += 1
            plan = self.plan_supply_route(ore)
            if plan is not None:
                self.store_supply_plan(ore, plan)
                return True
            self.deferred_supply_candidates.append(ore)

        if self.supply_plan_cursor < len(self.supply_plan_candidates):
            return False
        if self.deferred_supply_candidates:
            # Terrain which has not entered the cache cannot yet have a valid
            # A* route.  Keep the nearest deposit as a temporary exploration
            # target after the bounded attempts have all failed.
            ore = self.deferred_supply_candidates[0]
            self.supply_ore = ore
            self.supply_path = []
            self.supply_travel_path = []
            self.supply_travel_bridges = {}
            self.supply_travel_index = 0
            self.pending_supply_bridge = None
            self.supply_directions = {}
            self.supply_bridge_targets = {}
            self.supply_index = 0
            self.supply_plan_candidates = None
            self.supply_plan_cursor = 0
            self.deferred_supply_candidates = []
            return True
        self.supply_plan_candidates = None
        return False

    def store_supply_plan(
            self,
            ore: Position,
            plan: tuple[
                list[Position],
                dict[Position, Direction],
                dict[Position, Position],
                int,
            ],
    ) -> None:
        """Persist a Gunner-origin A* route and reset its conveyor build cursor."""
        path, directions, bridge_targets, _ = plan
        self.supply_ore = ore
        self.supply_path = path
        self.supply_travel_path = list(reversed(path))
        self.supply_travel_bridges = {
            target: source for source, target in bridge_targets.items()
        }
        self.supply_travel_index = 0
        self.pending_supply_bridge = None
        self.supply_directions = directions
        self.supply_bridge_targets = bridge_targets
        self.supply_index = 0
        self.supply_plan_candidates = None
        self.supply_plan_cursor = 0
        self.deferred_supply_candidates = []

    def explore_supply_route(self, controller: Controller, current: Position) -> None:
        """Reveal terrain between a harvested deposit and Gunner until A* can join them."""
        if self.gunner_site is None:
            return
        # The harvester now covers the ore tile, so trace back from its edge
        # toward the Gunner.  Each step refreshes the local cache; the next
        # turn retries A* and begins construction as soon as a full known
        # conveyor route exists.
        self.move_towards(controller, self.gunner_site)

    def plan_supply_route(
            self,
            ore: Position,
    ) -> tuple[
        list[Position],
        dict[Position, Direction],
        dict[Position, Position],
        int,
    ] | None:
        """Build a bridge-aware A* route from a Gunner input to an ore neighbour."""
        if self.gunner_site is None or self.gunner_direction is None:
            return None
        ore_entries = {
            entry
            for direction in ORTHOGONAL_DIRECTIONS
            if (entry := self.tile_cache.neighbor(ore, direction)) is not None
            and self.is_supply_tile(entry)
        }
        gunner_entries = {
            entry
            for direction in ORTHOGONAL_DIRECTIONS
            if direction != self.gunner_direction
            and (entry := self.tile_cache.neighbor(self.gunner_site, direction)) is not None
            and self.is_supply_tile(entry)
        }
        if not ore_entries or not gunner_entries:
            return None

        travel_plan = a_star_from_any_with_bridges(
            None,
            gunner_entries,
            ore_entries,
            lambda _controller, pos: self.is_supply_tile(pos),
            normal_step_cost=_SUPPLY_CONVEYOR_STEP_COST,
            bridge_step_cost=_SUPPLY_BRIDGE_STEP_COST,
            neighbor_fn=self.tile_cache.neighbor,
            max_expansions=_SUPPLY_BRIDGE_SEARCH_EXPANSIONS,
            bridge_landing_fn=lambda _controller, pos: self.is_cached_tile_passable(pos),
        )
        if travel_plan is None:
            return None
        travel_path, travel_bridge_targets, cost = travel_plan
        path = list(reversed(travel_path))
        bridge_targets = {
            target: source for source, target in travel_bridge_targets.items()
        }
        directions = {
            tile: tile.direction_to(path[index + 1] if index + 1 < len(path) else self.gunner_site)
            for index, tile in enumerate(path)
            if tile not in bridge_targets
        }
        return path, directions, bridge_targets, cost

    def is_supply_tile(self, pos: Position) -> bool:
        """Allow known ground that can become a conveyor, including enemy logistics.

        An enemy road or conveyor is a valid route cell: the Intruder can
        stand on it, fire at it from that cell, then replace it with a
        friendly conveyor.  Other foreign infrastructure remains excluded
        because it cannot be safely converted while preserving the branch's
        direction.
        """
        env = self.known_env.get(pos)
        if env is None or env in {Environment.WALL, Environment.ORE_TITANIUM, Environment.ORE_AXIONITE}:
            return False
        building = self.known_buildings.get(pos)
        return building is None or building[0] in _CLEARABLE_WALKABLE_BUILDINGS

    def ensure_supply_harvester(self, controller: Controller, current: Position) -> bool:
        """Reuse an occupied Ti harvester or build our own before laying its branch."""
        if self.supply_ore is None:
            return False
        building = self.known_buildings.get(self.supply_ore)
        if building is not None and building[0] == EntityType.HARVESTER:
            # Resource stacks are allowed to cross team boundaries, so an
            # enemy harvester can directly feed our adjacent conveyor.
            return True
        if building is not None:
            self.unavailable_titanium.update((self.supply_ore,))
            self.reset_supply_plan()
            return False
        if current.distance_squared(self.supply_ore) > GameConstants.ACTION_RADIUS_SQ:
            if self.supply_travel_path:
                self.follow_supply_travel_path(controller, current)
            else:
                self.explore_supply_route(controller, current)
            return False
        if not controller.can_build_harvester(self.supply_ore):
            return False
        harvester_id = controller.build_harvester(self.supply_ore)
        self.tile_cache.remember_building(
            self.supply_ore,
            harvester_id,
            EntityType.HARVESTER,
            self.team,
        )
        return False

    def follow_supply_travel_path(self, controller: Controller, current: Position) -> bool:
        """Follow the saved Gunner-to-ore A* route instead of greedily chasing ore."""
        if not self.supply_travel_path:
            return False
        if self.finish_pending_supply_bridge(controller, current):
            return True
        try:
            index = self.supply_travel_path.index(current)
        except ValueError:
            # A launcher should use the exact endpoint below.  Keep this
            # fallback defensive nevertheless: an unexpected displacement
            # must rejoin only the *remaining* suffix, never walk all the way
            # back to the Gunner-side beginning of the route.
            remaining = set(self.supply_travel_path[self.supply_travel_index:])
            if not remaining:
                return False
            approach = a_star_to_any(
                None,
                current,
                remaining,
                lambda _controller, pos: self.is_roadable_position(pos),
                self.tile_cache.neighbor,
                movement_directions=DIRECTIONS,
                max_expansions=_SUPPLY_SEARCH_EXPANSIONS,
            )
            if not approach:
                return False
            return self.move_to_supply_route_tile(
                controller,
                current,
                approach[0],
            )
        self.supply_travel_index = max(self.supply_travel_index, index)

        # Build each ordinary conveyor while travelling away from the Gunner.
        # Its direction points to the preceding travel node, i.e. back toward
        # the Gunner.  Thus infrastructure on the Gunner side of a bridge is
        # already complete before a Launcher crosses that wall.
        if current in self.supply_directions and not self.supply_tile_complete(current):
            self.build_supply_tile(controller, current)
            return True
        if index + 1 >= len(self.supply_travel_path):
            return False
        next_pos = self.supply_travel_path[index + 1]
        if current in self.supply_bridge_targets and not self.supply_tile_complete(current):
            # The Launcher has just deposited us on the far bridge endpoint.
            # A foreign road on that endpoint must be fired at while we are
            # still standing on it: BuilderBots cannot attack an adjacent
            # road.  Do not step off until it has been completely removed.
            self.pending_supply_bridge = current
            if self.known_buildings.get(current) is not None:
                self.build_supply_tile(controller, current)
                return True
            # A Builder cannot erect a Bridge underneath itself.  Once the
            # landing cell is clear, step exactly one A* tile toward the ore;
            # ``finish_pending_supply_bridge`` will create the Bridge behind
            # us on the following turn.
            return self.move_to_supply_route_tile(controller, current, next_pos)
        if self.supply_travel_bridges.get(current) == next_pos:
            direction = current.direction_to(next_pos)
            launched = self.start_launcher_crossing(
                controller,
                current,
                direction,
                next_pos,
                exact_landing=True,
            )
            if launched:
                self.supply_travel_index = index + 1
            return launched
        return self.move_to_supply_route_tile(controller, current, next_pos)

    def finish_pending_supply_bridge(self, controller: Controller, current: Position) -> bool:
        """Build the physical Bridge immediately after crossing its A* jump.

        A Builder cannot build a Bridge on the tile it currently occupies.
        After a Launcher lands on the far endpoint it therefore walks exactly
        one planned step toward the ore, then creates the Bridge behind it.
        This makes the virtual A* jump a real conveyor connection before the
        bot continues toward the Harvester.
        """
        bridge_tile = self.pending_supply_bridge
        if bridge_tile is None:
            return False
        if self.supply_tile_complete(bridge_tile):
            self.pending_supply_bridge = None
            return False
        if current == bridge_tile:
            return False
        if current.distance_squared(bridge_tile) <= GameConstants.ACTION_RADIUS_SQ:
            if self.build_supply_tile(controller, bridge_tile):
                self.pending_supply_bridge = None
            return True
        # This branch is only a safety net for an external displacement.  It
        # returns to the bridge endpoint rather than falling back to the
        # Gunner-side route prefix.
        approach = a_star_to_any(
            None,
            current,
            {bridge_tile},
            lambda _controller, pos: self.is_roadable_position(pos),
            self.tile_cache.neighbor,
            movement_directions=DIRECTIONS,
            max_expansions=_SUPPLY_SEARCH_EXPANSIONS,
        )
        if approach:
            self.move_to_supply_route_tile(controller, current, approach[0])
        return True

    def move_to_supply_route_tile(
            self,
            controller: Controller,
            current: Position,
            target: Position,
    ) -> bool:
        """Build a planned conveyor before entering its tile, never a temporary road.

        The generic movement primitive turns an empty cell into a road so a
        BuilderBot can walk onto it.  That is useful while infiltrating, but
        wrong for a supply route: the very same cell would have to be removed
        and rebuilt as a conveyor on the following two turns.  When the next
        step belongs to the saved route, place its final transport building
        first and then use the normal movement check without road creation.
        """
        direction = current.direction_to(target)
        if direction == Direction.CENTRE or current.distance_squared(target) > 2:
            return False
        is_planned_tile = target in self.supply_path
        building = self.known_buildings.get(target)
        if (
            is_planned_tile
            and building is not None
            and building[0] in _CLEARABLE_WALKABLE_BUILDINGS
            and building[1] != self.team
        ):
            # Builders can only fire at a building underneath themselves.
            # First enter the usable enemy logistics tile; subsequent turns
            # will remove it and replace it with our conveyor.
            return self.try_move_step(controller, direction, build_road=False)
        if is_planned_tile and not self.supply_tile_complete(target):
            if not self.build_supply_tile(controller, target):
                return False
        return self.try_move_step(
            controller,
            direction,
            build_road=not is_planned_tile,
        )

    def supply_tile_complete(self, tile: Position) -> bool:
        """Return whether the planned friendly conveyor or bridge already occupies ``tile``."""
        building = self.known_buildings.get(tile)
        if building is None or building[1] != self.team:
            return False
        expected = (
            EntityType.BRIDGE
            if tile in self.supply_bridge_targets
            else EntityType.CONVEYOR
        )
        return building[0] == expected

    def step_off_supply_bridge(self, controller: Controller) -> bool:
        """Vacate a bridge tile so the Builder can construct that bridge next turn."""
        for direction in DIRECTIONS:
            if controller.can_move(direction):
                controller.move(direction)
                return True
        return False

    def build_supply_tile(self, controller: Controller, tile: Position) -> bool:
        """Replace an obstructing road/conveyor and lay the planned transport tile."""
        building = self.known_buildings.get(tile)
        bridge_target = self.supply_bridge_targets.get(tile)
        wants_bridge = bridge_target is not None
        expected_type = EntityType.BRIDGE if wants_bridge else EntityType.CONVEYOR
        if self.supply_tile_complete(tile):
            return True
        if building is not None:
            if not self.clear_walkable_tile(
                    controller,
                    tile,
                    allow_enemy_road=True,
            ):
                return False
            # Destroying a road/conveyor consumes the Builder action.  Wait
            # for the next turn before attempting to place its replacement.
            return False
        if wants_bridge:
            if bridge_target is None:
                self.reset_supply_plan()
                return False
            # A conveyor may have been placed on the preceding move, leaving
            # the Builder with one round of action cooldown.  That is not a
            # route error: retain the exact bridge order and retry next turn.
            if not controller.can_build_bridge(tile, bridge_target):
                return False
            bridge_id = controller.build_bridge(tile, bridge_target)
            self.tile_cache.remember_building(tile, bridge_id, EntityType.BRIDGE, self.team)
            return True
        direction = self.supply_directions.get(tile)
        if direction is None:
            self.reset_supply_plan()
            return False
        if not controller.can_build_conveyor(tile, direction):
            return False
        conveyor_id = controller.build_conveyor(tile, direction)
        self.tile_cache.remember_building(
            tile,
            conveyor_id,
            EntityType.CONVEYOR,
            self.team,
            direction=direction,
        )
        return True

    def reset_supply_plan(self) -> None:
        """Forget an invalid route so the next turn can choose another Ti source."""
        self.supply_ore = None
        self.supply_path = []
        self.supply_travel_path = []
        self.supply_travel_bridges = {}
        self.supply_travel_index = 0
        self.pending_supply_bridge = None
        self.supply_directions = {}
        self.supply_bridge_targets = {}
        self.supply_index = 0
        self.supply_plan_candidates = None
        self.supply_plan_cursor = 0
        self.deferred_supply_candidates = []

    def search_for_titanium(self, controller: Controller) -> None:
        """Explore away from the enemy core until a cached titanium source becomes visible."""
        if self.gunner_direction is None:
            return
        dx, dy = self.gunner_direction.opposite().delta()
        current = self.get_cached_position()
        target = self.tile_cache.position_at(
            min(self.map_width - 1, max(0, current.x + dx * self.map_width)),
            min(self.map_height - 1, max(0, current.y + dy * self.map_height)),
        )
        if target is not None:
            self.move_towards(controller, target)
