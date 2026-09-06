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
# ``can_fire_from`` is a Controller call.  Keep enough of the 2 ms turn for
# cache refresh and later construction, then resume this persisted queue next
# round rather than changing the target-selection priority.
_GUNNER_SITE_VALIDATIONS_PER_TURN = 7
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
        # explicit A* route back to a wall-bypass branch point, or eventually
        # to the friendly Core after every branch has been exhausted.
        self.visited_tiles: set[Position] = set()
        # Each entry is the point at which a wall bypass began and the side
        # already explored from it.  A dead end on the left returns here to
        # explore the right; a dead end on the right pops the entry.
        self.exploration_branch_stack: list[tuple[Position, str]] = []
        self.return_branch_target: Position | None = None
        self.resuming_exploration_branch = False
        self.returning_to_core = False
        self.return_path: list[Position] = []
        self.waiting_launcher_origin: Position | None = None
        self.waiting_launcher_round: int | None = None

        self.gunner_site: Position | None = None
        self.gunner_direction: Direction | None = None
        self.rejected_gunner_sites: set[Position] = set()
        self.gunner_id: int | None = None
        self.gunner_site_candidates: list[tuple[Position, Direction, Position]] = []
        self.gunner_site_candidate_cursor = 0
        self.gunner_search_origin: Position | None = None
        self.gunner_search_destination: Position | None = None

        self.supply_ore: Position | None = None
        # The only supply route is ordered from the Gunner to the selected
        # ore entry.  The Intruder walks this exact sequence, building each
        # transport tile before it proceeds farther from the Gunner.
        self.supply_path: list[Position] = []
        # A crossing maps its Gunner-side source to its ore-side landing.
        # The physical Bridge is built on the landing and outputs back to the
        # source, so resources still flow from the ore to the Gunner.
        self.supply_bridge_crossings: dict[Position, Position] = {}
        self.pending_supply_bridge: Position | None = None
        self.supply_directions: dict[Position, Direction] = {}
        self.supply_bridge_targets: dict[Position, Position] = {}
        self.supply_index = 0
        # Bridge targets are not exposed by TileCache because only the
        # Intruder's supply planner needs them.  Cache canonical endpoints so
        # an already built bridge can be used as part of a new supply route.
        self.known_bridge_targets: dict[Position, Position] = {}
        self.known_bridge_ids: dict[Position, int] = {}
        self.unavailable_titanium: set[Position] = set()
        self.supply_plan_candidates: list[Position] | None = None
        self.supply_plan_cursor = 0
        self.deferred_supply_candidates: list[Position] = []
        # While the complete Gunner-to-ore route is still outside vision,
        # keep approaching one ore entry.  Re-selecting the nearest entry on
        # every turn can bounce between two adjacent entries around a mine.
        self.supply_exploration_entry: Position | None = None
        self.supply_exploring_back_to_gunner = False
        # Before a Ti deposit is visible, scouting chooses a reachable map
        # frontier.  A bounded A* can prove that a nearby frontier is cut off
        # by known terrain; retain that result so the next turn tries another
        # frontier rather than repeating the same failed search forever.
        self.failed_supply_search_frontiers: set[Position] = set()
        self.supply_search_target: Position | None = None
        # Once a supply route exists, newly scanned deposits are considered
        # incrementally.  This lets the Intruder abandon an early, distant
        # discovery for a genuinely cheaper route without repeating A* for
        # every already-known deposit each turn.
        self.supply_seen_ores: set[Position] = set()
        self.supply_replan_candidates: list[Position] = []
        self.supply_plan_cost: int | None = None

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
        self.refresh_known_bridge_targets(controller)
        if self.clear_cheap_enemy_building(controller, current):
            self.draw_goal_indicator(controller, current)
            return
        if self.wait_for_launcher(controller, current):
            self.draw_goal_indicator(controller, current)
            return
        if self.return_branch_target is not None:
            self.mode = "return_to_exploration_branch"
            self.return_to_exploration_branch(controller, current)
            self.draw_goal_indicator(controller, current)
            return
        if self.resuming_exploration_branch:
            self.mode = "resume_exploration_branch"
            self.resume_exploration_branch(controller, current)
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
        if self.return_branch_target is not None:
            target = self.return_branch_target
        elif self.returning_to_core:
            target = self.core_pos
        elif self.gunner_id is not None:
            # Until a Ti deposit has entered the cache, ``destination`` still
            # names the enemy Core.  Showing it here made a stationary supply
            # scout look as though it was trying to walk back into the Core.
            target = self.supply_ore or self.supply_search_target or self.gunner_site
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

    def has_unvisited_exploration_wall_exit(
            self,
            current: Position,
            target: Position | None,
            side: str,
    ) -> bool:
        """Check only the cells that the selected wall bypass can actually try.

        This intentionally keeps the same treatment of unknown cells, builders,
        and action cooldown as ``has_unvisited_exploration_exit``: those are
        transient and should cause a retry.  Unlike that broad check, however,
        it cannot keep a right-side bypass alive because of an unrelated cell
        on the left side of the branch point.
        """
        if target is None:
            return False
        desired = current.direction_to(target)
        if desired == Direction.CENTRE:
            return False
        heading = (
            desired.rotate_right().rotate_right()
            if side == "left"
            else desired.rotate_left().rotate_left()
        )
        directions = (
            self.left_wall_directions(heading)
            if side == "left"
            else self.right_wall_directions(heading)
        )
        for direction in directions:
            candidate = self.tile_cache.neighbor(current, direction)
            if candidate is None or candidate in self.visited_tiles:
                continue
            environment = self.known_env.get(candidate)
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
            if (
                building_type == EntityType.MARKER
                and building_team == self.team
            ):
                return True
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
            self.begin_return_from_exploration_dead_end(current)
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
        search_state = self.a_star_state("revisitable_target")
        path = a_star_to_any(
            None,
            current,
            {target},
            lambda _controller, pos: self.is_roadable_position(pos),
            self.tile_cache.neighbor,
            movement_directions=DIRECTIONS,
            max_expansions=_SUPPLY_SEARCH_EXPANSIONS,
            state=search_state,
        )
        if search_state.pending:
            return False
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
        search_state = self.a_star_state("return_to_core")
        path = a_star_to_any(
            None,
            current,
            {self.core_pos},
            lambda _controller, pos: self.return_path_traversable(pos),
            self.tile_cache.neighbor,
            movement_directions=DIRECTIONS,
            max_expansions=_RETURN_TO_CORE_A_STAR_MAX_EXPANSIONS,
            state=search_state,
        )
        if search_state.pending:
            return
        if not path:
            return
        self.returning_to_core = True
        self.return_path = path
        self.heading = None

    def begin_return_from_exploration_dead_end(self, current: Position) -> None:
        """Return by A* to the newest left-side branch, then eventually to Core.

        A branch whose right side has already failed is complete, so discard it
        before looking for the next parent.  The first remaining left-side
        branch receives the return route; arriving there retries the right
        side.  This is depth-first exploration without allowing ordinary
        forward movement to revisit historical tiles.
        """
        self.return_branch_target = None
        self.resuming_exploration_branch = False
        self.stop_exploration_wall_following()
        while self.exploration_branch_stack:
            branch_pos, attempted_side = self.exploration_branch_stack[-1]
            if attempted_side == "right":
                self.exploration_branch_stack.pop()
                continue
            if current == branch_pos:
                self.resuming_exploration_branch = True
                return
            search_state = self.a_star_state("return_to_branch")
            path = a_star_to_any(
                None,
                current,
                {branch_pos},
                lambda _controller, pos: self.return_path_traversable(pos),
                self.tile_cache.neighbor,
                movement_directions=DIRECTIONS,
                max_expansions=_RETURN_TO_CORE_A_STAR_MAX_EXPANSIONS,
                state=search_state,
            )
            if search_state.pending:
                return
            if path:
                self.return_branch_target = branch_pos
                self.return_path = path
                self.heading = None
                return
            # The branch cannot be reached through the known terrain, so it
            # cannot be used as a backtracking point.  Try its parent instead.
            self.exploration_branch_stack.pop()
        self.begin_return_to_core(current)

    def return_to_exploration_branch(
            self,
            controller: Controller,
            current: Position,
    ) -> None:
        """Follow the saved A* route to a branch point, allowing revisits."""
        target = self.return_branch_target
        if target is None:
            return
        if current == target:
            self.return_branch_target = None
            self.return_path = []
            self.resuming_exploration_branch = True
            self.resume_exploration_branch(controller, current)
            return
        while self.return_path and current == self.return_path[0]:
            self.return_path.pop(0)
        if not self.return_path:
            self.return_branch_target = None
            self.begin_return_from_exploration_dead_end(current)
            return
        next_pos = self.return_path[0]
        direction = current.direction_to(next_pos)
        if current.distance_squared(next_pos) > 2 or direction == Direction.CENTRE:
            self.return_branch_target = None
            self.begin_return_from_exploration_dead_end(current)
            return
        if self.try_move_step(controller, direction, avoid_visited=False):
            self.return_path.pop(0)
            self.heading = direction
            return
        # A moving builder or fresh obstacle can invalidate a cached segment.
        self.return_branch_target = None
        self.begin_return_from_exploration_dead_end(current)

    def resume_exploration_branch(
            self,
            controller: Controller,
            current: Position,
    ) -> bool:
        """At a left-side branch, explore its untried right-side bypass."""
        self.resuming_exploration_branch = False
        if not self.exploration_branch_stack:
            self.begin_return_to_core(current)
            return False
        branch_pos, attempted_side = self.exploration_branch_stack[-1]
        if current != branch_pos:
            self.begin_return_from_exploration_dead_end(current)
            return False
        if attempted_side != "left":
            self.exploration_branch_stack.pop()
            self.begin_return_from_exploration_dead_end(current)
            return False
        if self.start_exploration_wall_following(
                controller,
                self.destination,
                side="right",
                avoid_visited=True,
                record_branch=False,
        ):
            self.exploration_branch_stack[-1] = (branch_pos, "right")
            return True
        if self.has_unvisited_exploration_wall_exit(
                current,
                self.destination,
                "right",
        ):
            # A cell of this exact right-side bypass may be temporarily blocked
            # by a BuilderBot or an action cooldown.  Retain the branch and
            # retry it rather than discarding the untried side.
            self.resuming_exploration_branch = True
            return False
        self.exploration_branch_stack.pop()
        self.begin_return_from_exploration_dead_end(current)
        return False

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
                self.set_confirmed_enemy_destination(
                    self.tile_cache.entity_position(entity_id),
                )
                return

        inferred_core = self.tile_cache.enemy_core_position(self.team)
        if inferred_core is not None:
            self.set_confirmed_enemy_destination(inferred_core)
            return

        if self.destination is None:
            # The first hypothesis is the 180-degree counterpart.  It is
            # replaced immediately once TileCache confirms the real symmetry.
            self.destination = self.tile_cache.mirrored_position(
                self.core_pos,
                "rotational",
            )

    def set_confirmed_enemy_destination(self, destination: Position) -> None:
        """Update the Core goal without discarding a stable wall-follow state."""
        destination_changed = self.destination != destination
        newly_confirmed = not self.destination_is_confirmed_core
        if destination_changed:
            self.clear_exploration_branch_state()
        self.destination = destination
        self.destination_is_confirmed_core = True
        if destination_changed or newly_confirmed:
            self.wall_following = False
            self.stop_exploration_wall_following()

    def clear_exploration_branch_state(self) -> None:
        """Discard branch state that belongs to an obsolete exploration goal."""
        self.exploration_branch_stack = []
        self.resuming_exploration_branch = False
        if self.return_branch_target is not None:
            self.return_branch_target = None
            self.return_path = []

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
            record_branch: bool = True,
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
            if record_branch:
                self.exploration_branch_stack.append((current, side))
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
        """Choose a viable Core shot, then reach and build the Gunner."""
        if self.gunner_site is None:
            site_data = self.choose_gunner_site(controller, current)
            if site_data is None:
                if self.gunner_site_search_pending():
                    # The engine-check budget was spent before this ordered
                    # search reached a conclusion.  Keep the Intruder still
                    # so the next turn checks the same remaining candidates.
                    return
                # Only after every locally known Core-edge shot has been
                # checked across one or more turns may ordinary exploration
                # look for more terrain and a different firing position.
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

    def choose_gunner_site(
            self,
            controller: Controller,
            current: Position,
    ) -> tuple[Position, Direction] | None:
        """Immediately select the first viable Core shot near this Intruder.

        The first three Core-edge targets are the closest to ``current``.
        For each one, firing directions nearest the direct Intruder-to-target
        line are tried first.  Only if those three targets have no valid shot
        does the search fan out to Core edges farther from that direct line.
        """
        if not self.destination_is_confirmed_core or self.destination is None:
            self.clear_gunner_site_search()
            return None

        if (
            self.gunner_search_origin != current
            or self.gunner_search_destination != self.destination
        ):
            self.start_gunner_site_search(current)

        validations = 0
        while (
            self.gunner_site_candidate_cursor < len(self.gunner_site_candidates)
            and validations < _GUNNER_SITE_VALIDATIONS_PER_TURN
        ):
            site, facing, target = self.gunner_site_candidates[
                self.gunner_site_candidate_cursor
            ]
            self.gunner_site_candidate_cursor += 1
            if not self.is_gunner_site_locally_viable(site):
                continue
            validations += 1
            if controller.can_fire_from(
                site,
                facing,
                EntityType.GUNNER,
                target,
            ):
                self.clear_gunner_site_search()
                return site, facing

        if self.gunner_site_candidate_cursor >= len(self.gunner_site_candidates):
            self.clear_gunner_site_search()
        return None

    def start_gunner_site_search(self, current: Position) -> None:
        """Create one ordered, resumable queue of Core-shot candidates."""
        self.clear_gunner_site_search()
        self.gunner_search_origin = current
        self.gunner_search_destination = self.destination
        closest, fallback = self.gunner_target_groups(current)
        for targets in (closest, fallback):
            for target in targets:
                for facing in self.gunner_facing_directions(current, target):
                    max_distance = 3 if facing in ORTHOGONAL_DIRECTIONS else 2
                    for distance in range(max_distance, 0, -1):
                        site = target
                        for _ in range(distance):
                            site = self.tile_cache.neighbor(site, facing.opposite())
                            if site is None:
                                break
                        if site is not None:
                            self.gunner_site_candidates.append((site, facing, target))

    def gunner_site_search_pending(self) -> bool:
        """Return whether the current Core-shot queue still has untried entries."""
        return self.gunner_site_candidate_cursor < len(self.gunner_site_candidates)

    def clear_gunner_site_search(self) -> None:
        """Discard a completed or obsolete Gunner-site candidate queue."""
        self.gunner_site_candidates = []
        self.gunner_site_candidate_cursor = 0
        self.gunner_search_origin = None
        self.gunner_search_destination = None

    def ordered_gunner_targets(self, current: Position) -> tuple[Position, ...]:
        """Prioritise the three Core edges nearest the Intruder.

        Once those direct targets fail, try the remaining edge cells from the
        greatest perpendicular offset from the Core-to-Intruder segment.  The
        fallback therefore deliberately fans out from the direct approach.
        """
        closest, fallback = self.gunner_target_groups(current)
        return closest + fallback

    def gunner_target_groups(
            self,
            current: Position,
    ) -> tuple[tuple[Position, ...], tuple[Position, ...]]:
        """Return nearest Core edges and their lateral fallback separately."""
        enemy_core = self.destination
        if enemy_core is None:
            return (), ()
        targets = self.enemy_core_edge_tiles()
        closest = sorted(
            targets,
            key=lambda pos: (
                pos.distance_squared(current),
                self.distance_squared_to_core_segment(pos, enemy_core, current),
                pos.y,
                pos.x,
            ),
        )[:3]
        closest_set = set(closest)
        fallback = sorted(
            (pos for pos in targets if pos not in closest_set),
            key=lambda pos: (
                self.distance_squared_to_core_segment(pos, enemy_core, current),
                pos.distance_squared(current),
                pos.y,
                pos.x,
            ),
            reverse=True,
        )
        return tuple(closest), tuple(fallback)

    @staticmethod
    def distance_squared_to_core_segment(
            pos: Position,
            core: Position,
            intruder: Position,
    ) -> int:
        """Return a common-scale squared distance from ``pos`` to Core--Intruder."""
        dx = intruder.x - core.x
        dy = intruder.y - core.y
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            return pos.distance_squared(core)
        offset_x = pos.x - core.x
        offset_y = pos.y - core.y
        projection = offset_x * dx + offset_y * dy
        if projection <= 0:
            return pos.distance_squared(core) * length_squared
        if projection >= length_squared:
            return pos.distance_squared(intruder) * length_squared
        cross_product = offset_x * dy - offset_y * dx
        return cross_product * cross_product

    @staticmethod
    def gunner_facing_directions(
            current: Position,
            target: Position,
    ) -> tuple[Direction, ...]:
        """List Gunner facings from the direct Intruder-to-target ray outward."""
        direct = current.direction_to(target)
        if direct == Direction.CENTRE:
            return tuple(DIRECTIONS)
        directions = [direct]
        left = direct
        right = direct
        for _ in range(3):
            left = left.rotate_left()
            right = right.rotate_right()
            directions.extend((left, right))
        directions.append(direct.opposite())
        return tuple(dict.fromkeys(directions))

    def is_gunner_site_locally_viable(self, site: Position) -> bool:
        """Reject unscanned terrain, resources, solid buildings, and rejected sites."""
        if site in self.rejected_gunner_sites:
            return False
        if self.known_env.get(site) in {
            None,
            Environment.WALL,
            Environment.ORE_TITANIUM,
            Environment.ORE_AXIONITE,
        }:
            return False
        building = self.known_buildings.get(site)
        return (
            building is None
            or building[0] in _CLEARABLE_WALKABLE_BUILDINGS
        )

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
        """Build one Gunner-to-ore supply route, placing transport as we advance."""
        if self.supply_ore is None:
            if not self.select_supply_plan():
                if self.supply_plan_candidates is not None:
                    # An expensive route search is deliberately spread across
                    # turns.  Do not wander away while the next candidate is
                    # still awaiting its bounded A* attempt.
                    return
                self.search_for_titanium(controller)
                return
        elif not self.supply_path:
            # A committed route is a construction contract.  Do not replace
            # it merely because another deposit becomes visible: doing so
            # leaves its already placed Gunner-side conveyors as an isolated
            # fragment and starts the next route somewhere else.
            self.reconsider_supply_plan()
            if self.a_star_state("supply_route").pending:
                return

        if not self.supply_path:
            plan = self.plan_supply_route(self.supply_ore)
            if plan is None:
                if self.a_star_state("supply_route").pending:
                    return
                self.explore_supply_route(controller, current)
                return
            self.store_supply_plan(self.supply_ore, plan)
        if not self.follow_supply_path(controller, current):
            return
        if not self.ensure_supply_harvester(controller, current):
            return
        self.mode = "complete"

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
                ),
                key=lambda pos: (pos.distance_squared(self.gunner_site), pos.y, pos.x),
            )
            self.supply_seen_ores.update(self.supply_plan_candidates)
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
            if self.a_star_state("supply_route").pending:
                # Keep this candidate at the head of the queue so the next
                # turn continues its retained A* frontier.
                self.supply_plan_cursor -= 1
                return False
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
            self.supply_bridge_crossings = {}
            self.pending_supply_bridge = None
            self.supply_directions = {}
            self.supply_bridge_targets = {}
            self.supply_index = 0
            self.supply_exploration_entry = None
            self.supply_exploring_back_to_gunner = False
            self.supply_plan_candidates = None
            self.supply_plan_cursor = 0
            self.deferred_supply_candidates = []
            return True
        self.supply_plan_candidates = None
        return False

    def reconsider_supply_plan(self) -> bool:
        """Replace an active route when a newly seen Ti deposit is cheaper.

        The first visible deposit is often only a provisional choice.  A
        supply branch can encounter a much closer deposit while travelling
        toward it, so compare each *new* usable deposit with the saved A*
        cost.  The candidate queue and one-search limit keep this from
        turning map exploration into an unbounded per-turn route search.
        """
        if self.gunner_site is None or self.supply_ore is None:
            return False

        known_ores = {
            pos
            for pos, env in self.known_env.items()
            if env == Environment.ORE_TITANIUM
            and pos not in self.unavailable_titanium
        }
        new_ores = known_ores - self.supply_seen_ores
        if new_ores:
            self.supply_seen_ores.update(new_ores)
            self.supply_replan_candidates.extend(sorted(
                new_ores,
                key=lambda pos: (pos.distance_squared(self.gunner_site), pos.y, pos.x),
            ))

        if not self.supply_replan_candidates:
            return False

        # A bridge-aware search can be expensive.  One bounded attempt per
        # turn is enough to react immediately to a single newly seen deposit
        # while retaining the same timing guarantee as initial planning.
        ore = self.supply_replan_candidates.pop(0)
        plan = self.plan_supply_route(ore)
        if plan is None:
            if self.a_star_state("supply_route").pending:
                self.supply_replan_candidates.insert(0, ore)
            return False
        if self.supply_plan_cost is not None and plan[3] >= self.supply_plan_cost:
            return False

        self.store_supply_plan(ore, plan)
        return True

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
        """Persist one Gunner-to-ore A* route and reset its build cursor."""
        path, directions, bridge_targets, cost = plan
        self.supply_ore = ore
        self.supply_seen_ores.update((ore,))
        self.supply_plan_cost = cost
        self.supply_path = path
        self.supply_bridge_crossings = {
            source: target for target, source in bridge_targets.items()
        }
        self.pending_supply_bridge = None
        self.supply_directions = directions
        self.supply_bridge_targets = bridge_targets
        self.supply_index = 0
        self.supply_plan_candidates = None
        self.supply_plan_cursor = 0
        self.deferred_supply_candidates = []
        self.supply_exploration_entry = None
        self.supply_exploring_back_to_gunner = False
        self.supply_search_target = None

    def explore_supply_route(self, controller: Controller, current: Position) -> None:
        """Extend known supply terrain toward an ore entry without wall loops.

        A visible deposit is useful evidence even when the complete
        Gunner-to-ore A* route has not entered the cache yet.  The previous
        fallback walked back toward the Gunner and delegated its movement to
        generic wall following.  At a wall this can produce an endless loop
        while the deposit is already beside the Intruder.  Instead, approach
        a known usable neighbour of the deposit, or the closest known supply
        frontier leading to it.  A subsequent turn retries the full planner
        and then lays the real conveyors.
        """
        target = self.supply_exploration_entry
        if target is None or not self.is_supply_tile(target):
            target = (
                self.supply_route_frontier(current)
                if self.supply_exploring_back_to_gunner
                else self.supply_exploration_target(current)
            )
            self.supply_exploration_entry = target
        if target is None:
            return
        if current == target:
            # Reaching an ore entry does not mean that the Gunner-to-ore path
            # is known.  Continue revealing terrain back toward the Gunner;
            # otherwise a failed A* plan selects this same entry every turn
            # and leaves the Intruder stationary beside the deposit.
            self.supply_exploring_back_to_gunner = True
            target = self.supply_route_frontier(current)
            self.supply_exploration_entry = target
            if target is None:
                return

        search_state = self.a_star_state("supply_exploration")
        path = a_star_to_any(
            None,
            current,
            {target},
            lambda _controller, pos: self.is_roadable_position(pos),
            self.tile_cache.neighbor,
            movement_directions=DIRECTIONS,
            max_expansions=_SUPPLY_SEARCH_EXPANSIONS,
            state=search_state,
        )
        if search_state.pending:
            return
        if path:
            self.try_move_step(
                controller,
                current.direction_to(path[0]),
                build_road=True,
            )
            return

        # A partial cache may not yet contain a complete route.  Probe only
        # steps which move closer to the ore-side target; unlike
        # ``move_towards`` this never activates wall following or retreats to
        # the Gunner.
        self.move_towards(
            controller,
            target,
            forward_sector_only=True,
            require_closer=True,
        )

    def supply_route_frontier(self, current: Position) -> Position | None:
        """Pick an unseen-map frontier which tends back toward the Gunner."""
        if self.gunner_site is None:
            return None
        frontiers = [
            pos
            for pos in self.known_env
            if pos != current
            and pos not in self.visited_tiles
            and self.is_roadable_position(pos)
            and any(
                neighbor is not None and neighbor not in self.known_env
                for direction in DIRECTIONS
                if (neighbor := self.tile_cache.neighbor(pos, direction)) is not None
            )
        ]
        if not frontiers:
            return None
        gunner_side = [
            pos
            for pos in frontiers
            if pos.distance_squared(self.gunner_site)
            < current.distance_squared(self.gunner_site)
        ]
        candidates = gunner_side or frontiers
        return min(
            candidates,
            key=lambda pos: (
                current.distance_squared(pos),
                pos.distance_squared(self.gunner_site),
                pos.y,
                pos.x,
            ),
        )

    def supply_exploration_target(self, current: Position) -> Position | None:
        """Choose the nearest known supply entry or frontier for ``supply_ore``."""
        ore = self.supply_ore
        if ore is None:
            return None

        entries = [
            entry
            for direction in ORTHOGONAL_DIRECTIONS
            if (entry := self.tile_cache.neighbor(ore, direction)) is not None
            and self.is_supply_tile(entry)
        ]
        if entries:
            return min(
                entries,
                key=lambda pos: (current.distance_squared(pos), pos.y, pos.x),
            )

        frontiers = [
            pos
            for pos in self.known_env
            if self.is_supply_tile(pos)
            and any(
                neighbor is not None and self.known_env.get(neighbor) is None
                for direction in ORTHOGONAL_DIRECTIONS
                if (neighbor := self.tile_cache.neighbor(pos, direction)) is not None
            )
        ]
        if not frontiers:
            return None
        return min(
            frontiers,
            key=lambda pos: (
                pos.distance_squared(ore),
                current.distance_squared(pos),
                pos.y,
                pos.x,
            ),
        )

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
            self.clear_a_star_state("supply_route")
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
            self.clear_a_star_state("supply_route")
            return None

        search_state = self.a_star_state("supply_route")
        travel_plan = a_star_from_any_with_bridges(
            None,
            gunner_entries,
            ore_entries,
            lambda _controller, pos: self.is_supply_tile(pos),
            normal_step_cost=_SUPPLY_CONVEYOR_STEP_COST,
            bridge_step_cost=_SUPPLY_BRIDGE_STEP_COST,
            neighbor_fn=self.tile_cache.neighbor,
            max_expansions=_SUPPLY_BRIDGE_SEARCH_EXPANSIONS,
            bridge_landing_fn=lambda _controller, pos: self.is_supply_bridge_landing(pos),
            existing_bridge_crossings=self.known_supply_bridge_crossings(),
            state=search_state,
        )
        if travel_plan is None:
            return None
        path, new_bridge_crossings, cost = travel_plan
        bridge_targets = {
            target: source for source, target in new_bridge_crossings.items()
        }
        known_crossings = self.known_supply_bridge_crossings()
        for source, target in zip(path, path[1:]):
            if known_crossings.get(source) == target:
                bridge_targets[target] = source
        directions = {
            tile: tile.direction_to(
                self.gunner_site if index == 0 else path[index - 1]
            )
            for index, tile in enumerate(path)
            if tile not in bridge_targets
        }
        return path, directions, bridge_targets, cost

    def is_supply_tile(self, pos: Position) -> bool:
        """Allow a known route tile, including replaceable enemy logistics.

        A known Bridge remains a route tile only when its cached endpoint can
        be used as a supply crossing.  Roads and conveyors are deliberately
        admitted independent of owner: while standing on a foreign transport
        tile the Intruder can destroy it and replace it with its own conveyor.
        """
        env = self.known_env.get(pos)
        if env is None or env in {Environment.WALL, Environment.ORE_TITANIUM, Environment.ORE_AXIONITE}:
            return False
        building = self.known_buildings.get(pos)
        return (
            building is None
            or building[0] in _CLEARABLE_WALKABLE_BUILDINGS
            or (
                building[0] == EntityType.BRIDGE
                and pos in self.known_bridge_targets
            )
        )

    def is_supply_bridge_landing(self, pos: Position) -> bool:
        """Return whether a new supply Bridge may be installed on ``pos``."""
        env = self.known_env.get(pos)
        if env is None or env in {Environment.WALL, Environment.ORE_TITANIUM, Environment.ORE_AXIONITE}:
            return False
        building = self.known_buildings.get(pos)
        return building is None or building[0] in _CLEARABLE_WALKABLE_BUILDINGS

    def known_supply_bridge_crossings(self) -> dict[Position, Position]:
        """Map each existing Bridge's Gunner-side endpoint to its landing tile."""
        return {
            target: bridge
            for bridge, target in self.known_bridge_targets.items()
            if self.known_buildings.get(bridge) is not None
            and self.known_buildings[bridge][0] == EntityType.BRIDGE
        }

    def refresh_known_bridge_targets(self, controller: Controller) -> None:
        """Cache endpoints for visible Bridges so supply planning can reuse them."""
        for pos in self.tile_cache.visible_tiles:
            building = self.known_buildings.get(pos)
            if building is None or building[0] != EntityType.BRIDGE:
                self.known_bridge_targets.pop(pos, None)
                self.known_bridge_ids.pop(pos, None)
                continue
            building_id = self.tile_cache.building_id_at(pos)
            if building_id is None or self.known_bridge_ids.get(pos) == building_id:
                continue
            self.known_bridge_targets[pos] = self.tile_cache.canonicalize(
                controller.get_bridge_target(building_id)
            )
            self.known_bridge_ids[pos] = building_id

    def ensure_supply_harvester(self, controller: Controller, current: Position) -> bool:
        """Reuse an occupied Ti harvester, or build one from the route endpoint."""
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
        return True

    def follow_supply_path(self, controller: Controller, current: Position) -> bool:
        """Advance exactly along the Gunner-to-ore route, building as we go.

        ``True`` means every route tile is now in place and the Intruder is
        standing on the ore-side endpoint.  Any movement, construction,
        launcher crossing, or blocked rejoin consumes this turn and returns
        ``False`` so the next round resumes from the same route state.
        """
        if not self.supply_path:
            return False
        if self.finish_pending_supply_bridge(controller, current):
            return False
        self.advance_supply_index()
        if self.supply_index >= len(self.supply_path):
            if current == self.supply_path[-1]:
                return True
            self.rejoin_supply_path(controller, current, self.supply_path[-1])
            return False
        try:
            index = self.supply_path.index(current)
        except ValueError:
            self.rejoin_supply_path(
                controller,
                current,
                self.supply_path[self.supply_index],
            )
            return False
        if index > self.supply_index:
            # Scouting can cross a far route cell before its Gunner-side
            # prefix exists.  Do not promote the cursor to that position: go
            # back to the first missing tile and resume construction there.
            self.rejoin_supply_path(
                controller,
                current,
                self.supply_path[self.supply_index],
            )
            return False

        if current in self.supply_bridge_targets and not self.supply_tile_complete(current):
            # A Builder cannot create the ore-side Bridge underneath itself.
            # Clear a replaceable foreign transport tile first, then walk one
            # exact route step away and create the Bridge behind it.
            if self.known_buildings.get(current) is not None:
                self.build_supply_tile(controller, current)
                return False
            self.pending_supply_bridge = current
            if index + 1 < len(self.supply_path):
                self.move_to_supply_route_tile(
                    controller,
                    current,
                    self.supply_path[index + 1],
                )
            else:
                self.step_off_supply_bridge(controller)
            return False

        # Lay the final conveyor before moving farther from the Gunner.  Its
        # direction points to the preceding route node, preserving ore-to-
        # Gunner resource flow while the Intruder travels in the opposite way.
        if current in self.supply_directions and not self.supply_tile_complete(current):
            self.build_supply_tile(controller, current)
            return False
        if index + 1 >= len(self.supply_path):
            return True
        next_pos = self.supply_path[index + 1]
        if self.supply_bridge_crossings.get(current) == next_pos:
            direction = current.direction_to(next_pos)
            self.start_launcher_crossing(
                controller,
                current,
                direction,
                next_pos,
                exact_landing=True,
            )
            return False
        self.move_to_supply_route_tile(controller, current, next_pos)
        return False

    def advance_supply_index(self) -> None:
        """Keep the cursor on the first route tile not yet in final form."""
        while (
            self.supply_index < len(self.supply_path)
            and self.supply_tile_complete(self.supply_path[self.supply_index])
        ):
            self.supply_index += 1

    def rejoin_supply_path(
            self,
            controller: Controller,
            current: Position,
            target: Position,
    ) -> None:
        """Walk back to one required route cell without laying a later conveyor."""
        search_state = self.a_star_state("supply_rejoin")
        approach = a_star_to_any(
            None,
            current,
            {target},
            lambda _controller, pos: self.is_roadable_position(pos),
            self.tile_cache.neighbor,
            movement_directions=DIRECTIONS,
            max_expansions=_SUPPLY_SEARCH_EXPANSIONS,
            state=search_state,
        )
        if search_state.pending:
            return
        if approach:
            self.move_to_supply_route_tile(
                controller,
                current,
                approach[0],
                construct_supply_tile=False,
            )

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
        search_state = self.a_star_state("pending_supply_bridge")
        approach = a_star_to_any(
            None,
            current,
            {bridge_tile},
            lambda _controller, pos: self.is_roadable_position(pos),
            self.tile_cache.neighbor,
            movement_directions=DIRECTIONS,
            max_expansions=_SUPPLY_SEARCH_EXPANSIONS,
            state=search_state,
        )
        if search_state.pending:
            return True
        if approach:
            self.move_to_supply_route_tile(controller, current, approach[0])
        return True

    def move_to_supply_route_tile(
            self,
            controller: Controller,
            current: Position,
            target: Position,
            construct_supply_tile: bool = True,
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
            construct_supply_tile
            and is_planned_tile
            and building is not None
            and building[0] in _CLEARABLE_WALKABLE_BUILDINGS
            and building[1] != self.team
        ):
            # Builders can only fire at a building underneath themselves.
            # First enter the usable enemy logistics tile; subsequent turns
            # will remove it and replace it with our conveyor.
            return self.try_move_step(controller, direction, build_road=False)
        if (
            construct_supply_tile
            and is_planned_tile
            and not self.supply_tile_complete(target)
        ):
            if not self.build_supply_tile(controller, target):
                return False
        return self.try_move_step(
            controller,
            direction,
            build_road=not (construct_supply_tile and is_planned_tile),
        )

    def supply_tile_complete(self, tile: Position) -> bool:
        """Return whether the planned friendly conveyor or bridge already occupies ``tile``."""
        building = self.known_buildings.get(tile)
        if building is None:
            return False
        bridge_target = self.supply_bridge_targets.get(tile)
        if bridge_target is not None:
            # Resources may cross team boundaries, so a pre-existing bridge
            # with the exact ore-to-Gunner endpoint is reusable regardless of
            # owner.  A differently aimed bridge never enters this route.
            return (
                building[0] == EntityType.BRIDGE
                and self.known_bridge_targets.get(tile) == bridge_target
            )
        direction = self.supply_directions.get(tile)
        if direction is None:
            return False
        building_id = self.tile_cache.building_id_at(tile)
        return (
            building[0] == EntityType.CONVEYOR
            and building[1] == self.team
            and building_id is not None
            and self.tile_cache.entity_direction(building_id) == direction
        )

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
            self.known_bridge_targets[tile] = bridge_target
            self.known_bridge_ids[tile] = bridge_id
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
        for name in (
                "supply_route",
                "supply_exploration",
                "supply_rejoin",
                "pending_supply_bridge",
                "titanium_search",
        ):
            self.clear_a_star_state(name)
        self.supply_ore = None
        self.supply_path = []
        self.supply_bridge_crossings = {}
        self.pending_supply_bridge = None
        self.supply_directions = {}
        self.supply_bridge_targets = {}
        self.supply_index = 0
        self.supply_plan_candidates = None
        self.supply_plan_cursor = 0
        self.deferred_supply_candidates = []
        self.supply_exploration_entry = None
        self.supply_exploring_back_to_gunner = False
        self.failed_supply_search_frontiers = set()
        self.supply_search_target = None
        self.supply_seen_ores = set()
        self.supply_replan_candidates = []
        self.supply_plan_cost = None

    def search_for_titanium(self, controller: Controller) -> None:
        """Explore the Gunner's rear-side frontier until titanium becomes visible.

        This is reconnaissance, not supply-line construction.  It uses a
        bounded A* path through known terrain instead of repeatedly walking
        in one absolute direction; a wall at the edge of the map otherwise
        strands the Intruder after it has built a trail of roads.
        """
        if self.gunner_direction is None:
            return
        current = self.get_cached_position()
        target = self.supply_search_frontier(current)
        self.supply_search_target = target
        if target is None:
            return
        search_state = self.a_star_state("titanium_search")
        path = a_star_to_any(
            None,
            current,
            {target},
            lambda _controller, pos: self.is_roadable_position(pos),
            self.tile_cache.neighbor,
            movement_directions=DIRECTIONS,
            max_expansions=_SUPPLY_SEARCH_EXPANSIONS,
            state=search_state,
        )
        if search_state.pending:
            return
        if not path:
            # This frontier may be visible across a wall or behind a solid
            # building.  A* has already spent its bounded attempt proving no
            # known route, so allow the next turn to consider another one.
            self.failed_supply_search_frontiers.update((target,))
            return
        if self.try_move_step(
            controller,
            current.direction_to(path[0]),
            build_road=True,
        ):
            # A successful scout move changes the visible frontier ring, so
            # earlier failed routes may now have a way around their obstacle.
            self.failed_supply_search_frontiers = set()

    def supply_search_frontier(self, current: Position) -> Position | None:
        """Choose the next untried cache frontier on the Gunner's resource side."""
        if self.gunner_site is None or self.gunner_direction is None:
            return None
        dx, dy = self.gunner_direction.opposite().delta()
        frontiers = [
            pos
            for pos in self.known_env
            if pos != current
            and pos not in self.failed_supply_search_frontiers
            and self.is_roadable_position(pos)
            and any(
                neighbor is not None and neighbor not in self.known_env
                for direction in DIRECTIONS
                if (neighbor := self.tile_cache.neighbor(pos, direction)) is not None
            )
        ]
        if not frontiers:
            return None

        def resource_side(pos: Position) -> bool:
            return (
                (pos.x - self.gunner_site.x) * dx
                + (pos.y - self.gunner_site.y) * dy
            ) >= 0

        candidates = [pos for pos in frontiers if resource_side(pos)] or frontiers
        return min(
            candidates,
            key=lambda pos: (
                current.distance_squared(pos),
                -(
                    (pos.x - self.gunner_site.x) * dx
                    + (pos.y - self.gunner_site.y) * dy
                ),
                pos.y,
                pos.x,
            ),
        )
