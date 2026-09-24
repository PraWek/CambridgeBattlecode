from __future__ import annotations
from typing import Callable

from cambc import Controller, Direction, EntityType, Position

from botlib import State
from .. import STATE

from botlib import (
    vision_iter,
    MAP_INFO,
    UNIT_INFO,
    BITBOARD_ALLY,
    ALLY_BUILDINGS,
    get_building_direction,
    get_bridge_target_idx,
    BITBOARD_ENEMY,
    HAZARDS,
    MAX_MAP_SIZE,
    BITBOARD_ENV,
    RESOURCE_FLOW,
    get_flow_rate,
    get_stall_rate,
    idx_to_pos,
    pos_to_idx,
    xy_to_idx,
    not_in_bounds,
    print_env_bitboard,
    BITBOARD_TARGETS,
)


from botlib.constants import (
    POSITION_CACHE,
    DIRECTION_CACHE,
    DIRECTION_DELTAS,
    CARDINAL_DIRECTION_DELTAS,
)

from ..utility.movement import move_to
from ..utility.building import can_afford, build_if_can

from ..data import (
    ORES_COMPLETED,
    BOT_PATHING,
    RESOURCE_PATHING,
    DISTANCE_FIELD,
    CORE_DELTAS,
    TITANIUM_HARVESTERS_PLACED,
)

import random

# ------------- STATES -------------

from .fatigue import task_fatigue
from .fundamentals import break_tile, path_to, BREAK_TILE_COOLDOWN


def check_safety(idx):
    if (
        (BITBOARD_ALLY[idx] & 0b0001_1000_0111_1000) == 0
        # ['gunner', 'sentinel', 'breach', 'launcher', 'harvester', 'foundry']
        and (BITBOARD_ENEMY[idx] & 0b0101_1010_0111_1000) == 0
        # ['gunner', 'sentinel', 'breach', 'launcher', 'armoured_conveyor', 'harvester', 'foundry', 'barrier']
        and (HAZARDS[3][idx] == 0)
        and (HAZARDS[4][idx] == 0)
    ):
        return True
    else:
        return False


# TODO convert into a resumable state
class path_place_harvester:
    # TODO flow check

    class state:
        STATE_ID: int = -1

    target_ore_idx: int = -1
    switch_target: bool = True
    ore_type: int = 0b0100
    exit_state: State = None
    do_foundry: bool = False
    path_task: path_to.state | None = None
    bail: Callable[[Controller, path_place_harvester], bool] | None = None

    @classmethod
    def enter(
        self,
        ct: Controller,
        idx: int,
        ore_type: int,
        exit_state: State,
        switch_target: bool = False,
        do_foundry: bool = False,
        path_task: path_to.state | None = None,
        bail: Callable[[Controller, path_place_harvester], bool] | None = None,
    ):
        print(f"found ore, target: {POSITION_CACHE[idx]}")

        self.target_ore_idx = idx
        self.ore_type = ore_type
        self.exit_state = exit_state
        self.switch_target = switch_target
        self.do_foundry = do_foundry
        self.path_task = path_task
        self.bail = bail

    @classmethod
    def run(self, ct: Controller):
        target_ore_pos = POSITION_CACHE[self.target_ore_idx]

        print(f"placing ore at {target_ore_pos}")

        if self.bail is not None and self.bail(ct, self):
            return

        if self.path_task is not None and not self.path_task.path_to_succeeded:
            print("could not path to harvester")
            STATE.switch_to(ct, self.exit_state)
            return

        if BREAK_TILE_COOLDOWN[self.target_ore_idx] > STATE.tick:
            STATE.switch_to(ct, self.exit_state)
            return

        # TODO SAFETY, HARVESTERS CAN OVERWRITE CONVEYORS, BRIDGES ETC.
        if check_safety(self.target_ore_idx) == False:
            print("UNSAFE, BAILING!")
            STATE.switch_to(ct, self.exit_state)
            return

        print(target_ore_pos.distance_squared(UNIT_INFO.position))
        # # if not on tile, move in
        if target_ore_pos.distance_squared(UNIT_INFO.position) > 0:
            if self.switch_target:
                DISTANCE_FIELD.solve(UNIT_INFO.ally_core_idx)
                if DISTANCE_FIELD.ready():
                    # checks if there are is a nearer ore to target, then updates the index
                    target_dist = DISTANCE_FIELD.dist(self.target_ore_idx)
                    if target_dist is not None:
                        for idx in vision_iter():
                            idx_dist = DISTANCE_FIELD.dist(idx)
                            if idx_dist is None:
                                continue
                            if (
                                idx != self.target_ore_idx
                                and (BITBOARD_ENV[idx] & self.ore_type) != 0
                                and not ORES_COMPLETED[idx]
                                and check_safety(idx)
                                and idx_dist < target_dist
                            ):
                                self.target_ore_idx = idx
                                target_ore_pos = POSITION_CACHE[idx]

            if (
                target_ore_pos.distance_squared(UNIT_INFO.position) > 0
                and target_ore_pos.distance_squared(UNIT_INFO.position) <= 2
            ):
                # if reserved with barrier, deconstruct
                if BITBOARD_ALLY[self.target_ore_idx] & 0b0100_0000_0000_0000 != 0:
                    # ^ Barrier
                    ct.destroy(idx_to_pos(self.target_ore_idx))

            path_task = path_to.state(
                self.target_ore_idx, 0, None, bail=path_place_harvester.bail
            )
            path_task.exit_state = State(
                path_place_harvester,
                self.target_ore_idx,
                self.ore_type,
                self.exit_state,
                self.switch_target,
                self.do_foundry,
                path_task=path_task,
                bail=self.bail,
            )
            STATE.switch_to(ct, path_task)
            return
        # if you are on the tile
        else:
            # if enemy road tile start to destroy, enemy road tile, then return
            if (BITBOARD_ENEMY[self.target_ore_idx] & 0b0010_0101_1000_0000) != 0:
                # ['conveyor', 'splitter', 'bridge', 'road']
                STATE.switch_to(
                    ct,
                    break_tile.state(
                        self.target_ore_idx,
                        State(
                            path_place_harvester,
                            self.target_ore_idx,
                            self.ore_type,
                            self.exit_state,
                            self.switch_target,
                            self.do_foundry,
                            bail=self.bail,
                        ),
                    ),
                )
                return
            # you are on the tile, no enemy road, time to check cardinals

            x, y = POSITION_CACHE[self.target_ore_idx]

            print("before loop")

            for dx, dy in CARDINAL_DIRECTION_DELTAS:
                ny = y + dy
                nx = x + dx
                if not_in_bounds(nx, ny):
                    continue
                neighbour_idx = ny * MAP_INFO.width + nx

                # if wall continue
                if (BITBOARD_ENV[neighbour_idx] & 0b0010) != 0:
                    continue

                # if enemy indestructible sink on cardinal, bail
                if (BITBOARD_ENEMY[neighbour_idx] & 0b0001_0010_0011_1100) != 0:
                    # ['core', 'gunner', 'sentinel', 'breach', 'armoured_conveyor', 'foundry']
                    STATE.switch_to(ct, self.exit_state)
                    return

                # if enemy destructible, try and destroy
                if (BITBOARD_ENEMY[neighbour_idx] & 0b0010_0101_1000_0000) != 0:
                    # ['conveyor', 'splitter', 'bridge', 'road']
                    STATE.switch_to(
                        ct,
                        break_tile.state(
                            neighbour_idx,
                            State(
                                path_place_harvester,
                                self.target_ore_idx,
                                self.ore_type,
                                self.exit_state,
                                self.switch_target,
                                self.do_foundry,
                                bail=self.bail,
                            ),
                        ),
                    )
                    return

                # if ally building/conveyor, not road, marker or ,it is owned already => continue
                if (BITBOARD_ALLY[neighbour_idx] & 0b0101_1111_1111_1100) != 0:
                    # ['core', 'gunner', 'sentinel', 'breach', 'launcher', 'conveyor', 'splitter', 'armoured_conveyor', 'bridge', 'harvester', 'foundry']
                    continue

                # if ally road, try and break
                if (BITBOARD_ALLY[neighbour_idx] & 0b0010_0000_0000_0000) != 0:
                    if ct.can_destroy(idx_to_pos(neighbour_idx)):
                        ct.destroy(idx_to_pos(neighbour_idx))

                print(idx_to_pos(neighbour_idx))
                print("reached if")
                # if empty, ore or marker (by elimination) place conveyor
                if ct.can_build_conveyor(idx_to_pos(neighbour_idx), Direction.NORTH):
                    print("thinks can build")
                    if dy == 1:
                        ct.build_conveyor(idx_to_pos(neighbour_idx), Direction.NORTH)
                    elif dy == -1:
                        ct.build_conveyor(idx_to_pos(neighbour_idx), Direction.SOUTH)
                    elif dx == 1:
                        ct.build_conveyor(idx_to_pos(neighbour_idx), Direction.WEST)
                    elif dx == -1:
                        ct.build_conveyor(idx_to_pos(neighbour_idx), Direction.EAST)
                    return

            # TODO check and debug, should only reach here when setup

            if (
                (not self.do_foundry and not can_afford(ct, EntityType.HARVESTER))
                or (self.do_foundry and not can_afford(ct, EntityType.FOUNDRY))
            ) and (BITBOARD_ALLY[self.target_ore_idx] & 0b0010) != 0:
                # another bot probably is waiting to place harvester
                STATE.switch_to(ct, self.exit_state)
                return

            # only enact if can move, act, or afford harvester NOW
            if (
                ct.get_move_cooldown() != 0
                or ct.get_action_cooldown() != 0
                or (not self.do_foundry and not can_afford(ct, EntityType.HARVESTER))
                or (self.do_foundry and not can_afford(ct, EntityType.FOUNDRY))
            ):
                print("can't afford harvester/foundry")
                return

            print("moving...")
            move_to(ct, self.target_ore_idx, forwards=False)

            # clear the position
            if ct.can_destroy(target_ore_pos):
                ct.destroy(target_ore_pos)

            # build the harvester
            if not self.do_foundry:
                if build_if_can(ct, EntityType.HARVESTER, target_ore_pos):
                    ORES_COMPLETED[self.target_ore_idx] = 50
                    # switch back to old state
                    STATE.switch_to(ct, self.exit_state)
                    return
            else:
                if build_if_can(ct, EntityType.FOUNDRY, target_ore_pos):
                    # switch back to old state
                    STATE.switch_to(ct, self.exit_state)
                    return

    @classmethod
    def exit(self, ct: Controller):
        pass

    @classmethod
    def bail(self, ct: Controller, state: path_to.state):
        if (
            (not self.do_foundry and not can_afford(ct, EntityType.HARVESTER))
            or (self.do_foundry and not can_afford(ct, EntityType.FOUNDRY))
        ) and (BITBOARD_ALLY[self.target_ore_idx] & 0b0010) != 0:
            # another bot probably is waiting to place harvester
            ORES_COMPLETED[self.target_ore_idx] = 50
            STATE.switch_to(ct, self.exit_state)
            return


STATE.register(path_place_harvester)


class build_resource_line:
    # ------------- STATE -------------
    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            start_idx: int,
            end_idx: int,
            exit_state: State,
            reverse=False,
            merge=True,
            true_merge=False,
            armoured=False,
            skip_first=True,
            treat_ally_core_as_conveyor=True,
            inflate_ti_harvesters=False,
            check_flow=True,
            placed_lines: int = 0,
            stop_after: int = -1,
            end_on_bridge=False,
            continue_lines=None,
            bail: Callable[[Controller, build_resource_line.state], bool] | None = None,
        ):
            self.exit_state = exit_state

            self.current_idx = start_idx
            self.last_idx = start_idx

            self.start_idx = start_idx
            self.end_idx = end_idx

            self.unreachable = 0
            self.placed_lines = placed_lines
            self.allow_merging = merge
            self.treat_existing_lines_as_wall = true_merge
            self.treat_ally_core_as_conveyor = treat_ally_core_as_conveyor
            self.inflate_ti_harvesters = inflate_ti_harvesters
            self.end_on_bridge = end_on_bridge
            self.continue_lines = continue_lines
            self.check_flow = check_flow

            self.no_placed = 0
            self.reverse = reverse

            self.built = [0] * MAX_MAP_SIZE
            self.build_version = 1

            self.armoured = armoured

            self.stop_after = stop_after
            self.bail = bail
            self.skip_first = skip_first

            self.move_fatigue = task_fatigue(0)

            self.failed_to_build = False

            self.last_valid_path_solve: int | None = None
            self.give_up = 20

    current_state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

        RESOURCE_PATHING.invalidate_last_solve()
        RESOURCE_PATHING.solver.inflate_titanium_harvesters = self.inflate_ti_harvesters
        RESOURCE_PATHING.solver.placed_lines = self.placed_lines
        if self.continue_lines is not None:
            for line in self.continue_lines:
                RESOURCE_PATHING.solver.placed_lines |= line.placed_lines
        RESOURCE_PATHING.solver.allow_merging = self.allow_merging
        RESOURCE_PATHING.solver.treat_existing_lines_as_wall = (
            self.treat_existing_lines_as_wall
        )
        RESOURCE_PATHING.solver.treat_ally_core_as_conveyor = (
            self.treat_ally_core_as_conveyor
        )
        RESOURCE_PATHING.solver.check_flow = self.check_flow

        self.last_valid_path_solve = None

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        if self.bail is not None and self.bail(ct, self):
            return

        # if to core, update to closest open core slot every turn
        if self.end_idx == UNIT_INFO.ally_core_idx:
            closest_dist = -1
            for dx, dy in CORE_DELTAS:
                neighbour_idx = xy_to_idx(
                    UNIT_INFO.ally_core_pos.x + dx, UNIT_INFO.ally_core_pos.y + dy
                )
                if (BITBOARD_ALLY[neighbour_idx] & 0b1010_0000_0000_0000) != 0 or (
                    BITBOARD_ENV[neighbour_idx] & 0b0001
                ) != 0:
                    # Removed distance field method, instead just draw the line
                    dist_2 = (UNIT_INFO.ally_core_pos.x + dx - UNIT_INFO.position.x) * (
                        UNIT_INFO.ally_core_pos.x + dx - UNIT_INFO.position.x
                    ) + (UNIT_INFO.ally_core_pos.y + dy - UNIT_INFO.position.y) * (
                        UNIT_INFO.ally_core_pos.y + dy - UNIT_INFO.position.y
                    )
                    if dist_2 < closest_dist or closest_dist == -1:
                        closest_dist = dist_2
                        self.end_idx = neighbour_idx
                        print(f"set target: {idx_to_pos(neighbour_idx)}")

        if self.current_idx == self.end_idx:
            # End of resource line
            print("end of resource line")
            STATE.switch_to(ct, self.exit_state)
            return

        if (BITBOARD_ALLY[self.end_idx] & 0b0100) != 0 and (
            BITBOARD_ALLY[self.current_idx] & 0b0100
        ) != 0:
            # [ "ally_core" ]
            # [ "ally_core" ]
            # trying to resource line to core, we hit some core point - end
            print("reached core")
            STATE.switch_to(ct, self.exit_state)
            return

        # reachability check
        print(
            f"pre reachable check {POSITION_CACHE[self.start_idx]} {POSITION_CACHE[self.end_idx]}"
        )

        RESOURCE_PATHING.reachable = 0
        RESOURCE_PATHING.mark_reachable(self.current_idx)
        RESOURCE_PATHING.mark_reachable(self.end_idx)
        # Update what we know is unreachable for the bot so resource pathing only
        # creates a path the bot can actually build
        RESOURCE_PATHING.unreachable = self.unreachable
        RESOURCE_PATHING.solve(self.end_idx, stop_idx=self.current_idx)
        RESOURCE_PATHING.unreachable = 0

        if not RESOURCE_PATHING.ready():
            print("NOT READY")
            return

        # RESOURCE_PATHING.debug_path(self.current_idx)

        # start_idx, end_idx not reachable => exit the state
        if not RESOURCE_PATHING.has_visited(self.current_idx):
            if (
                self.last_valid_path_solve is not None
                and RESOURCE_PATHING.solve_version > self.last_valid_path_solve
            ):
                # start is unreachable - you cannot build a resouce line from end_idx to start_idx
                self.failed_to_build = True
                STATE.switch_to(ct, self.exit_state)
                return
            else:
                # Could be unreachable, but wait for one more solve to progress in case
                # solver just hasn't reacted to changes yet (solve occures over multiple frames)
                self.last_valid_path_solve = RESOURCE_PATHING.solve_version
                return

        next_tile_idx = RESOURCE_PATHING.get_next_tile(self.current_idx)

        if next_tile_idx is None:
            if (
                self.last_valid_path_solve is not None
                and RESOURCE_PATHING.solve_version > self.last_valid_path_solve
            ):
                self.failed_to_build = True
                STATE.switch_to(ct, self.exit_state)
                return
            else:
                # Could be unreachable, but wait for one more solve to progress in case
                # solver just hasn't reacted to changes yet (solve occures over multiple frames)
                self.last_valid_path_solve = RESOURCE_PATHING.solve_version
                return

        # We have a path
        self.last_valid_path_solve = None

        # if we are on start, push one
        if self.current_idx == self.start_idx and self.skip_first:
            if next_tile_idx == self.end_idx:
                print("we reached the end!")
                # next tile is the exit, we are actually already done, we can exit
                STATE.switch_to(ct, self.exit_state)
                return

            next_next_tile = RESOURCE_PATHING.get_next_tile(next_tile_idx)
            if next_next_tile is None:
                # Bitboard did not finish updating newly spotted walls, so the next tile position
                # was a wall, wait until bitboard re-syncs
                return

            self.current_idx = next_tile_idx
            next_tile_idx = next_next_tile

        built = False
        bridge_placed = False

        if (
            self.end_on_bridge
            and (
                dist := (
                    current_tile_pos := POSITION_CACHE[self.current_idx]
                ).distance_squared(next_tile_pos := POSITION_CACHE[self.end_idx])
            )
            <= 9
            and dist > 0
        ):
            # overwrite logic, TODO consider whether to overwrite armoured conveyor
            if (BITBOARD_ALLY[self.current_idx] & 0b1010_0111_1000_0000) != 0:
                # ['conveyor', 'splitter', 'armoured_conveyor', 'bridge', 'road', 'marker']
                if ct.can_destroy(current_tile_pos) and can_afford(
                    ct, EntityType.BRIDGE
                ):
                    ct.destroy(current_tile_pos)

            # if enemy destructible in the way, try to remove it
            if (BITBOARD_ENEMY[self.current_idx] & 0b1010_0111_1000_0000) != 0:
                if STATE.tick > BREAK_TILE_COOLDOWN[self.current_idx]:
                    print("break tile!!!")
                    STATE.switch_to(
                        ct,
                        break_tile.state(self.current_idx, self),
                    )
                    return
                else:
                    RESOURCE_PATHING.invalidate_last_solve()
                    self.unreachable |= BITBOARD_TARGETS[self.current_idx]
                    self.current_idx = self.last_idx
                    return

            print(f"{current_tile_pos} {next_tile_pos}")

            if build_if_can(ct, EntityType.BRIDGE, current_tile_pos, self.end_idx):
                self.built[self.current_idx] = self.build_version
                self.last_idx = self.current_idx
                self.current_idx = next_tile_idx

                STATE.switch_to(ct, self.exit_state)
                return

        # MERGE LOGIC if not this version, and conveyor type in the correct direction, skip to next current index
        # directly
        if (
            not built
            and self.built[self.current_idx] != self.build_version
            and (BITBOARD_ALLY[self.current_idx] & 0b0111_1000_0000) != 0
            # ['conveyor', 'splitter', 'armoured_conveyor', 'bridge']
        ):
            next_tile_pos = POSITION_CACHE[next_tile_idx]
            current_tile_pos = POSITION_CACHE[self.current_idx]
            if current_tile_pos.distance_squared(next_tile_pos) == 1:
                conveyor_direction = current_tile_pos.direction_to(next_tile_pos)
                if (
                    BITBOARD_ALLY[self.current_idx] & 0b0011_1000_0000
                    # conveyor, armoured, splitter
                ) != 0 and get_building_direction(
                    ALLY_BUILDINGS[self.current_idx]
                ) == conveyor_direction:
                    # print(f"skip already built conveyor")
                    # built = True
                    self.last_idx = self.current_idx
                    self.current_idx = next_tile_idx
                    STATE.switch_to(ct, self.exit_state)
                    return
            else:
                if (
                    BITBOARD_ALLY[self.current_idx] & 0b0100_0000_0000
                ) != 0 and get_bridge_target_idx(
                    ALLY_BUILDINGS[self.current_idx]
                ) == next_tile_idx:
                    # print(f"skip already built bridge")
                    # built = True
                    # bridge_placed = True
                    self.last_idx = self.current_idx
                    self.current_idx = next_tile_idx
                    STATE.switch_to(ct, self.exit_state)
                    return

        # are you in range to perform action?
        if (
            not built
            and UNIT_INFO.position.distance_squared(POSITION_CACHE[self.current_idx])
            <= 2
        ):
            next_tile_pos = POSITION_CACHE[next_tile_idx]
            current_tile_pos = POSITION_CACHE[self.current_idx]

            # TODO check if we can afford bridge / conveyor before destroying ???

            # overwrite logic, TODO consider whether to overwrite armoured conveyor
            if (BITBOARD_ALLY[self.current_idx] & 0b1010_0111_1000_0000) != 0:
                # ['conveyor', 'splitter', 'armoured_conveyor', 'bridge', 'road', 'marker']
                if ct.can_destroy(current_tile_pos) and (
                    (
                        current_tile_pos.distance_squared(next_tile_pos) == 1
                        and can_afford(ct, EntityType.CONVEYOR)
                    )
                    or can_afford(ct, EntityType.BRIDGE)
                ):
                    ct.destroy(current_tile_pos)

            # if enemy destructible in the way, try to remove it
            if (BITBOARD_ENEMY[self.current_idx] & 0b1010_0111_1000_0000) != 0:
                if STATE.tick > BREAK_TILE_COOLDOWN[self.current_idx]:
                    print("break tile!!!")
                    STATE.switch_to(
                        ct,
                        break_tile.state(self.current_idx, self),
                    )
                    return
                else:
                    RESOURCE_PATHING.invalidate_last_solve()
                    self.unreachable |= BITBOARD_TARGETS[self.current_idx]
                    self.current_idx = self.last_idx
                    return

            # if conveyor (distance is only one), build it
            if current_tile_pos.distance_squared(next_tile_pos) == 1:
                conveyor_direction = current_tile_pos.direction_to(next_tile_pos)
                if self.armoured:
                    built = build_if_can(
                        ct,
                        EntityType.ARMOURED_CONVEYOR,
                        current_tile_pos,
                        conveyor_direction,
                    )
                else:
                    built = build_if_can(
                        ct, EntityType.CONVEYOR, current_tile_pos, conveyor_direction
                    )
            # if distance > 1 must be a BRIDGE
            else:
                built = build_if_can(
                    ct, EntityType.BRIDGE, current_tile_pos, next_tile_pos
                )
                bridge_placed = built

        if built:
            # Let pather know that the tile we placed is part of the line we are constructing
            self.no_placed += 1
            print(f"no placed: {self.no_placed}")

            RESOURCE_PATHING.solver.mark_placed_lines(self.current_idx)
            self.placed_lines = RESOURCE_PATHING.solver.placed_lines

            self.built[self.current_idx] = self.build_version
            self.last_idx = self.current_idx
            self.current_idx = next_tile_idx

            self.give_up = 20

            if self.stop_after > 0:
                self.stop_after -= 1
            if self.stop_after == 0:
                self.stop_after = -1
                print("exit due to stop_after")
                STATE.switch_to(ct, self.exit_state)
                return

            if bridge_placed:
                print("bridge placed")
                STATE.switch_to(
                    ct,
                    bridge_coor,
                    self.last_idx,
                    self.current_idx,
                    self,
                )
                return

        # TODO convert this code to use `path_to` state rather than handling move manually
        #      this can't handle unreachable very well

        BOT_PATHING.solve(self.current_idx, stop_idx=UNIT_INFO.position_idx)
        if BOT_PATHING.ready():
            # BOT_PATHING.debug_path(UNIT_INFO.position_idx)
            if not BOT_PATHING.has_visited(UNIT_INFO.position_idx):
                print("unreachable according to bot pathing!")
                # If we cannot reach target location, mark it as unreachable and invalidate solve
                RESOURCE_PATHING.invalidate_last_solve()
                self.unreachable |= BOT_PATHING.visited
                self.current_idx = self.last_idx

                # TODO there will be a point where bot cannot path at all no matter how many resource line reroutes
                #      need to full exit state at that point
            else:
                DISTANCE_FIELD.solve(self.current_idx, stop_idx=UNIT_INFO.position_idx)
                if DISTANCE_FIELD.ready():
                    if not DISTANCE_FIELD.has_visited(UNIT_INFO.position_idx):
                        print("unreachable according to distance field!")
                        # If we cannot reach target location, mark it as unreachable and invalidate solve
                        RESOURCE_PATHING.invalidate_last_solve()
                        self.unreachable |= DISTANCE_FIELD.visited
                        self.current_idx = self.last_idx

                        # TODO there will be a point where bot cannot path at all no matter how many resource line reroutes
                        #      need to full exit state at that point

                if self.move_fatigue.is_fatigued():
                    if random.random() < 0.5:
                        move_to(
                            ct,
                            self.current_idx,
                            solve_already_done=True,
                            forwards=False,
                        )
                    return

                move_to(ct, self.current_idx, solve_already_done=True)
                if BOT_PATHING.solver.fallback_move:
                    self.give_up -= 1
                    if self.give_up <= 0:
                        STATE.switch_to(ct, self.exit_state)
                        return
                    self.move_fatigue.fatigue = random.randint(0, 3)

    @classmethod
    def exit(self, ct: Controller):
        pass


STATE.register(build_resource_line)


class bridge_coor:
    class state:
        STATE_ID: int = -1

    current_tile_idx: int
    next_tile_idx: int
    fatigue: int

    exit_state: State

    @classmethod
    def enter(
        self,
        ct: Controller,
        current_tile_idx,
        next_tile_idx,
        exit_state: State,
        fatigue: int = 1,
    ):
        # WILL BE ON THE BRIDGE
        # CHECK FOR ALLY BOTS
        # WAIT X TURNS TO SEE IF THEY CONTINUE LINE
        # CHECK IF WALKING IS CHEAP, access RESOURCE_PATHING.dist() (make sure its solved first - RESOURCE_PATHING.ready())

        self.current_tile_idx = current_tile_idx
        self.next_tile_idx = next_tile_idx
        self.fatigue = fatigue
        self.exit_state = exit_state

        print("Bridge coordinating...")

    @classmethod
    def run(self, ct: Controller):
        if UNIT_INFO.position_idx != self.current_tile_idx:
            move_to(ct, self.current_tile_idx)

        if (
            BITBOARD_ALLY[self.next_tile_idx] & 0b0001_0111_1011_1100
        ) != 0 or self.fatigue == 0:
            # ['core', 'gunner', 'sentinel', 'breach', 'conveyor', 'splitter', 'armoured_conveyor', 'bridge', 'foundry']
            # if next tile, is not a result, or fatigue is out, switch back
            STATE.switch_to(ct, self.exit_state)
            return

        # Solve distance field, if can walk 20 tiles, dont bother coordinating
        DISTANCE_FIELD.solve(
            self.current_tile_idx, stop_idx=self.next_tile_idx, stop_cost=20
        )
        if (
            DISTANCE_FIELD.ready()
            and (dist := DISTANCE_FIELD.dist(self.next_tile_idx)) is not None
            and dist < 20
        ):
            STATE.switch_to(ct, self.exit_state)
            return

        # check for nearby allies
        ally_present = False
        for idx in vision_iter():
            if (BITBOARD_ALLY[idx] & 0b0010) != 0:
                # ['builder_bot']
                ally_present = True

        if ally_present:
            self.fatigue -= 1
        else:
            STATE.switch_to(ct, self.exit_state)
            return

    @classmethod
    def exit(self, ct: Controller):
        pass


STATE.register(bridge_coor)


class check_ore_groups:
    # ------------- STATE -------------

    STATE_ID: int = -1

    class state:
        def __init__(
            self,
            ore_bits: int,
            target_ore_idx: int,
            destination_idx: int,
            exit_state: State,
            resource_bail: Callable[[build_resource_line.state], bool] | None = None,
            origin_path_to: path_to.state | None = None,
        ):
            self.exit_state = exit_state
            self.destination_idx = destination_idx
            self.ore_bits = ore_bits
            self.target_ore_idx = target_ore_idx
            self.group_ore_idx_list: list = None
            self.resource_bail = resource_bail
            self.origin_path_to = origin_path_to

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

        print("Checking for nearby ore")

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        candidate_idx_list = []
        distances = []

        if (
            self.origin_path_to is not None
            and not self.origin_path_to.path_to_succeeded
        ):
            # If failed to path to ore mark it as completed
            ORES_COMPLETED[self.origin_path_to.target_idx] = 50

        DISTANCE_FIELD.solve(UNIT_INFO.position_idx, stop_cost=4)
        if not DISTANCE_FIELD.ready():
            return

        if DISTANCE_FIELD.has_visited(UNIT_INFO.ally_core_idx):
            print("fast route")
            STATE.switch_to(
                ct,
                State(
                    path_place_harvester,
                    self.target_ore_idx,
                    self.ore_bits,
                    build_resource_line.state(
                        self.target_ore_idx,
                        self.destination_idx,
                        self.exit_state,
                        merge=(False if self.ore_bits == 0b1000 else True),
                    ),
                ),
            )
            return

        # TODO DONT SCAN ALLY ENEMY BUILDINGS (ALREADY TAKEN ORE SPOTS)

        for idx in vision_iter():
            if (BITBOARD_ENV[idx] & self.ore_bits) != 0 and not ORES_COMPLETED[idx]:
                dist = DISTANCE_FIELD.dist(idx)
                if dist is not None:
                    candidate_idx_list.append(idx)
                    distances.append(dist)

        closest_4 = sorted(range(len(distances)), key=distances.__getitem__)[:4]
        # TODO REVERSE

        self.group_ore_idx_list = []

        for idx in closest_4:
            # group_ore_idx_list.append(pos_to_idx(candidate_pos_list[idx]))
            self.group_ore_idx_list.append(candidate_idx_list[idx])
            if ct.get_global_resources()[0] < 500:
                break

        print(
            f"closest 4 ores: {[POSITION_CACHE[idx] for idx in self.group_ore_idx_list]}"
        )

        # TODO where to switch to
        STATE.switch_to(ct, collect_ore_groups.state(self, self.exit_state))
        return

    @classmethod
    def exit(self, ct: Controller):
        pass


STATE.register(check_ore_groups)


class collect_ore_groups:
    # ------------- STATE -------------

    class state:
        STATE_ID: int = -1

        def __init__(self, check_ore_groups_state: check_ore_groups.state, exit_state):
            self.exit_state = exit_state

            self.check_ore_groups = check_ore_groups_state

            self.merge_point = None
            self.current_idx = -1

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        group_size = len(self.check_ore_groups.group_ore_idx_list)
        self.current_idx += 1

        print(f"{self.current_idx + 1}/{group_size}")

        if self.current_idx < group_size:
            print("collect -> path place")
            STATE.switch_to(
                ct,
                path_place_harvester,
                self.check_ore_groups.group_ore_idx_list[self.current_idx],
                self.check_ore_groups.ore_bits,
                self,
            )
            ORES_COMPLETED[
                self.check_ore_groups.group_ore_idx_list[self.current_idx]
            ] = 50
            # TODO need a more reliable way to mark ores as complete
            # since path_place_harvester may not do it (unreachable)
            return

        # TODO build the lines
        # most recent harvester
        self.current_idx -= 1

        print("finished collection")
        STATE.switch_to(
            ct,
            merge_ore_groups.state(self.check_ore_groups, self.exit_state),
        )

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state


STATE.register(collect_ore_groups)


class merge_ore_groups:
    # ------------- STATE -------------
    class state:
        STATE_ID: int = -1

        def __init__(self, check_ore_groups_state: check_ore_groups.state, exit_state):
            self.exit_state = exit_state

            self.check_ore_groups = check_ore_groups_state
            self.original_resource_line: build_resource_line.state | None = None

            self.merge_point = None
            self.current_idx = -1

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        group_size = len(self.check_ore_groups.group_ore_idx_list)
        self.current_idx += 1

        print(f"{self.current_idx + 1}/{group_size}")

        if self.current_idx < group_size:
            if (
                self.original_resource_line == None
                or self.original_resource_line.failed_to_build
            ):
                print(f"starting line")
                self.original_resource_line = build_resource_line.state(
                    self.check_ore_groups.group_ore_idx_list[self.current_idx],
                    self.check_ore_groups.destination_idx,
                    State(merge_ore_groups, self),
                    stop_after=2,
                    merge=(
                        False
                        if group_size >= 2 or self.check_ore_groups.ore_bits == 0b1000
                        else True
                    ),
                )
                STATE.switch_to(ct, self.original_resource_line)
            else:
                print(
                    f"merge to {POSITION_CACHE[self.original_resource_line.last_idx]}"
                )
                STATE.switch_to(
                    ct,
                    build_resource_line,
                    build_resource_line.state(
                        self.check_ore_groups.group_ore_idx_list[self.current_idx],
                        self.original_resource_line.last_idx,
                        State(merge_ore_groups, self),
                    ),
                )
            return

        if self.original_resource_line is not None:
            self.original_resource_line.exit_state = self.exit_state
            self.original_resource_line.bail = self.check_ore_groups.resource_bail
            self.original_resource_line.stop_after = -1
            STATE.switch_to(ct, self.original_resource_line)
        else:
            STATE.switch_to(ct, self.exit_state)

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state


STATE.register(merge_ore_groups)
