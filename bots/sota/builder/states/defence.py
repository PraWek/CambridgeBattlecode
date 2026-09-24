from __future__ import annotations

from cambc import Controller, Direction, EntityType, Position
from typing import Callable
from botlib import State
from .. import STATE

from botlib import MAP_INFO, UNIT_INFO, ALLY_BUILDINGS
from botlib import (
    best_enemy_core_idx,
    idx_to_pos,
    xy_to_idx,
    pos_to_idx,
    get_building_hp,
    get_building_maxhp,
    get_building_id,
    get_building,
    get_bridge_target_idx,
    get_building_type,
    get_building_direction,
    not_in_bounds,
    bind_bail,
    vision_iter,
    get_bot_id,
    SINKS,
    RESOURCE_FLOW,
    CARDINAL_DIRECTION_DELTAS,
    BITBOARD_ALLY,
    BITBOARD_ENEMY,
    BITBOARD_ENV,
    ENEMY_BUILDINGS,
    ENEMY_BUILDER_BOT,
    ALLY_BUILDER_BOT,
    print_flow_values,
    cardinal_iter,
    distance_sq_to,
    format_bits,
    adjacent_iter,
)
from botlib.constants import POSITION_CACHE, DIRECTION_CACHE

from ..utility.movement import move_to
from ..utility.building import direction_to, destroy, build_if_can_afford

from ..data import BOT_PATHING

from .fundamentals import path_to, move_to, break_tile, heal_target

from .gather_resource import build_resource_line

from .attacks import smart_turret_place, attack_harvester

# TODO testing of vvvv state


# Only returns ally sources, returns None if no ally source found
def check_turret_sinks(ct: Controller, idx: int, max_depth: int):
    if max_depth <= 0:
        return None

    print(f"checking: {POSITION_CACHE[idx]} depth={max_depth}")

    # if it has a source you own, reroute destroy
    bridge_sources = SINKS[idx].bridge_sources
    conveyor_sources = SINKS[idx].conveyor_sources
    harvester_sources = SINKS[idx].harvester_sources
    foundry_sources = SINKS[idx].foundry_sources

    soft_sources = bridge_sources | conveyor_sources
    solid_sources = harvester_sources | foundry_sources

    for source in soft_sources:
        if path_to.is_on_cooldown(source):
            continue

        if ALLY_BUILDINGS[source] != 0:
            return source
        else:
            down_stream = check_turret_sinks(ct, source, max_depth - 1)
            if down_stream is not None:
                return down_stream

    for source in solid_sources:
        if path_to.is_on_cooldown(source):
            continue

        if ALLY_BUILDINGS[source] != 0:
            return source

    return None


class reroute_destroy:
    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            target_idx: int,
            exit_state,
            target_id: int = -1,
            fatigue: int = 5,
            known_turret_sink_idx: int | None = None,
            bail: Callable[[Controller, heal_target.state], bool] | None = None,
        ):
            self.target_idx = target_idx
            self.known_turret_sink_idx = known_turret_sink_idx
            self.exit_state = exit_state

            self.target_id = target_id
            self.fatigue = fatigue

            self.bail = bail

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

        if self.target_id == -1:
            self.target_id = get_building_id(get_building(self.target_idx))

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        print(f"target: {POSITION_CACHE[self.target_idx]}")

        self.fatigue -= 1

        # bail function
        if self.bail is not None and self.bail(ct, self):
            return

        # leave if target is destroyed
        if self.target_id != get_building_id(get_building(self.target_idx)):
            STATE.switch_to(ct, self.exit_state)
            return

        # leave if fatigued
        if self.fatigue == 0:
            STATE.switch_to(ct, self.exit_state)
            return

        turret_sink_idx = self.known_turret_sink_idx
        if self.known_turret_sink_idx is None:
            turret_sink_idx = check_turret_sinks(ct, self.target_idx, 3)
        else:
            self.known_turret_sink_idx = None

        if turret_sink_idx is None:
            STATE.switch_to(ct, self.exit_state)
            return

        if path_to.is_on_cooldown(turret_sink_idx):
            print("Cannot reach target sink")
            STATE.switch_to(ct, self.exit_state)
            return

        # foundary / harvester
        is_source_solid = (BITBOARD_ALLY[turret_sink_idx] & 0b0001_1000_0000_0000) != 0

        idx = turret_sink_idx
        pos = idx_to_pos(idx)
        if is_source_solid:
            self.fatigue += 1
            # if solid source is out of range, path to
            if UNIT_INFO.position.distance_squared(pos) > 2:
                STATE.switch_to(
                    ct,
                    path_to.state(idx, 2, self),
                )
                return

            # in range, try to attack it
            # if two + enemy turrets, now go destroy it and replace with a barrier
            enemy_turret_count = 0
            for cardinal_idx in cardinal_iter(idx):
                if (BITBOARD_ENEMY[cardinal_idx] & 0b0011_1000) != 0:
                    # gunner sentinel breach
                    enemy_turret_count += 1

            if UNIT_INFO.position.distance_squared(pos) <= 2 and enemy_turret_count >= 2:
                if ct.can_destroy(pos):
                    ct.destroy(pos)
                    if ct.can_build(EntityType.BARRIER, pos):
                        ct.build_barrier(pos)
            else:
                spot_available = False
                is_being_fought = False
                x, y = pos

                for dx, dy in CARDINAL_DIRECTION_DELTAS:
                    nx = x + dx
                    ny = y + dy
                    if not_in_bounds(nx, ny):
                        continue
                    neighbour_idx = ny * MAP_INFO.width + nx
                    if break_tile.is_on_cooldown(neighbour_idx) or path_to.is_on_cooldown(
                        neighbour_idx
                    ):
                        continue

                    if (BITBOARD_ALLY[neighbour_idx] & 0b1000) != 0:
                        tx, ty = POSITION_CACHE[self.target_idx]
                        # gunner
                        if tx != nx and ty != ny:
                            is_being_fought = True
                    elif (BITBOARD_ALLY[neighbour_idx] & 0b0001_0000) != 0:
                        # sentinel
                        building_dir = get_building_direction(get_building(neighbour_idx))
                        if pos in ct.get_attackable_tiles_from(
                            idx_to_pos(neighbour_idx), building_dir, EntityType.SENTINEL
                        ):
                            is_being_fought = True

                    if (
                        (BITBOARD_ALLY[neighbour_idx] & 0b0101_1010_0111_1110) == 0
                        and (BITBOARD_ENEMY[neighbour_idx] & 0b0101_1010_0111_1110) == 0
                        and (BITBOARD_ENV[neighbour_idx] & 0b0010) == 0
                    ):
                        # ['builder_bot', 'core', 'gunner', 'sentinel', 'breach', 'launcher', 'armoured_conveyor', 'harvester', 'foundry', 'barrier']
                        # wall
                        # if no impassibles, there is a valid spot so break away
                        spot_available = True
                        break

                print(f"is fought: {is_being_fought}")

                if spot_available:
                    # TODO switch to "defend_harvester"
                    STATE.switch_to(ct, attack_harvester.state(idx, self))
                    return
                elif not is_being_fought:
                    if ct.can_destroy(pos):
                        ct.destroy(pos)
                        if ct.can_build(EntityType.BARRIER, pos):
                            ct.build_barrier(pos)
                    return
        else:
            # if soft source has flow
            if (
                RESOURCE_FLOW[0][idx].get_flow() > 0
                or RESOURCE_FLOW[0][idx].get_stall() > 0
                or RESOURCE_FLOW[2][idx].get_flow() > 0
                or RESOURCE_FLOW[2][idx].get_stall() > 0
            ):
                self.fatigue += 1
                # if soft source is out of range, path to
                if UNIT_INFO.position.distance_squared(pos) > 2:
                    STATE.switch_to(
                        ct,
                        path_to.state(idx, 2, self),
                    )
                    return
                # if soft source is in range, destroy source
                if UNIT_INFO.position.distance_squared(pos) <= 2:
                    print("breaking and building turret")
                    if ct.can_destroy(pos):
                        ct.destroy(pos)
                    # if the cardinal, build gunner
                    if UNIT_INFO.position.distance_squared(pos) == 1:
                        if ct.can_build(
                            EntityType.GUNNER,
                            pos,
                            direction_to(pos, idx_to_pos(self.target_idx)),
                        ):
                            ct.build_gunner(pos, direction_to(pos, idx_to_pos(self.target_idx)))
                            return
                    # if the cardinal, build sentinel
                    else:
                        if ct.can_build(
                            EntityType.SENTINEL,
                            pos,
                            direction_to(pos, idx_to_pos(self.target_idx)),
                        ):
                            ct.build_sentinel(pos, direction_to(pos, idx_to_pos(self.target_idx)))
                            return
            # if no flow, just break sink
            else:
                self.fatigue += 1
                # if soft source is out of range, path to
                if UNIT_INFO.position.distance_squared(pos) > 2:
                    STATE.switch_to(
                        ct,
                        path_to.state(idx, 2, self),
                    )
                    return
                # if soft source is in range, destroy source
                if UNIT_INFO.position.distance_squared(pos) <= 2:
                    print("breaking sink")
                    if ct.can_destroy(pos):
                        ct.destroy(pos)
                    build_if_can_afford(ct, EntityType.BARRIER, pos)
                    return

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state


STATE.register(reroute_destroy)


class repair_throughput:
    # you should trigger this state targetting a tile you think is stalled out
    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            target_idx: int,
            exit_state,
            bail: Callable[[Controller, heal_target.state], bool] | None = None,
        ):
            self.target_idx = target_idx
            self.exit_state = exit_state

            self.fixed = False
            self.forward = True
            self.bail = bail

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        # bail function
        if self.bail is not None and self.bail(ct, self):
            return

        print(f"repairing from {POSITION_CACHE[self.target_idx]}")

        # if marked as complete, exit
        if self.fixed:
            print(f"line is fixed")
            STATE.switch_to(ct, self.exit_state)

        # FORWARD PASS, DOWN THE LINE, WITH THE FLOW OF RESOURCE
        # walk the target idx along the line, and move with it, keep it in vision at all times
        # TODO, check flow here
        seen = set()
        while self.forward:
            if self.target_idx in seen:
                break
            seen.add(self.target_idx)

            pos = idx_to_pos(self.target_idx)

            if not ct.is_in_vision(pos):
                # TODO use `path_to` state rather than move_to, use a "is_in_vision" bail out to return
                #      path_to has better movement management (handles fatigue and unreachable)
                move_to(ct, self.target_idx)
                return

            # TODO check, you cant repair throughput on enemy tiles
            type = get_building_type(ALLY_BUILDINGS[self.target_idx])

            if type == EntityType.CONVEYOR or type == EntityType.ARMOURED_CONVEYOR:
                dir = get_building_direction(ALLY_BUILDINGS[self.target_idx])
                dx, dy = dir.delta()
                x, y = pos.x, pos.y

                new_target_idx = xy_to_idx(x + dx, y + dy)
                if (BITBOARD_ALLY[new_target_idx] & 0b0001_1000_0000_0000) != 0:
                    # foundary + harvester
                    # if the target is going into a foundary or harvester, we should not proceed
                    break

                self.target_idx = new_target_idx

            elif type == EntityType.BRIDGE:

                new_target_idx = get_bridge_target_idx(ALLY_BUILDINGS[self.target_idx])
                if (BITBOARD_ALLY[new_target_idx] & 0b0001_1000_0000_0000) != 0:
                    # foundary + harvester
                    # if the target is going into a foundary or harvester, we should not proceed
                    break

                self.target_idx = new_target_idx

            elif type == EntityType.SPLITTER:
                cardinals = [
                    Direction.NORTH,
                    Direction.EAST,
                    Direction.SOUTH,
                    Direction.WEST,
                ]
                splitter_dirs = cardinals.remove(
                    get_building_direction(ALLY_BUILDINGS[self.target_idx])
                )
                new_target_idx = None
                for dir in splitter_dirs:
                    dir = get_building_direction(ALLY_BUILDINGS[self.target_idx])
                    dx, dy = dir.delta()
                    x, y = pos.x, pos.y
                    splitter_idx = xy_to_idx(x + dx, y + dy)
                    if (BITBOARD_ALLY[splitter_idx] & 0b0001_1000_0000_0000) != 0:
                        # foundary + harvester
                        # if the target is going into a foundary or harvester, we should not proceed
                        continue
                    new_target_idx = splitter_idx

                if new_target_idx is None:
                    break

                self.target_idx = new_target_idx
            else:
                break

        print(f"repairing new target {POSITION_CACHE[self.target_idx]}")

        # forward pass responses
        if self.forward:
            print(f"im here")

            # if it routes to enemy (not road, marker or bot though)
            if (BITBOARD_ENEMY[self.target_idx] & (~0b1010_0000_0000_0010)) != 0:
                self.fixed = True
                STATE.switch_to(
                    ct,
                    reroute_destroy,
                    reroute_destroy.state(self.target_idx, self.exit_state, bail=self.bail),
                )
                return

            # if out of range, move into range, then build resource line home
            if UNIT_INFO.position.distance_squared(idx_to_pos(self.target_idx)) > 2:
                STATE.switch_to(
                    ct,
                    path_to.state(self.target_idx, 2, self, bail=self.bail),
                )
                return
            else:
                self.fixed = True
                destroy(ct, idx_to_pos(self.target_idx))
                STATE.switch_to(
                    ct,
                    build_resource_line,
                    build_resource_line.state(
                        self.target_idx,
                        UNIT_INFO.ally_core_idx,
                        self.exit_state,
                        skip_first=False,
                        bail=self.bail,
                    ),
                )
                return

        # # BACKWARD PASS
        # while not self.forward:
        #     pos = idx_to_pos(self.target_idx)

        #     if not ct.is_in_vision(pos):
        #         # TODO use `path_to` state rather than move_to, use a "is_in_vision" bail out to return
        #         #      path_to has better movement management (handles fatigue and unreachable)
        #         move_to(ct, self.target_idx)
        #         return

        #     type = ct.get_entity_type(pos)

        #     bridge_sources = SINKS[self.target_idx].bridge_sources
        #     conveyor_sources = SINKS[self.target_idx].conveyor_sources
        #     harvester_sources = SINKS[self.target_idx].harvester_sources
        #     foundry_sources = SINKS[self.target_idx].foundry_sources

        #     soft_sources = bridge_sources | conveyor_sources
        #     solid_sources = harvester_sources | foundry_sources

        #     if len(soft_sources) == 0:
        #         # if you hit a solid source, then reoute the current tile to base, true merge on,
        #         if len(solid_sources) != 0:
        #             # if the current tile is sourced by a solid source, destroy then reroute
        #             destroy(ct, pos)
        #             STATE.switch_to(
        #                 ct,
        #                 build_resource_line,
        #                 build_resource_line.state(
        #                     self.target_idx,
        #                     UNIT_INFO.ally_core_idx,
        #                     self.exit_state,
        #                     skip_first=False,
        #                 ),
        #             )
        #             pass
        #         else:
        #             print("couldnt any sources, exiting")
        #             STATE.switch_to(ct, self.exit_state)
        #             return

        #     self.target_idx = soft_sources[0]

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state


STATE.register(repair_throughput)


class repair_harvester:
    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            harvester_idx,
            destination_idx,
            exit_state,
            bail: Callable[[Controller, repair_harvester.state], bool] | None = None,
        ):
            self.harvester_idx = harvester_idx
            self.destination_idx = destination_idx
            self.exit_state = exit_state
            self.bail = bail

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        if self.bail is not None and self.bail(ct, self):
            return

        if (BITBOARD_ALLY[self.harvester_idx] & 0b1000_0000_0000) == 0:
            print("Not our harvester, replacing...")
            # not a harvester, was probably barrier claimed, need to reclaim
            STATE.switch_to(
                ct,
                break_tile.state(
                    self.harvester_idx, self, bail=self.bail, replace_with=EntityType.HARVESTER
                ),
            )
            return

        if (BITBOARD_ENEMY[self.harvester_idx] & 0b0101_1010_0111_1010) != 0:
            print("Enemy somehow captured the ore spot, bailing")
            STATE.switch_to(ct, self.exit_state)
            return

        # TODO Choose better neighbour rather than only based on distance to destination
        best_neighbour_idx = None
        best_neighbour_dist = 0
        best_has_turret = False
        for neighbour_idx in cardinal_iter(self.harvester_idx):
            if (BITBOARD_ENV[neighbour_idx] & 0b0010) != 0:
                # skip walls
                continue

            if break_tile.is_on_cooldown(neighbour_idx):
                continue

            if path_to.is_on_cooldown(neighbour_idx):
                continue

            has_turret = (BITBOARD_ALLY[neighbour_idx] & 0b0011_1000) != 0
            dist = distance_sq_to(neighbour_idx, self.destination_idx)
            if best_neighbour_idx is None or (
                dist < best_neighbour_dist and best_has_turret and not has_turret
            ):
                best_neighbour_idx = neighbour_idx
                best_neighbour_dist = dist
                best_has_turret = has_turret

        if best_neighbour_idx is None:
            print("Cannot find best neighbour")
            STATE.switch_to(ct, self.exit_state)
            return

        if (BITBOARD_ALLY[best_neighbour_idx] & (~0b0111_1000_0000)) != 0 or BITBOARD_ENEMY[
            best_neighbour_idx
        ] != 0:
            # Walk to destroy ally or enemy building
            STATE.switch_to(ct, break_tile.state(best_neighbour_idx, self, bail=self.bail))
            return

        print(f"starting resource line at: {POSITION_CACHE[best_neighbour_idx]}")

        STATE.switch_to(
            ct,
            build_resource_line.state(
                best_neighbour_idx,
                self.destination_idx,
                self.exit_state,
                bail=self.bail,
                skip_first=False,
            ),
        )
        return

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state

    # ------------- METHODS -------------

    @classmethod
    def example_method(self):
        pass


STATE.register(repair_harvester)


class follow_enemy:
    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            target_bot_id: int,
            exit_state,
            target_id: int = -1,
            fatigue: int = 40,
            stop_on_better_ally_nearby=False,
            bail: Callable[[Controller, follow_enemy.state], bool] | None = None,
        ):
            self.target_bot_id = target_bot_id
            self.exit_state = exit_state

            self.target_id = target_id
            self.fatigue = fatigue

            self.stop_on_better_ally_nearby = stop_on_better_ally_nearby

            self.bail = bail

        def reroute_path_to_bail(self, ct: Controller, path_to_bot: path_to.state):
            # bail function
            if self.bail is not None and self.bail(ct, self):
                return

            self.fatigue -= 1

            # leave if fatigued
            if self.fatigue == 0:
                STATE.switch_to(ct, self.exit_state)
                return

            target_bot_idx = -1

            for idx in vision_iter():
                if get_bot_id(ENEMY_BUILDER_BOT[idx]) == self.target_bot_id:
                    target_bot_idx = idx
                    break

            if target_bot_idx == -1:
                print("lost vision")
                STATE.switch_to(ct, self.exit_state)
                return

            path_to_bot.target_idx = target_bot_idx

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        # bail function
        if self.bail is not None and self.bail(ct, self):
            return

        self.fatigue -= 1

        # leave if fatigued
        if self.fatigue == 0:
            STATE.switch_to(ct, self.exit_state)
            return

        target_bot_idx = -1

        for idx in vision_iter():
            if get_bot_id(ENEMY_BUILDER_BOT[idx]) == self.target_bot_id:
                target_bot_idx = idx
                break

        if target_bot_idx == -1:
            print("lost vision")
            STATE.switch_to(ct, self.exit_state)
            return

        for idx in adjacent_iter(target_bot_idx):
            ally_id = get_bot_id(ALLY_BUILDER_BOT[idx])
            if ally_id is not None and ally_id < UNIT_INFO.id:
                print("Another bot is following already")
                STATE.switch_to(ct, self.exit_state)
                return

        target_bot_pos = idx_to_pos(target_bot_idx)

        # TODO, the bail function, using def bail maybe?
        if UNIT_INFO.position.distance_squared(target_bot_pos) > 2:
            path_to_bot = path_to.state(target_bot_idx, 2, self)
            path_to_bot.bail = bind_bail(self.reroute_path_to_bail, path_to_bot)
            STATE.switch_to(ct, path_to_bot)
            return

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state


STATE.register(follow_enemy)
