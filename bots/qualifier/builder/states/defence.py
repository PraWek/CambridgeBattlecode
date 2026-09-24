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
    get_building_info,
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
    print_flow_values,
)
from botlib.constants import POSITION_CACHE, DIRECTION_CACHE

from ..utility.movement import move_to
from ..utility.building import direction_to, destroy, build_if_can_afford

from ..data import BOT_PATHING

from .fundamentals import path_to, move_to, break_tile, heal_target

from .gather_resource import build_resource_line

from .attacks import smart_turret_place

# TODO testing of vvvv state


class reroute_destroy:
    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            target_idx: int,
            exit_state,
            target_id: int = -1,
            fatigue: int = 5,
            bail: Callable[[Controller, heal_target.state], bool] | None = None,
        ):
            self.target_idx = target_idx
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
            self.target_id = get_building_id(get_building_info(self.target_idx))

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        print(f"target: {POSITION_CACHE[self.target_idx]}")

        self.fatigue -= 1

        # bail function
        if self.bail is not None and self.bail(ct, self):
            return

        # leave if target is destroyed
        if self.target_id != get_building_id(get_building_info(self.target_idx)):
            STATE.switch_to(ct, self.exit_state)
            return

        # leave if fatigued
        if self.fatigue == 0:
            STATE.switch_to(ct, self.exit_state)
            return

        # look for sources
        bridge_sources = SINKS[self.target_idx].bridge_sources
        conveyor_sources = SINKS[self.target_idx].conveyor_sources
        harvester_sources = SINKS[self.target_idx].harvester_sources
        foundry_sources = SINKS[self.target_idx].foundry_sources

        soft_sources = bridge_sources | conveyor_sources
        solid_sources = harvester_sources | foundry_sources

        if len(solid_sources) + len(soft_sources) == 0:
            STATE.switch_to(ct, self.exit_state)
            return

        for idx in soft_sources:
            pos = idx_to_pos(idx)
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
                    if ct.can_destroy(pos):
                        ct.destroy(pos)
                    # if the cardinal, build gunner
                    if UNIT_INFO.position.distance_squared(pos) == 1:
                        if ct.can_build(
                            EntityType.GUNNER,
                            pos,
                            direction_to(pos, idx_to_pos(self.target_idx)),
                        ):
                            ct.build_gunner(
                                pos, direction_to(pos, idx_to_pos(self.target_idx))
                            )
                    # if the cardinal, build sentinel
                    else:
                        if ct.can_build(
                            EntityType.SENTINEL,
                            pos,
                            direction_to(pos, idx_to_pos(self.target_idx)),
                        ):
                            ct.build_gunner(
                                pos, direction_to(pos, idx_to_pos(self.target_idx))
                            )

                    return
            # if no flow, pass
            else:
                pass

        for idx in solid_sources:
            pos = idx_to_pos(idx)
            self.fatigue += 1
            # if solid source is out of range, path to
            if UNIT_INFO.position.distance_squared(pos) > 2:
                STATE.switch_to(
                    ct,
                    path_to.state(idx, 2, self),
                )
                return

            # in range, now go destroy it and replace with a barrier
            if UNIT_INFO.position.distance_squared(pos) <= 2:
                if ct.can_destroy(pos):
                    ct.destroy(pos)
                    if ct.can_build(EntityType.BARRIER, pos):
                        ct.build_barrier(pos)

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
                self.target_idx = xy_to_idx(x + dx, y + dy)

            elif type == EntityType.BRIDGE:
                self.target_idx = get_bridge_target_idx(ALLY_BUILDINGS[self.target_idx])

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
                for dir in splitter_dirs:
                    dir = get_building_direction(ALLY_BUILDINGS[self.target_idx])
                    dx, dy = dir.delta()
                    x, y = pos.x, pos.y
                    self.target_idx = xy_to_idx(x + dx, y + dy)
            else:
                break

        print(f"repairing new target {POSITION_CACHE[self.target_idx]}")

        # forward pass responses
        if self.forward:
            print(f"im here")

            # if it routes to enemy
            if BITBOARD_ENEMY[self.target_idx] != 0:
                if BITBOARD_ENEMY[self.target_idx] != 0:
                    self.fixed = True
                    STATE.switch_to(
                        ct,
                        reroute_destroy,
                        reroute_destroy.state(self.target_idx, self.exit_state),
                    )
                    return

            if (
                BITBOARD_ALLY[self.target_idx] != 0
                and (BITBOARD_ALLY[self.target_idx] & 0b0001_0111_1000_0100) != 0
            ):
                # core, foundry, NOT turrets, all conveyors types count as valid sink
                # set mode to backwards pass
                print(f"im here 3")
                self.forward = False
                self.fixed = True
                return

            # TODO i think this indent got changed by liveshare, idk, i put it where i think it should be
            else:
                print(f"im here 2")

                # if out of range, move into range, then build resource line home
                if UNIT_INFO.position.distance_squared(idx_to_pos(self.target_idx)) > 2:
                    STATE.switch_to(
                        ct,
                        path_to.state(self.target_idx, 2, self),
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
                        ),
                    )
                    return
            # TODO debug
            if self.fixed:
                print(f"fixed!")
                STATE.switch_to(ct, self.exit_state)
                return
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


class follow_enemy:
    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            target_bot_id: int,
            exit_state,
            target_id: int = -1,
            fatigue: int = 5,
            bail: Callable[[Controller, heal_target.state], bool] | None = None,
        ):
            self.target_bot_id = target_bot_id
            self.exit_state = exit_state

            self.target_id = target_id
            self.fatigue = fatigue

            self.bail = bail

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        self.fatigue -= 1

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

        target_bot_pos = idx_to_pos(target_bot_idx)

        # TODO, the bail function, using def bail maybe?
        if UNIT_INFO.position.distance_squared(target_bot_pos) > 2:
            STATE.switch_to(
                ct,
                path_to.state(target_bot_idx, 2, self, bail=bind_bail(self.bail, self)),
            )
            return

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state


STATE.register(follow_enemy)
