from __future__ import annotations

from typing import Callable

from cambc import Controller, Direction, EntityType, Position

from botlib import State
from .. import STATE

from botlib import (
    MAP_INFO,
    UNIT_INFO,
    BITBOARD_TARGETS,
    BITBOARD_GRID,
    BITBOARD_ALLY,
    BITBOARD_ENEMY,
    BITBOARD_ENV,
    ALLY_BUILDINGS,
    HAZARDS,
    SINKS,
    RESOURCE_FLOW,
)
from botlib import (
    best_enemy_core_idx,
    xy_to_idx,
    in_bounds,
    vision_iter,
    get_bridge_target_idx,
    not_in_bounds,
)

from botlib.constants import (
    POSITION_CACHE,
    DIRECTION_CACHE,
    DIRECTION_DELTAS,
    CARDINAL_DIRECTION_DELTAS,
)

from ..utility.movement import move_to, safe_move
from ..utility.building import build_if_can_afford, can_afford, build_if_can, destroy

from ..data import (
    BOT_PATHING,
    DISTANCE_FIELD,
    VISION_DELTAS,
    BRIDGE_COOP_COOLDOWN,
    ORES_COMPLETED,
)

import random

# ------------- STATES -------------

from .fundamentals import path_to, fog_max, break_tile, BREAK_TILE_COOLDOWN
from .gather_resource import build_resource_line, check_ore_groups

from ..data import CORE_CANDIDATE_LOCATIONS

# ------------- STATES -------------


from .attacks import attack_harvester


def rush_bail(ct: Controller, self: fog_max.state):
    # Check core positions
    core_x, core_y = POSITION_CACHE[best_enemy_core_idx()]
    for dx, dy in CORE_CANDIDATE_LOCATIONS:
        nx = core_x + dx
        ny = core_y + dy
        idx = xy_to_idx(nx, ny)
        if BITBOARD_ENEMY[idx] != 0 and STATE.tick < BREAK_TILE_COOLDOWN[idx]:
            continue
        if (BITBOARD_ENEMY[idx] & 0b0010_0101_1000_0000) == 0:
            continue
        if (BITBOARD_ALLY[idx] & 0b0010) != 0:
            continue
        sink_info = SINKS[idx]
        ti_flow = RESOURCE_FLOW[0][idx]
        ref_flow = RESOURCE_FLOW[2][idx]
        num_sources = (
            sink_info.num_bridge_sources
            + sink_info.num_conveyor_sources
            + sink_info.num_harvester_sources
        )
        if num_sources > 0 and (
            (ti_flow.get_flow() > 0 or ti_flow.get_stall() > 0)
            or (ref_flow.get_flow() > 0 or ref_flow.get_stall() > 0)
        ):
            if can_afford(ct, EntityType.SENTINEL):
                STATE.switch_to(ct, sprint_1.state(idx, self))
            else:
                STATE.switch_to(ct, break_tile.state(idx, self))
            return True

    # Look for harvesters to attack
    for idx in vision_iter():
        if (BITBOARD_ENEMY[idx] & 0b1000_0000_0000) != 0 and (BITBOARD_ENV[idx] & 0b0100) != 0:
            # Harvester

            x, y = POSITION_CACHE[idx]

            # if all cardinals are full, leave

            spot_available = False

            for dx, dy in CARDINAL_DIRECTION_DELTAS:
                nx = x + dx
                ny = y + dy
                if not_in_bounds(nx, ny):
                    continue
                neighbour_idx = ny * MAP_INFO.width + nx
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

            if spot_available:
                STATE.switch_to(
                    ct,
                    attack_harvester.state(idx, self),
                )
                return True

    return False


class sprint_1:
    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            target_idx: int,
            exit_state,
            bail: Callable[[Controller, sprint_1.state], bool] | None = None,
        ):
            self.exit_state = exit_state
            self.bail = bail
            self.target_idx = target_idx
            self.fatigue = 50

        def launcher_break_bail(self, ct, state: break_tile.state):
            if (BITBOARD_ENEMY[self.target_idx] & 0b0010) == 0:
                STATE.switch_to(ct, state.exit_state)
                return True

        def path_to_bail(self, ct, state: path_to.state):
            if (BITBOARD_ALLY[self.target_idx] & 0b0010) != 0 or (
                BITBOARD_ENEMY[self.target_idx] & 0b0010
            ) != 0:
                STATE.switch_to(ct, state.exit_state)
                return True

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        print(f"sprint1 {POSITION_CACHE[self.target_idx]}")

        if self.bail is not None and self.bail(ct, self):
            return

        if BREAK_TILE_COOLDOWN[self.target_idx] > STATE.tick:
            print("on break cooldown")
            STATE.switch_to(ct, self.exit_state)
            return

        if (BITBOARD_ALLY[self.target_idx] & 0b0010) != 0:
            print("another ally bot on it (potentially attacking it)")
            STATE.switch_to(ct, self.exit_state)
            return

        if UNIT_INFO.position.distance_squared(POSITION_CACHE[self.target_idx]) > 2:
            STATE.switch_to(ct, path_to.state(self.target_idx, 2, self, bail=self.path_to_bail))
            return

        if (BITBOARD_ENEMY[self.target_idx] & 0b0010) != 0:
            # Bot on it, we have to launcher
            x, y = UNIT_INFO.position
            has_enemy_tile = []
            has_ally_tile = []
            empty = []
            for dx, dy in DIRECTION_DELTAS:
                nx = x + dx
                ny = y + dy
                idx = xy_to_idx(nx, ny)
                if (BITBOARD_ENEMY[idx] & 0b0010) != 0 or (BITBOARD_ALLY[idx] & 0b0010) != 0:
                    continue
                if BITBOARD_ENEMY[idx] == 0 and BITBOARD_ALLY[idx] == 0:
                    empty.append(idx)
                    continue
                if (BITBOARD_ENEMY[idx] & 0b0010_0101_1000_0000) != 0 and BREAK_TILE_COOLDOWN[
                    idx
                ] < STATE.tick:
                    has_enemy_tile.append(idx)
                elif (BITBOARD_ALLY[idx] & 0b0111_1111_1000_0000) != 0:
                    has_ally_tile.append(idx)

            if len(empty) > 0:
                if not build_if_can_afford(ct, EntityType.LAUNCHER, empty[0]):
                    return
            elif len(has_ally_tile) > 0:
                if can_afford(ct, EntityType.LAUNCHER):
                    destroy(ct, has_ally_tile[0])
                    build_if_can(ct, EntityType.LAUNCHER, has_ally_tile[0])
                else:
                    return
            elif len(has_enemy_tile) > 0:
                STATE.switch_to(ct, break_tile.state(idx, self, self.launcher_break_bail))
                return

        if (BITBOARD_ENEMY[self.target_idx] & 0b0010_0101_1000_0000) != 0:
            if BREAK_TILE_COOLDOWN[self.target_idx] < STATE.tick:
                STATE.switch_to(ct, break_tile.state(self.target_idx, self))
                return
            else:
                STATE.switch_to(ct, self.exit_state)
                return

        if not can_afford(ct, EntityType.SENTINEL) and not can_afford(ct, EntityType.BARRIER):
            self.fatigue -= 1
            if self.fatigue <= 0:
                STATE.switch_to(ct, self.exit_state)
                return
            return

        if UNIT_INFO.position_idx == self.target_idx:
            move_to(ct, self.target_idx, forwards=False)

        type = EntityType.SENTINEL
        if (
            POSITION_CACHE[self.target_idx].distance_squared(POSITION_CACHE[best_enemy_core_idx()])
            < 13
        ):
            type = EntityType.GUNNER

        if not build_if_can(
            ct,
            type,
            self.target_idx,
            POSITION_CACHE[self.target_idx].direction_to(POSITION_CACHE[best_enemy_core_idx()]),
        ):
            build_if_can(ct, EntityType.BARRIER, self.target_idx)

        STATE.switch_to(ct, self.exit_state)
        return

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state


STATE.register(sprint_1)
