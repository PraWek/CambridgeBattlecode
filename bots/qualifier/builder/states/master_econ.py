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
    ENEMY_BUILDINGS,
    HAZARDS,
)
from botlib import (
    best_enemy_core_idx,
    xy_to_idx,
    in_bounds,
    vision_iter,
    get_bridge_target_idx,
    get_building_type,
    CT_GET_CURRENT_ROUND,
)

from botlib.constants import (
    POSITION_CACHE,
    DIRECTION_CACHE,
    DIRECTION_DELTAS,
    CARDINAL_DIRECTION_DELTAS,
)

from ..utility.movement import move_to, safe_move
from ..utility.building import build_if_can_afford, destroy

from ..data import (
    BOT_PATHING,
    DISTANCE_FIELD,
    VISION_DELTAS,
    BRIDGE_COOP_COOLDOWN,
    ORES_COMPLETED,
    TITANIUM_HARVESTERS_PLACED,
)

import random

# ------------- STATES -------------

from .fundamentals import path_to, fog_max, break_tile, BREAK_TILE_COOLDOWN
from .gather_resource import build_resource_line, check_ore_groups, path_place_harvester


def econ_bail(ct: Controller, top_level, self: fog_max.state):
    # Check for nearby ore, if any switch to gather_resource

    if len(TITANIUM_HARVESTERS_PLACED) > 0:
        top_level.local = None

        # NOTE, when do you enable axionite
        ax_round = 500

        for idx in vision_iter():
            # Produce ref ax
            if (
                (BITBOARD_ENV[idx] & 0b1000) != 0
                and not ORES_COMPLETED[idx]
                # TODO NOT COMPATIBLE WITH DEFENSIVE CONVEYORS NEEDS CHANGE ALSO DO IN PATH TO ORE
                and (BITBOARD_ALLY[idx] & 0b0001_1111_1011_1000) == 0
                and (BITBOARD_ENEMY[idx] & 0b0101_1010_0111_1000) == 0
                and (HAZARDS[3][idx] == 0)
                and (HAZARDS[4][idx] == 0)
                # NOTE TODO, BAND AID FIX
                and ct.get_current_round() >= ax_round
            ):
                is_defended = False
                x, y = POSITION_CACHE[idx]
                for dx, dy in CARDINAL_DIRECTION_DELTAS:
                    if in_bounds(x + dx, y + dy):
                        neighbour_idx = (y + dy) * MAP_INFO.width + (x + dx)
                        if (BITBOARD_ENEMY[neighbour_idx] & 0b0011_1100) != 0:
                            is_defended = True
                            break
                if not is_defended:
                    print(f"candidate ref ax: {POSITION_CACHE[idx]}")
                    # Find closest titanium harvester
                    DISTANCE_FIELD.solver.use_buildings = False
                    DISTANCE_FIELD.solve(idx, stop_mask=BITBOARD_GRID.ally[11])
                    DISTANCE_FIELD.solver.use_buildings = True
                    if DISTANCE_FIELD.ready():
                        best_ti_idx = -1
                        best_dist = 0
                        for titanium_idx in TITANIUM_HARVESTERS_PLACED:
                            dist = DISTANCE_FIELD.dist(titanium_idx)
                            if dist is None:
                                continue
                            if best_ti_idx == -1 or dist < best_dist:
                                best_ti_idx = titanium_idx
                                best_dist = dist

                        if best_ti_idx != -1:
                            STATE.switch_to(
                                ct,
                                produce_ref_ax.state(idx, best_ti_idx, self),
                            )
                            return True

    DISTANCE_FIELD.solve(UNIT_INFO.position_idx, stop_cost=10)
    for idx in vision_iter():
        if (BITBOARD_ENV[idx] & 0b0100) != 0 and get_building_type(
            ALLY_BUILDINGS[idx]
        ) == EntityType.HARVESTER:
            TITANIUM_HARVESTERS_PLACED.add(idx)

        if (
            (BITBOARD_ENV[idx] & 0b1100) != 0
            and (
                get_building_type(ALLY_BUILDINGS[idx]) != EntityType.HARVESTER
                and get_building_type(ENEMY_BUILDINGS[idx]) != EntityType.HARVESTER
            )
            and ORES_COMPLETED[idx] > 0
        ):
            ORES_COMPLETED[idx] -= 1

        # Bridge coop
        if (
            BRIDGE_COOP_COOLDOWN[idx] < STATE.tick
            and DISTANCE_FIELD.ready()
            and (BITBOARD_ALLY[idx] & 0b0100_0000_0000) != 0
        ):
            if (
                BITBOARD_ALLY[bridge_target := get_bridge_target_idx(ALLY_BUILDINGS[idx])]
                & 0b0111_1000_0110
            ) == 0:
                dist = DISTANCE_FIELD.dist(bridge_target)
                if dist is not None:
                    print(
                        f"bridge coop: {POSITION_CACHE[bridge_target]}"
                    )  # TODO gets stuck if cannot build resource line
                    BRIDGE_COOP_COOLDOWN[idx] = STATE.tick + 20  # 20 round cooldown
                    STATE.switch_to(
                        ct,
                        build_resource_line.state(
                            bridge_target,
                            UNIT_INFO.ally_core_idx,
                            self,
                            skip_first=False,
                            check_flow=False,
                        ),
                    )
                    return True

        # Look for ore
        if (
            (BITBOARD_ENV[idx] & 0b0100) != 0
            and not ORES_COMPLETED[idx]
            # TODO NOT COMPATIBLE WITH DEFENSIVE CONVEYORS NEEDS CHANGE ALSO DO IN PATHTOORE
            and (BITBOARD_ALLY[idx] & 0b0001_1111_1011_1010) == 0
            and (BITBOARD_ENEMY[idx] & 0b0101_1010_0111_1000) == 0
            and (HAZARDS[3][idx] == 0)
            and (HAZARDS[4][idx] == 0)
        ):
            if BREAK_TILE_COOLDOWN[idx] > STATE.tick:
                continue

            is_defended = False
            x, y = POSITION_CACHE[idx]
            for dx, dy in CARDINAL_DIRECTION_DELTAS:
                if in_bounds(x + dx, y + dy):
                    neighbour_idx = (y + dy) * MAP_INFO.width + (x + dx)
                    if (BITBOARD_ENEMY[neighbour_idx] & 0b0011_1100) != 0:
                        is_defended = True
                        break
            if not is_defended:
                path_to_ore = path_to.state(idx, 2, None)  # TODO think about dist to walk
                check_ore_group = check_ore_groups.state(
                    0b0100,
                    idx,
                    UNIT_INFO.ally_core_idx,
                    self,
                    origin_path_to=path_to_ore,
                )
                path_to_ore.exit_state = check_ore_group
                STATE.switch_to(ct, path_to_ore)
                return True

    return False


class produce_ref_ax:
    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            ax_ore_idx,
            ti_harvester_idx,
            exit_state,
            bail: Callable[[Controller, produce_ref_ax.state], bool] | None = None,
        ):
            self.ax_ore_idx = ax_ore_idx
            self.ti_harvester_idx = ti_harvester_idx
            self.exit_state = exit_state
            self.bail = bail

            self.path_task: path_to.state | None = None

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

        DISTANCE_FIELD.solve(self.ti_harvester_idx, stop_cost=2)
        if not DISTANCE_FIELD.ready():
            return

        best_target_idx = -1
        best_dist = 0

        x, y = POSITION_CACHE[self.ti_harvester_idx]
        for dx, dy in CARDINAL_DIRECTION_DELTAS:
            nx = x + dx
            ny = y + dy
            idx = xy_to_idx(nx, ny)
            if (BITBOARD_ALLY[idx] & 0b1010_0111_1000_0000) != 0:
                dist = DISTANCE_FIELD.dist(idx)
                if dist is None:
                    continue
                if best_target_idx == -1 or dist < best_dist:
                    best_target_idx = idx
                    best_dist = dist

        if best_target_idx == -1:
            # just exit on no valid targets
            print("no valid targets")
            ORES_COMPLETED[self.ax_ore_idx] = 50
            STATE.switch_to(ct, self.exit_state)
            return

        ax_ore_pos = POSITION_CACHE[self.ax_ore_idx]
        print(f"pathing to {ax_ore_pos}")
        if UNIT_INFO.position.distance_squared(ax_ore_pos) > 2:
            self.path_task = path_to.state(self.ax_ore_idx, 2, self)
            STATE.switch_to(ct, self.path_task)
            return

        if self.path_task is not None and not self.path_task.path_to_succeeded:
            print("unreachable ore")
            STATE.switch_to(ct, self.exit_state)
            return

        if (BITBOARD_ALLY[self.ax_ore_idx] & 0b0001_0000_0000_0000) != 0:
            STATE.switch_to(ct, self.exit_state)
            return

        if (BITBOARD_ALLY[self.ax_ore_idx] & 0b0110_0111_1111_1000) != 0:
            destroy(ct, self.ax_ore_idx)

        if (BITBOARD_ENEMY[self.ax_ore_idx] & 0b0010_0101_1000_0000) != 0:
            STATE.switch_to(ct, break_tile.state(self.ax_ore_idx, self))
            return

        if UNIT_INFO.position_idx == self.ax_ore_idx:
            move_to(ct, self.ax_ore_idx, forwards=False)

        if not build_if_can_afford(ct, EntityType.HARVESTER, ax_ore_pos):
            print("cant afford yet")
            return

        ORES_COMPLETED[self.ax_ore_idx] = 50

        ax_to_ti = build_resource_line.state(
            self.ax_ore_idx,
            best_target_idx,
            None,
            treat_ally_core_as_conveyor=False,
            merge=False,
            inflate_ti_harvesters=True,
            end_on_bridge=True,
        )
        foundry_to_core = build_resource_line.state(
            best_target_idx,
            UNIT_INFO.ally_core_idx,
            self.exit_state,
            continue_lines=[ax_to_ti],
        )
        ax_to_ti.exit_state = State(
            path_place_harvester,
            best_target_idx,
            0b1000,
            foundry_to_core,
            do_foundry=True,
        )

        STATE.switch_to(ct, ax_to_ti)

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state


STATE.register(produce_ref_ax)
