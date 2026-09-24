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
    ENEMY_BUILDER_BOT,
    HAZARDS,
    RESOURCE_FLOW,
)
from botlib import (
    best_enemy_core_idx,
    xy_to_idx,
    idx_to_pos,
    in_bounds,
    vision_iter,
    get_bridge_target_idx,
    get_building_type,
    get_building_maxhp,
    get_building_hp,
    get_building_direction,
    get_bot_id,
    not_in_bounds,
    CT_GET_CURRENT_ROUND,
    RESOURCE_FLOW,
    SINKS,
    bind_bail,
)

from botlib.constants import (
    POSITION_CACHE,
    DIRECTION_CACHE,
    DIRECTION_DELTAS,
    CARDINAL_DIRECTION_DELTAS,
    CONVEYOR_DIRECTIONS,
    SPLITTER_DIRECTIONS,
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

from .fundamentals import path_to, fog_max, break_tile, heal_target
from .gather_resource import build_resource_line, check_ore_groups, path_place_harvester
from .defence import reroute_destroy, repair_throughput, follow_enemy
from .master_def import def_bail


def generic_bail(
    ct: Controller,
    top_level,
    self: fog_max.state,
    check_econ: bool = True,
    check_repair_lines: bool = True,
    heal_everything: bool = True,
):
    print("generic bail")

    harvester_idxs = []
    under_bot_attack_idxs = []
    other_damaged_idxs = []
    ally_bots_idxs = []
    # doesnt include launchers
    enemy_turret_idxs = []
    ally_turret_idxs = []

    # includes launchers
    enemy_lb_idxs = []

    ally_ammo_flow = []
    ally_stall = []

    for idx in vision_iter():
        # find damaged or under bot attack tiles
        target_hp = get_building_hp(ALLY_BUILDINGS[idx])
        max_hp = get_building_maxhp(ALLY_BUILDINGS[idx])

        # TODO, bandaid fix for line 113 elif (BITBOARD_ALLY[idx] != 0) and target_hp < max_hp:
        if target_hp is None:
            target_hp = -1
        if max_hp is None:
            max_hp = -1

        if (
            (BITBOARD_ENEMY[idx] & 0b0010) != 0
            and (BITBOARD_ALLY[idx] != 0)
            and (target_hp < max_hp)
        ):
            # if enemy bot on ally tile, and damaged
            under_bot_attack_idxs.append(idx)
        elif (BITBOARD_ALLY[idx] != 0) and target_hp < max_hp:
            other_damaged_idxs.append(idx)

        # find ally bots, and not yourself
        if (BITBOARD_ALLY[idx] & 0b0010) != 0 and UNIT_INFO.position_idx != idx:
            ally_bots_idxs.append(idx)

        # find enemy turrets, gunner senti breach
        if (BITBOARD_ENEMY[idx] & 0b0011_1000) != 0:
            enemy_turret_idxs.append(idx)

        # find ally turrets, gunner senti breach
        if (BITBOARD_ALLY[idx] & 0b0011_1000) != 0:
            ally_turret_idxs.append(idx)

        # find enemy nuisances, launcher barrier
        if (BITBOARD_ENEMY[idx] & 0b0100_0000_0100_0000) != 0:
            enemy_lb_idxs.append(idx)
            # TODO FOURTH priority, heal random damaged, this could lead to the bot chasing nonsense, and not exploring enough

        # find flow, and ensure we own it
        if (
            RESOURCE_FLOW[0][idx].get_flow() > 0
            or RESOURCE_FLOW[0][idx].get_stall() > 0
            or RESOURCE_FLOW[2][idx].get_flow() > 0
            or RESOURCE_FLOW[2][idx].get_stall() > 0
        ) and (BITBOARD_ALLY[idx] != 0):
            ally_ammo_flow.append(idx)

        if (BITBOARD_ALLY[idx] & 0b1000_0000_0000) != 0 and (BITBOARD_ENV[idx] & 0b0100) != 0:
            # Harvester NOTE only cares about titanium harvesters

            # look for neighbours we own, if we do then add to ally_ammo_flow
            harvester_idxs.append(idx)

            x, y = POSITION_CACHE[idx]

            for dx, dy in CARDINAL_DIRECTION_DELTAS:
                nx = x + dx
                ny = y + dy
                if not_in_bounds(nx, ny):
                    continue
                neighbour_idx = ny * MAP_INFO.width + nx
                # if we own the tile, then we can use it for ammo flow
                if (BITBOARD_ALLY[neighbour_idx] & 0b1111_1111_1111_1100) != 0:
                    # everything except builder bot
                    ally_ammo_flow.append(idx)

    # NOTE, comment this out for stability, UNTESTED
    # TODO, test
    # if enemy turret, try and reroute destroy it
    for idx in enemy_turret_idxs:
        # if it has a source you own, reroute destroy
        bridge_sources = SINKS[idx].bridge_sources
        conveyor_sources = SINKS[idx].conveyor_sources
        harvester_sources = SINKS[idx].harvester_sources
        foundry_sources = SINKS[idx].foundry_sources

        soft_sources = bridge_sources | conveyor_sources
        solid_sources = harvester_sources | foundry_sources

        sources = soft_sources | solid_sources

        own_a_source = False

        for source_idx in sources:
            if BITBOARD_ALLY[source_idx] != 0:
                own_a_source = True
                break

        if len(soft_sources | solid_sources) > 0 and own_a_source:
            STATE.switch_to(ct, reroute_destroy.state(idx, self))
            return True

    # if we are the closest bot to the tile being attacked, go heal it

    for idx in under_bot_attack_idxs:
        my_d2 = Position.distance_squared(UNIT_INFO.position, idx_to_pos(idx))
        best_d2 = 9999
        closest_ally_idx = None

        enemy_bot_id = get_bot_id(ENEMY_BUILDER_BOT[idx])

        for bot_idx in ally_bots_idxs:
            d2 = Position.distance_squared(idx_to_pos(bot_idx), idx_to_pos(idx))
            if d2 < best_d2:
                best_d2 = d2
                closest_ally_idx = bot_idx

        is_best_candidate = False
        if my_d2 < best_d2:
            is_best_candidate = True
        elif my_d2 == best_d2 and closest_ally_idx is not None:
            if UNIT_INFO.position_idx < closest_ally_idx:
                is_best_candidate = True

        if is_best_candidate:
            # STATE.switch_to(ct, heal_target.state(idx, self, fatigue=4))
            STATE.switch_to(
                ct,
                heal_target.state(
                    idx,
                    follow_enemy.state(
                        enemy_bot_id,
                        self,
                        fatigue=10,
                        bail=bind_bail(
                            generic_bail,
                            top_level,
                            self,
                            check_repair_lines=False,
                            check_econ=False,
                        ),
                    ),
                    fatigue=4,
                ),
            )
            return True

    # TODO START OF REPAIR LOGIC --------------------------

    if check_repair_lines:
        for idx in vision_iter():
            stall_thr = 0.5
            splitter_target_idxs = []
            target_idx = -1
            # if conveyor or bridge type and stalling, check the target tile
            if (BITBOARD_ALLY[idx] & 0b0111_1000_0000) != 0 and (
                RESOURCE_FLOW[0][idx].get_stall() > stall_thr
                or RESOURCE_FLOW[2][idx].get_stall() > stall_thr
            ):
                # if bridge, set target idx to bridge target
                if (BITBOARD_ALLY[idx] & 0b0100_0000_0000) != 0:
                    target_idx = get_bridge_target_idx(ALLY_BUILDINGS[idx])
                # must be a conveyor, set target idx, to direction thingy
                else:
                    x, y = idx_to_pos(idx)
                    direction = (ALLY_BUILDINGS[idx] >> 4) & 0xFFF
                    if (BITBOARD_ALLY[idx] & 0b0010_1000_0000) != 0:
                        dx, dy = CONVEYOR_DIRECTIONS[direction]
                        target_idx = xy_to_idx(x + dx, y + dy)
                    else:
                        for dx, dy in SPLITTER_DIRECTIONS[direction]:
                            splitter_target_idxs.append(xy_to_idx(x + dx, y + dy))
                if target_idx != -1:
                    print(f"checking if conveyor points at {idx_to_pos(target_idx)}")
                    # check if it is a valid resource sink, i.e. converyor splitter bridge core foundry
                    if (BITBOARD_ALLY[target_idx] & 0b0001_0111_1000_0100) == 0:
                        # TODO trigger repair
                        STATE.switch_to(
                            ct,
                            repair_throughput.state(
                                idx,
                                self,
                                bail=bind_bail(
                                    generic_bail,
                                    top_level,
                                    self,
                                    check_repair_lines=False,
                                    check_econ=False,
                                ),
                            ),
                        )
                        return True
                if len(splitter_target_idxs):
                    splitter_good = False
                    for splitter_target_idx in splitter_target_idxs:
                        if (BITBOARD_ALLY[splitter_target_idx] & 0b0001_0111_1000_0100) != 0:
                            splitter_good = True
                    if not splitter_good:
                        # TODO trigger repair
                        STATE.switch_to(
                            ct,
                            repair_throughput.state(
                                idx,
                                self,
                                bail=bind_bail(
                                    generic_bail,
                                    top_level,
                                    self,
                                    check_repair_lines=False,
                                    check_econ=False,
                                ),
                            ),
                        )
                        return True

    # START OF ECONOMY LOGIC --------------------------
    # Check for nearby ore, if any switch to gather_resource

    if len(TITANIUM_HARVESTERS_PLACED) > 0 and top_level.econ and check_econ:
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
                                produce_ref_ax.state(
                                    idx,
                                    best_ti_idx,
                                    self,
                                    bail=bind_bail(generic_bail, top_level, self, check_econ=False),
                                ),
                            )
                            return True

    print(f"check eco = {top_level.econ}")

    if top_level.econ and check_econ:
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
                                bail=bind_bail(generic_bail, top_level, self, check_econ=False),
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
                    path_to_ore.bail = bind_bail(
                        closer_ore_bail,
                        path_to_ore,
                        bind_bail(generic_bail, top_level, self, check_econ=False),
                    )
                    check_ore_group = check_ore_groups.state(
                        0b0100,
                        idx,
                        UNIT_INFO.ally_core_idx,
                        self,
                        origin_path_to=path_to_ore,
                        resource_bail=bind_bail(generic_bail, top_level, self, check_econ=False),
                    )
                    path_to_ore.exit_state = check_ore_group
                    STATE.switch_to(ct, path_to_ore)
                    return True

    if heal_everything:
        # TODO Heal leftovers?
        for idx in other_damaged_idxs:
            STATE.switch_to(ct, heal_target.state(idx, self, fatigue=4))
            return True

    return False


def closer_ore_bail(ct: Controller, self: path_to.state, def_bail):
    if def_bail(ct, None):
        return

    DISTANCE_FIELD.solve(UNIT_INFO.position_idx, stop_cost=10)
    if not DISTANCE_FIELD.ready():
        return

    dist = DISTANCE_FIELD.dist(self.target_idx)
    for idx in vision_iter():
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
            is_defended = False
            x, y = POSITION_CACHE[idx]
            for dx, dy in CARDINAL_DIRECTION_DELTAS:
                if in_bounds(x + dx, y + dy):
                    neighbour_idx = (y + dy) * MAP_INFO.width + (x + dx)
                    if (BITBOARD_ENEMY[neighbour_idx] & 0b0011_1100) != 0:
                        is_defended = True
                        break
            if not is_defended:
                dist2 = DISTANCE_FIELD.dist(idx)
                if dist is None or (dist2 is not None and dist2 < dist):
                    print(f"reroute to {POSITION_CACHE[idx]}")
                    self.target_idx = idx
                    self.exit_state.target_ore_idx = idx
                    dist = dist2
                    return False

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
            if not_in_bounds(nx, ny):
                continue
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
