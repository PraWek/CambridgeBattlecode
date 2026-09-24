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
    get_flow_rate,
    get_stall_rate,
    sort_closest_idxs,
    distance_sq_to,
    direction_to,
    pos_to_idx,
    can_afford,
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

from .fundamentals import path_to, fog_max, break_tile, BREAK_TILE_COOLDOWN
from .gather_resource import build_resource_line, check_ore_groups

from .fundamentals import path_to, fog_max, break_tile, heal_target
from .gather_resource import build_resource_line, check_ore_groups, path_place_harvester
from .defence import reroute_destroy, repair_throughput, follow_enemy
from .master_def import def_bail
from .attacks import snipe_replace, place_turret, attack_harvester


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

    sources = soft_sources | solid_sources

    own_a_source = False

    for source_idx in sources:
        if BITBOARD_ALLY[source_idx] != 0:
            own_a_source = True
            break

    if len(soft_sources) > 0:
        source = next(iter(soft_sources))
        if own_a_source and HAZARDS[6][source] == 0:
            print(f"submitting: {POSITION_CACHE[idx]}")
            return idx
        else:
            return check_turret_sinks(ct, next(iter(soft_sources)), max_depth - 1)

    if len(solid_sources) > 0 and own_a_source:
        print(f"solid: {POSITION_CACHE[idx]}")
        return idx


def combat_bail(
    ct: Controller,
    self: fog_max.state,
    check_harvester_attack: bool = True,
    check_core_attack: bool = True,
    check_repair_lines: bool = True,
    check_heal: bool = True,
    check_snipe: bool = True,
    check_reroute_repair: bool = True,
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
    ally_ammo_flow_only = []
    harvester_neighbour_idxs = []

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

        # find flow, and NOT stall
        if (
            RESOURCE_FLOW[0][idx].get_flow() > 0
            # or RESOURCE_FLOW[0][idx].get_stall() > 0
            or RESOURCE_FLOW[2][idx].get_flow() > 0
            # or RESOURCE_FLOW[2][idx].get_stall() > 0
        ) and (BITBOARD_ALLY[idx] != 0):
            ally_ammo_flow_only.append(idx)

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
                harvester_neighbour_idxs.append(neighbour_idx)
                # if we own the tile, then we can use it for ammo flow
                if (BITBOARD_ALLY[neighbour_idx] & 0b1111_1111_1111_1100) != 0:
                    # everything except builder bot
                    ally_ammo_flow.append(neighbour_idx)

    if check_reroute_repair:
        # if enemy turret, try and reroute destroy it
        # TODO test, if you cannot reroute destroy, use snipe scan and place a turret that way
        print(f"{[idx_to_pos(idx) for idx in enemy_turret_idxs]}")
        for idx in enemy_turret_idxs:
            ally_idx = check_turret_sinks(ct, idx, 3)
            if break_tile.is_on_cooldown(ally_idx) or path_to.is_on_cooldown(ally_idx):
                continue
            if ally_idx is not None:
                spot_available = False
                x, y = POSITION_CACHE[ally_idx]

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
                        reroute_destroy.state(
                            ally_idx,
                            self,
                            bail=bind_bail(combat_bail, self, check_reroute_repair=False),
                        ),
                    )
                    return True
    else:
        return False

    # if we are the closest bot to the tile being attacked, go heal it

    if check_heal:
        print(f"{under_bot_attack_idxs}")
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

            print(f"- {is_best_candidate} {idx_to_pos(idx)} {my_d2 == best_d2} {closest_ally_idx}")

            if is_best_candidate:
                # STATE.switch_to(ct, heal_target.state(idx, self, fatigue=4))
                STATE.switch_to(
                    ct,
                    heal_target.state(
                        idx,
                        follow_enemy.state(
                            enemy_bot_id,
                            self,
                            bail=bind_bail(combat_bail, self, check_heal),
                        ),
                        fatigue=0,
                    ),
                )
                return True
    else:
        return False

    # sniper launchers

    if check_snipe:
        for idx in enemy_lb_idxs:
            # TODO snipe replace defence
            # enemy_turret_idxs.extend(enemy_lb_idxs)

            # # if no source, do a gunner scan from the enemy turret
            # # TODO add placing harvesters to this, with a ti condition, same with the sentinel part
            # for idx in enemy_turret_idxs:
            #     print("SNIPE: entered, saw enemy turret without allied source")

            gunner_ready = False
            ammo_idxs = []
            x, y = idx_to_pos(idx).x, idx_to_pos(idx).y

            ## TODO REIMPLEMENT GBUNNER
            # # scan gunner
            # # cast the rays, list all the possible targets
            # for dir in DIRECTION_CACHE:
            #     # print(f"dir: {dir}")
            #     dx, dy = dir.delta()
            #     # print(f"dx, dy: {dx}, {dy}")
            #     count = 4
            #     if (
            #         (dir == Direction.NORTHEAST)
            #         or (dir == Direction.NORTHWEST)
            #         or (dir == Direction.SOUTHEAST)
            #         or (dir == Direction.SOUTHWEST)
            #     ):
            #         count = 3
            #     # print(f"count: {count}")
            #     for i in range(1, count):
            #         nx = x + i * dx
            #         ny = y + i * dy
            #         # print(f"nx, ny: {nx}, {ny}")
            #         target_idx = xy_to_idx(nx, ny)
            #         # if out of bounds, break
            #         if not_in_bounds(nx, ny) or not ct.is_in_vision(idx_to_pos(target_idx)):
            #             break
            #         # if allied gunner, with a source, bail entirely
            #         if (BITBOARD_ALLY[target_idx] & 0b1000) != 0 and SINKS[
            #             target_idx
            #         ].get_total_sources() > 0:
            #             print("found defensive gunner")
            #             gunner_ready = True
            #             break

            #         # if allied tile (not marker or road) or wall, stop checking this direction
            #         if (BITBOARD_ALLY[target_idx] & 0b0101_1111_1111_1111 != 0) or (
            #             BITBOARD_ENV[target_idx] & 0b0010 == 0
            #         ):
            #             # TODO NOTE, check the fn is running, no highlight?
            #             # check in here if its TODO neighbouring a harvester and then
            #             if BITBOARD_ALLY[target_idx] != 0 and (
            #                 RESOURCE_FLOW[0][target_idx].get_flow() > 0
            #                 # or RESOURCE_FLOW[0][target_idx].get_stall() > 0
            #                 or RESOURCE_FLOW[2][target_idx].get_flow() > 0
            #                 # or RESOURCE_FLOW[2][target_idx].get_stall() > 0
            #             ):
            #                 ammo_idxs.append(target_idx)
            #             break

            #         # if enemy, continue, and check the next tile
            #         # TODO if enemy bot is on our tile can this break scanning, i think above catches them
            #         if BITBOARD_ENEMY[target_idx != 0]:
            #             continue

            #         # if empty, continue, and check the next tile
            #         if (BITBOARD_ENEMY[target_idx] == 0) and (
            #             BITBOARD_ALLY[target_idx] == 0
            #         ):
            #             continue

            #         # if marker, continue
            #         if (BITBOARD_ENEMY[target_idx] & 0b1000_0000_0000_0000 != 0) or (
            #             BITBOARD_ALLY[target_idx] & 0b1000_0000_0000_0000 != 0
            #         ):
            #             continue
            #         else:
            #             print(f"should be here? go debug")
            #             continue

            # # if you found ammo, and there is no gunner ready, place one
            # if len(ammo_idxs) > 0 and not gunner_ready:
            #     sorted_ammo_idxs = sort_closest_idxs(ct, ammo_idxs)
            #     ammo_idx = sorted_ammo_idxs[0]
            #     ammo_pos = idx_to_pos(ammo_idx)

            #     # sniper replace
            #     STATE.switch_to(
            #         ct,
            #         snipe_replace.state(idx, ammo_idx, self, turret_type=EntityType.GUNNER),
            #     )
            #     return True

            # there was no gunner, so trying sentinel
            ammo_idxs = []
            print("SNIPE: no gunner spots, or one already, looking for sentinel")

            sentinel_ready = False

            # scan for ready sentinel
            for ally_turret_idx in ally_turret_idxs:
                if get_building_type(ALLY_BUILDINGS[ally_turret_idx]) != EntityType.SENTINEL:
                    continue

                # check if the direction of the turret is correct, it is funded, and in range
                if (
                    get_building_direction(ALLY_BUILDINGS[ally_turret_idx])
                    == direction_to(ally_turret_idx, idx)
                    and SINKS[ally_turret_idx].get_total_sources() > 0
                    and distance_sq_to(ally_turret_idx, idx) <= 32
                ):
                    sentinel_ready = True

            # scan for potential sentinel spots
            for ammo_idx in harvester_neighbour_idxs:
                print(f"- checking {idx_to_pos(ammo_idx)}")

                d2 = distance_sq_to(ammo_idx, idx)
                reverse_turret_dir = direction_to(idx, ammo_idx)
                turret_dir = direction_to(ammo_idx, idx)
                # TODO harvester_sources = SINKS[self.ammo_idx].harvester_sources
                funded = False
                neighbour_harvester_idxs = SINKS[ammo_idx].harvester_sources
                for harvester_idx in neighbour_harvester_idxs:
                    print(
                        f"- neighbour {idx_to_pos(harvester_idx)} {turret_dir} {direction_to(ammo_idx, harvester_idx)}"
                    )
                    if turret_dir != direction_to(ammo_idx, harvester_idx):
                        funded = True

                print(f"- {d2} {funded}")

                # TODO make sure sentinel cant brick by facing its source

                # if in sentinel range and funded (no direction issues)
                # try to build that sentinel
                if d2 <= 32 and funded:
                    ammo_idxs.append(ammo_idx)

            print([idx_to_pos(ammo_idx) for ammo_idx in ammo_idxs])
            print(f"sent rdy: {sentinel_ready}")

            # if you found ammo, and there is no sentinel ready, place one
            if len(ammo_idxs) > 0 and not sentinel_ready:
                sorted_ammo_idxs = sort_closest_idxs(ct, ammo_idxs)
                ammo_idx = sorted_ammo_idxs[0]
                ammo_pos = idx_to_pos(ammo_idx)

                print("SNIPE; triggering sentinel")
                # place senti
                STATE.switch_to(
                    ct,
                    place_turret.state(
                        ammo_idx,
                        EntityType.SENTINEL,
                        direction_to(ammo_idx, idx),
                        self,
                        bail=bind_bail(combat_bail, self, check_snipe=False),
                    ),
                )
                return True
    else:
        return False

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
                                    combat_bail,
                                    self,
                                    check_repair_lines=False,
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
                                    combat_bail,
                                    self,
                                    check_repair_lines=False,
                                ),
                            ),
                        )
                        return True
    else:
        return False

    if check_harvester_attack:
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
                    if break_tile.is_on_cooldown(neighbour_idx) or path_to.is_on_cooldown(
                        neighbour_idx
                    ):
                        continue
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
                        attack_harvester.state(
                            idx,
                            self,
                            bail=bind_bail(combat_bail, self, check_harvester_attack=False),
                        ),
                    )
                    return True
    else:
        return False

    return False
