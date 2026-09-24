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
    ALLY_BUILDER_BOT,
    PREV_ALLY_BUILDINGS,
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
    cardinal_iter,
    adjacent_iter,
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
from .defence import (
    reroute_destroy,
    repair_throughput,
    follow_enemy,
    check_turret_sinks,
    repair_harvester,
)
from .master_def import def_bail
from .attacks import snipe_replace, place_turret


def generic_bail(
    ct: Controller,
    top_level,
    self: fog_max.state,
    check_econ: bool = True,
    check_repair_lines: bool = True,
    check_heal: bool = True,
    check_snipe: bool = True,
    check_reroute_repair: bool = True,
    check_repair_harvesters: bool = True,
    heal_everything: bool = True,
    check_follow_enemies: bool = True,
    dont_repair_harvesters: bool = False,
):
    print("generic bail")

    harvester_idxs = []
    under_bot_attack_idxs = []
    under_bot_attack_ids = []
    other_damaged_idxs = []
    ally_bots_idxs = []
    ally_bot_ids = []
    enemy_bot_idxs = []
    enemy_bot_ids = []
    # doesnt include launchers
    enemy_turret_idxs = []
    ally_turret_idxs = []

    # includes launchers
    enemy_lb_idxs = []

    ally_ammo_flow = []
    ally_ammo_flow_only = []
    harvester_neighbour_idxs = []

    enemy_harvester_idxs = []

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
            under_bot_attack_ids.append(get_bot_id(ENEMY_BUILDER_BOT[idx]))
        elif (BITBOARD_ALLY[idx] != 0) and target_hp < max_hp:
            other_damaged_idxs.append(idx)

        # find ally bots, and not yourself
        if (BITBOARD_ALLY[idx] & 0b0010) != 0 and UNIT_INFO.position_idx != idx:
            ally_bots_idxs.append(idx)
            ally_bot_ids.append(get_bot_id(ALLY_BUILDER_BOT[idx]))

        if (BITBOARD_ENEMY[idx] & 0b0010) != 0:
            enemy_bot_idxs.append(idx)
            enemy_bot_ids.append(get_bot_id(ENEMY_BUILDER_BOT[idx]))

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

        if (BITBOARD_ENEMY[idx] & 0b1000_0000_0000) != 0 and (
            BITBOARD_ENV[idx] & 0b0100
        ) != 0:
            # Harvester NOTE only cares about titanium harvesters

            # look for neighbours we own, if we do then add to ally_ammo_flow
            enemy_harvester_idxs.append(idx)

        if (BITBOARD_ALLY[idx] & 0b1000_0000_0000) != 0 and (
            BITBOARD_ENV[idx] & 0b0100
        ) != 0:
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
            if ally_idx is not None:
                print(f"turret sink {POSITION_CACHE[ally_idx]}")
                STATE.switch_to(
                    ct,
                    reroute_destroy.state(
                        idx,
                        self,
                        known_turret_sink_idx=ally_idx,
                        bail=bind_bail(
                            generic_bail, top_level, self, check_reroute_repair=False
                        ),
                    ),
                )
                return True
                # if break_tile.is_on_cooldown(ally_idx) or path_to.is_on_cooldown(ally_idx):
                #     continue
                # spot_available = False
                # x, y = POSITION_CACHE[ally_idx]

                # for dx, dy in CARDINAL_DIRECTION_DELTAS:
                #     nx = x + dx
                #     ny = y + dy
                #     if not_in_bounds(nx, ny):
                #         continue
                #     neighbour_idx = ny * MAP_INFO.width + nx
                #     if break_tile.is_on_cooldown(neighbour_idx) or path_to.is_on_cooldown(
                #         neighbour_idx
                #     ):
                #         continue
                #     if (
                #         (BITBOARD_ALLY[neighbour_idx] & 0b0101_1010_0111_1110) == 0
                #         and (BITBOARD_ENEMY[neighbour_idx] & 0b0101_1010_0111_1110) == 0
                #         and (BITBOARD_ENV[neighbour_idx] & 0b0010) == 0
                #     ):
                #         print(f"found spot {POSITION_CACHE[neighbour_idx]}")
                #         # ['builder_bot', 'core', 'gunner', 'sentinel', 'breach', 'launcher', 'armoured_conveyor', 'harvester', 'foundry', 'barrier']
                #         # wall
                #         # if no impassibles, there is a valid spot so break away
                #         spot_available = True
                #         break
                # if spot_available:
                #     STATE.switch_to(
                #         ct,
                #         reroute_destroy.state(
                #             ally_idx,
                #             self,
                #             bail=bind_bail(
                #                 generic_bail, top_level, self, check_reroute_repair=False
                #             ),
                #         ),
                #     )
                #     return True
    else:
        return False

    # if we are the closest bot to the tile being attacked, go heal it

    if check_heal:
        print(f"check heal: {under_bot_attack_ids}")

        if len(under_bot_attack_ids) > 0:
            enemy_order = sorted(
                range(len(under_bot_attack_ids)), key=under_bot_attack_ids.__getitem__
            )
            ally_order = sorted(range(len(ally_bot_ids)), key=ally_bot_ids.__getitem__)

            i = 0
            while i < len(ally_order):
                if ally_bot_ids[ally_order[i]] == UNIT_INFO.id:
                    break
                i += 1

            chosen_i = enemy_order[i % len(enemy_order)]
            tile_idx = under_bot_attack_idxs[chosen_i]

            print(
                f"{get_building_hp(PREV_ALLY_BUILDINGS[tile_idx])} {
                    get_building_hp(ALLY_BUILDINGS[tile_idx])
                }"
            )

            prev_hp = get_building_hp(PREV_ALLY_BUILDINGS[tile_idx])
            curr_hp = get_building_hp(ALLY_BUILDINGS[tile_idx])
            loosing_hp = (
                prev_hp is not None and curr_hp is not None and prev_hp > curr_hp
            )

            if i < len(enemy_order) or loosing_hp:
                STATE.switch_to(
                    ct,
                    heal_target.state(
                        tile_idx,
                        follow_enemy.state(
                            under_bot_attack_ids[chosen_i],
                            self,
                            bail=bind_bail(
                                generic_bail, top_level, self, check_econ=False
                            ),
                        ),
                        fatigue=0,
                        bail=bind_bail(generic_bail, top_level, self, check_heal=False),
                    ),
                )
                return True

        # under_bot_attack_idxs = sort_closest_idxs(ct, under_bot_attack_idxs)
        # print(f"{under_bot_attack_idxs}")
        # for idx in under_bot_attack_idxs:
        #     my_d2 = Position.distance_squared(UNIT_INFO.position, idx_to_pos(idx))
        #     best_d2 = 9999
        #     closest_ally_idx = None

        #     enemy_bot_id = get_bot_id(ENEMY_BUILDER_BOT[idx])

        #     for bot_idx in ally_bots_idxs:
        #         d2 = Position.distance_squared(idx_to_pos(bot_idx), idx_to_pos(idx))
        #         if d2 < best_d2:
        #             best_d2 = d2
        #             closest_ally_idx = bot_idx

        #     is_best_candidate = False
        #     if my_d2 < best_d2:
        #         is_best_candidate = True
        #     elif my_d2 == best_d2 and closest_ally_idx is not None:
        #         if UNIT_INFO.position_idx < closest_ally_idx:
        #             is_best_candidate = True

        #     print(f"- {is_best_candidate} {idx_to_pos(idx)} {my_d2 == best_d2} {closest_ally_idx}")

        #     if is_best_candidate:
        #         # STATE.switch_to(ct, heal_target.state(idx, self, fatigue=4))
        #         STATE.switch_to(
        #             ct,
        #             heal_target.state(
        #                 idx,
        #                 follow_enemy.state(
        #                     enemy_bot_id,
        #                     self,
        #                     bail=bind_bail(generic_bail, top_level, self, check_econ=False),
        #                 ),
        #                 fatigue=0,
        #             ),
        #         )
        #         return True

    else:
        return False

    # sniper launchers

    enemy_lb_idxs.extend(enemy_harvester_idxs)
    enemy_lb_idxs.extend(enemy_turret_idxs)

    if check_snipe:
        print(f"snipe afford: {can_afford(ct, EntityType.SENTINEL)}")
        if can_afford(ct, EntityType.SENTINEL):
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
                    if (
                        get_building_type(ALLY_BUILDINGS[ally_turret_idx])
                        != EntityType.SENTINEL
                    ):
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

                    if (BITBOARD_ENEMY[ammo_idx] & 0b10) != 0:
                        continue

                    if (BITBOARD_ENV[ammo_idx & 0b0010]) != 0:
                        continue

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
                            bail=bind_bail(
                                generic_bail, top_level, self, check_snipe=False
                            ),
                        ),
                    )
                    return True
    else:
        return False

    # TODO START OF REPAIR LOGIC --------------------------

    print(
        f"repair harvesters: {check_repair_harvesters and not dont_repair_harvesters}"
    )

    if check_repair_harvesters and not dont_repair_harvesters:
        # TODO allow enemy harvesters as well, maybe consider separate
        for harvester_idx in harvester_idxs:
            working = False
            num_facing_in = 0
            for neighbour_idx in cardinal_iter(harvester_idx):
                if (BITBOARD_ENEMY[neighbour_idx] & ~(0b1010_0000_0000_0010)) != 0:
                    # if there is an enemy, non road or marker, working = false, and break
                    working = False
                    break
                elif (BITBOARD_ALLY[neighbour_idx] & 0b0001_0000_0000_0000) != 0:
                    working = True
                elif (BITBOARD_ALLY[neighbour_idx] & 0b0100_0000_0100) != 0:
                    # if bridge or core then harvester is working
                    working = True
                elif (
                    BITBOARD_ALLY[neighbour_idx] & 0b0011_1000_0000
                ) != 0 and get_building_direction(
                    ALLY_BUILDINGS[neighbour_idx]
                ) != direction_to(neighbour_idx, harvester_idx):
                    # is there is an ally conveyor type, and its direction is NOT into the harvester, then it is working
                    working = True
                elif (
                    BITBOARD_ALLY[neighbour_idx] & 0b0011_1000_0000
                ) != 0 and get_building_direction(
                    ALLY_BUILDINGS[neighbour_idx]
                ) == direction_to(neighbour_idx, harvester_idx):
                    # is there is an ally conveyor type, and its direction is into the harvester, then count
                    num_facing_in += 1
            if num_facing_in == 4:
                working = True
            if not working:
                STATE.switch_to(
                    ct,
                    repair_harvester.state(
                        harvester_idx,
                        UNIT_INFO.ally_core_idx,
                        self,
                        bail=bind_bail(
                            generic_bail, top_level, self, check_repair_harvesters=False
                        ),
                    ),
                )
    elif not dont_repair_harvesters:
        return

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
                                ),
                            ),
                        )
                        return True
                if len(splitter_target_idxs):
                    splitter_good = False
                    for splitter_target_idx in splitter_target_idxs:
                        if (
                            BITBOARD_ALLY[splitter_target_idx] & 0b0001_0111_1000_0100
                        ) != 0:
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
                                ),
                            ),
                        )
                        return True
    else:
        return False

    # START OF ECONOMY LOGIC --------------------------
    # Check for nearby ore, if any switch to gather_resource

    if check_econ:
        if len(TITANIUM_HARVESTERS_PLACED) > 0 and top_level.econ:
            top_level.local = None  # Disable local so bots can explore to find ax

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
                                        bail=bind_bail(
                                            generic_bail,
                                            top_level,
                                            self,
                                            check_econ=False,
                                        ),
                                    ),
                                )
                                return True

        print(f"check eco = {top_level.econ}")

        if top_level.econ:
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
                        and get_building_type(ENEMY_BUILDINGS[idx])
                        != EntityType.HARVESTER
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
                        BITBOARD_ALLY[
                            bridge_target := get_bridge_target_idx(ALLY_BUILDINGS[idx])
                        ]
                        & 0b0111_1000_0110
                    ) == 0:
                        dist = DISTANCE_FIELD.dist(bridge_target)
                        if dist is not None:
                            print(
                                f"bridge coop: {POSITION_CACHE[bridge_target]}"
                            )  # TODO gets stuck if cannot build resource line
                            BRIDGE_COOP_COOLDOWN[idx] = (
                                STATE.tick + 20
                            )  # 20 round cooldown
                            STATE.switch_to(
                                ct,
                                build_resource_line.state(
                                    bridge_target,
                                    UNIT_INFO.ally_core_idx,
                                    self,
                                    skip_first=False,
                                    bail=bind_bail(
                                        generic_bail, top_level, self, check_econ=False
                                    ),
                                ),
                            )
                            return True

                # Look for ore
                if (
                    (BITBOARD_ENV[idx] & 0b0100) != 0
                    and not ORES_COMPLETED[idx]
                    # TODO NOT COMPATIBLE WITH DEFENSIVE CONVEYORS NEEDS CHANGE ALSO DO IN PATHTOORE
                    and (BITBOARD_ALLY[idx] & 0b0001_1111_0011_1010) == 0
                    and (BITBOARD_ENEMY[idx] & 0b0101_1010_0111_1000) == 0
                    and (HAZARDS[3][idx] == 0)
                    and (HAZARDS[4][idx] == 0)
                ):
                    if (BITBOARD_ALLY[idx] & 0b0000_0000_1000_0000) != 0:
                        harvester_pos = idx_to_pos(idx).add(
                            get_building_direction(ALLY_BUILDINGS[idx])
                        )
                        if (
                            BITBOARD_ALLY[pos_to_idx(harvester_pos)] & 0b1000_0000_0000
                        ) != 0:
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
                        path_to_ore = path_to.state(
                            idx, 2, None
                        )  # TODO think about dist to walk
                        path_to_ore.bail = bind_bail(
                            closer_ore_bail,
                            path_to_ore,
                            bind_bail(
                                generic_bail,
                                top_level,
                                self,
                                check_econ=False,
                                dont_repair_harvesters=True,
                            ),
                        )
                        check_ore_group = check_ore_groups.state(
                            0b0100,
                            idx,
                            UNIT_INFO.ally_core_idx,
                            self,
                            origin_path_to=path_to_ore,
                            resource_bail=bind_bail(
                                generic_bail,
                                top_level,
                                self,
                                check_econ=False,
                                dont_repair_harvesters=True,
                            ),
                        )
                        path_to_ore.exit_state = check_ore_group
                        STATE.switch_to(ct, path_to_ore)
                        return True
    else:
        return False

    # If nothing, but spot enemy follow them
    print("checking for enemies")

    if check_follow_enemies:
        enemy_bot_idxs = sort_closest_idxs(ct, enemy_bot_idxs)
        for idx in enemy_bot_idxs:
            has_ally = False
            for adjacent_idx in adjacent_iter(idx):
                if (BITBOARD_ALLY[adjacent_idx] & 0b10) != 0:
                    has_ally = True
                    break
            if has_ally:
                continue

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

            print(
                f"- {is_best_candidate} {idx_to_pos(idx)} {my_d2 == best_d2} {closest_ally_idx}"
            )

            if is_best_candidate:
                follow_enemy_state = follow_enemy.state(
                    enemy_bot_id,
                    self,
                    bail=bind_bail(
                        generic_bail, top_level, self, check_follow_enemies=False
                    ),
                    stop_on_better_ally_nearby=True,
                )

                STATE.switch_to(ct, follow_enemy_state)
                return True
    else:
        return False

    if heal_everything:
        # TODO Heal leftovers?
        for idx in other_damaged_idxs:
            STATE.switch_to(
                ct,
                heal_target.state(
                    idx,
                    self,
                    fatigue=4,
                    bail=bind_bail(
                        generic_bail, top_level, self, heal_everything=False
                    ),
                ),
            )
            return True
    else:
        return False

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
            self.path_task = path_to.state(self.ax_ore_idx, 2, self, bail=self.bail)
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
            STATE.switch_to(ct, break_tile.state(self.ax_ore_idx, self, bail=self.bail))
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
            bail=self.bail,
        )
        foundry_to_core = build_resource_line.state(
            best_target_idx,
            UNIT_INFO.ally_core_idx,
            self.exit_state,
            continue_lines=[ax_to_ti],
            bail=self.bail,
        )
        ax_to_ti.exit_state = State(
            path_place_harvester,
            best_target_idx,
            0b1000,
            foundry_to_core,
            do_foundry=True,
            bail=self.bail,
        )

        STATE.switch_to(ct, ax_to_ti)

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state


STATE.register(produce_ref_ax)
