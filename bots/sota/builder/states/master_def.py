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
    ENEMY_BUILDER_BOT,
    SINKS,
    RESOURCE_FLOW,
)
from botlib import (
    best_enemy_core_idx,
    idx_to_pos,
    pos_to_idx,
    xy_to_idx,
    in_bounds,
    vision_iter,
    get_bridge_target_idx,
    not_in_bounds,
    get_building_maxhp,
    get_building_hp,
    get_bot_id,
    get_building_type,
    get_building_id,
    get_building_direction,
)

from botlib.constants import (
    POSITION_CACHE,
    DIRECTION_CACHE,
    DIRECTION_DELTAS,
    CARDINAL_DIRECTION_DELTAS,
)

from ..utility.movement import move_to, safe_move

from ..data import (
    BOT_PATHING,
    DISTANCE_FIELD,
    VISION_DELTAS,
    BRIDGE_COOP_COOLDOWN,
    ORES_COMPLETED,
)

import random

# ------------- STATES -------------

from .fundamentals import path_to, fog_max, heal_target
from .gather_resource import build_resource_line, check_ore_groups

# ------------- STATES -------------


from .attacks import attack_harvester, snipe
from .defence import reroute_destroy


def heal_bail_chase(ct: Controller, self, enemy_bot_id, chase_distance: int = 10):

    return False


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


def def_bail(ct: Controller, self: fog_max.state):
    # Look for ally harvesters to defend (attack)

    harvester_idxs = []
    under_bot_attack_idxs = []
    other_damaged_idxs = []
    ally_bots_idxs = []
    # doesnt include launchers
    enemy_turret_idxs = []

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
                # if we own the tile, then we can use it for ammo flow
                if (BITBOARD_ALLY[neighbour_idx] & 0b1111_1111_1111_1100) != 0:
                    # everything except builder bot
                    ally_ammo_flow.append(idx)

    # NOTE, comment this out for stability, UNTESTED
    # TODO, test
    # if enemy turret, try and reroute destroy it
    for idx in enemy_turret_idxs:
        ally_idx = check_turret_sinks(ct, idx, 3)
        if ally_idx is not None:
            STATE.switch_to(ct, reroute_destroy.state(ally_idx, self))
            return True

    # if we are the closest bot to the tile being attacked, go heal it

    for idx in under_bot_attack_idxs:
        my_d2 = Position.distance_squared(UNIT_INFO.position, idx_to_pos(idx))
        best_d2 = 9999
        enemy_bot_id = get_bot_id(ENEMY_BUILDER_BOT[idx])
        for bot_idx in ally_bots_idxs:
            d2 = Position.distance_squared(idx_to_pos(bot_idx), idx_to_pos(idx))
            if d2 < best_d2:
                best_d2 = d2
        if my_d2 < best_d2:
            # TODO, do that bail function that  stops healing if the guy moves, then chases them for x turns
            # bail=heal_bail_chase()
            STATE.switch_to(ct, heal_target.state(idx, self, fatigue=4))
            return True

    # TODO readd this whole sniper thing

    # # the above logic should catch all enemy turrets that we fund
    # # therefore we add in other obstructions, and try to attack them
    # # so enemy/unfunded turrets, launchers, or barriers
    # enemy_turret_idxs.extend(enemy_lb_idxs)

    # # try and snipe them
    # for idx in enemy_turret_idxs:
    #     # find the spot we want to use to fund our attack
    #     # TODO NOTE check
    #     # we are in vision range => we always use a sentinel, which should clear the obstacle
    #     for ammo_idx in ally_ammo_flow:
    #         ammo_type = get_building_type(ALLY_BUILDINGS[ammo_idx])
    #         ammo_dir = get_building_direction(ALLY_BUILDINGS[ammo_idx])
    #         ammo_bridge_idx = get_bridge_target_idx(ALLY_BUILDINGS[ammo_idx])

    #         target_id = get_building_id(idx)

    #         # TODO maybe bail function
    #         if ammo_type == EntityType.BRIDGE:
    #             STATE.switch_to(
    #                 ct,
    #                 snipe,
    #                 snipe.state(
    #                     idx,
    #                     ammo_idx,
    #                     self,
    #                     ammo_type=ammo_type,
    #                     target_id=target_id,
    #                     ammo_bridge_idx=ammo_bridge_idx,
    #                 ),
    #             )
    #             return True
    #         else:
    #             STATE.switch_to(
    #                 ct,
    #                 snipe,
    #                 snipe.state(
    #                     idx,
    #                     ammo_idx,
    #                     self,
    #                     ammo_type=ammo_type,
    #                     target_id=target_id,
    #                     ammo_dir=ammo_dir,
    #                 ),
    #             )
    #             return True

    # TODO, reimplement harvesters with repair strategy
    for idx in harvester_idxs:
        spot_available = False
        enemy_on_harvester = False

        x, y = POSITION_CACHE[idx]
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
                # break

            if (BITBOARD_ENEMY[neighbour_idx] & 0b1111_1111_1111_1100) != 0:
                # NOTE exluded the bots, maybe need to exclude bots, i think the attack state will probably just bail anyways, weird behaviour potentially
                enemy_on_harvester = True

            # TODO sometimes has no spots in attack harvester state causing bounce back and bricking the bot
            if spot_available and enemy_on_harvester:
                # TODO change self to the repurpose state
                STATE.switch_to(
                    ct,
                    attack_harvester.state(idx, self),
                )
                return True

    # TODO Heal leftovers?
    # for idx in other_damaged_idxs:
    #     STATE.switch_to(ct, heal_target.state(idx, self, fatigue=4))
    #     return True

    return False


# NOTE, managing how the patrol works is important

# SECOND priority, reroute destroy enemy turrets, or barriers, launchers
# NOTE THIS ONLY DESTROYS LINE INTERRUPTIONS, need new logic to clear misc garbage (mainly launchers)

# THIRD priority, heal tile being attacked, with enemy bot on it, i.e. being attacked now, maybe follow for fatigue after?

# HEALER PROTOCOL
# look for tile being damaged by enemy bot
# TODO FOURTH priority, heal random damaged, this could lead to the bot chasing nonsense, and not exploring enough
