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
    ENEMY_BUILDER_BOT,
    ALLY_BUILDER_BOT,
    COVERAGE,
)
from botlib import (
    best_enemy_core_idx,
    xy_to_idx,
    in_bounds,
    vision_iter,
    get_bridge_target_idx,
    not_in_bounds,
    bind_bail,
    get_building_maxhp,
    get_building_hp,
    get_bot_hp,
    get_bot_id,
    cardinal_iter,
    get_building_direction,
    direction_to,
    sort_closest_idxs,
    distance_sq_to,
    get_building_type,
    get_building,
    pos_to_idx,
    idx_to_pos,
    xy_to_idx,
    adjacent_iter,
)

from botlib.constants import (
    POSITION_CACHE,
    DIRECTION_CACHE,
    DIRECTION_DELTAS,
    CARDINAL_DIRECTION_DELTAS,
    SPLITTER_DIRECTIONS,
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
from .gather_resource import build_resource_line, check_ore_groups, path_place_harvester
from .defence import repair_harvester

# ------------- STATES -------------


from .attacks import attack_harvester


# TODO doesnt seem to work as intended
def is_funding_ally_turret(idx: int, max_depth=5):
    if max_depth <= 0:
        return False

    if (BITBOARD_ALLY[idx] & 0b0011_1000) != 0:
        # funding turret
        return True

    if (BITBOARD_ALLY[idx] & 0b0001_1111_1000_0000) == 0 and (
        BITBOARD_ENEMY[idx] & 0b0001_1111_1000_0000
    ) == 0:
        # Not a conveyor tile
        return False

    if (BITBOARD_ALLY[idx] & 0b0010_1000_0000) != 0 or (
        BITBOARD_ENEMY[idx] & 0b0010_1000_0000
    ) != 0:
        # Conveyor
        next_idx = pos_to_idx(POSITION_CACHE[idx].add(get_building_direction(get_building(idx))))
        return is_funding_ally_turret(next_idx, max_depth - 1)

    if (BITBOARD_ALLY[idx] & 0b0001_0000_0000) != 0 or (
        BITBOARD_ENEMY[idx] & 0b0001_0000_0000
    ) != 0:
        # Splitter
        x, y = idx_to_pos(idx)
        for dx, dy in SPLITTER_DIRECTIONS[(get_building(idx) >> 4) & 0xFFF]:
            nx = x + dx
            ny = y + dy
            result = is_funding_ally_turret(xy_to_idx(nx, ny), max_depth - 1)
            if result:
                return result
        return False

    if (BITBOARD_ALLY[idx] & 0b0100_0000_0000) != 0 or (
        BITBOARD_ENEMY[idx] & 0b0100_0000_0000
    ) != 0:
        # Bridge
        return is_funding_ally_turret(get_bridge_target_idx(get_building(idx)), max_depth - 1)

    return False


def pd_creep_bail(ct: Controller, self: build_resource_line.state, check_snipe=True):
    print("PD CREEP BAIL")
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
    free_ti_idxs = []

    enemy_harvester_idxs = []

    # includes launchers
    enemy_lb_idxs = []

    ally_ammo_flow = []
    ally_ammo_flow_only = []
    harvester_neighbour_idxs = []
    free_ti_neighbour_idxs = []

    enemy_empty_soft_sinks = []
    enemy_soft_sinks = []

    enemy_core_idxs = []

    for idx in vision_iter():
        if (BITBOARD_ENV[idx] & 0b0010) != 0:
            # Skip walls
            continue

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

        if (BITBOARD_ENEMY[idx] & 0b1000_0000_0000) != 0 and (BITBOARD_ENV[idx] & 0b0100) != 0:
            # Harvester NOTE only cares about titanium harvesters

            # look for neighbours we own, if we do then add to ally_ammo_flow
            enemy_harvester_idxs.append(idx)

        if (BITBOARD_ENEMY[idx] & 0b0000_0000_0000_0100) != 0:
            # Harvester NOTE only cares about titanium harvesters

            # look for neighbours we own, if we do then add to ally_ammo_flow
            enemy_core_idxs.append(idx)

        if (
            BITBOARD_ENEMY[idx] == 0
            and BITBOARD_ALLY[idx] == 0
            and (BITBOARD_ENV[idx] & 0b0100) != 0
        ):
            # look for free ti and neighbours
            free_ti_idxs.append(idx)

            x, y = POSITION_CACHE[idx]

            for dx, dy in CARDINAL_DIRECTION_DELTAS:
                nx = x + dx
                ny = y + dy
                if not_in_bounds(nx, ny):
                    continue
                neighbour_idx = ny * MAP_INFO.width + nx
                free_ti_neighbour_idxs.append(neighbour_idx)

        # Look at enemy sinks
        if BITBOARD_ENEMY[idx] != 0:
            sink = SINKS[idx]
            soft_sources = sink.bridge_sources | sink.conveyor_sources
            has_ammo_flow = False
            for source_idx in soft_sources:
                ti_flow_node = RESOURCE_FLOW[0][source_idx]
                ti_flow = ti_flow_node.get_flow()
                ti_stall = ti_flow_node.get_stall()
                ref_flow_node = RESOURCE_FLOW[2][source_idx]
                ref_flow = ref_flow_node.get_flow()
                ref_stall = ref_flow_node.get_stall()
                if ti_flow > 0.2 or ref_flow > 0.2 or ti_stall > 0.2 or ref_stall > 0.2:
                    has_ammo_flow = True
                    break

            if (
                has_ammo_flow
                and not break_tile.is_on_cooldown(idx)
                and (BITBOARD_ALLY[idx] & 0b10) == 0
            ):
                # Check its not funding a turret downstream
                if is_funding_ally_turret(idx):
                    continue

                if BITBOARD_ENEMY[idx] == 0:
                    enemy_empty_soft_sinks.append(idx)
                elif (BITBOARD_ENEMY[idx] & 0b1010_0101_1000_0000) != 0:
                    enemy_soft_sinks.append(idx)

    enemy_sniper_target_idxs = []

    enemy_sniper_target_idxs.extend(enemy_core_idxs)
    enemy_sniper_target_idxs.extend(enemy_harvester_idxs)

    if check_snipe:
        turret_idx = self.last_idx
        if self.last_idx == self.start_idx:
            turret_idx = None
            closest = 0
            for nidx in cardinal_iter(self.last_idx):
                dist = distance_sq_to(nidx, self.end_idx)
                if turret_idx is None or dist < closest:
                    closest = dist
                    turret_idx = nidx

            if turret_idx is None:
                print("EXIT")
                STATE.switch_to(ct, self.exit_state)
                return True
        for idx in enemy_sniper_target_idxs:
            if distance_sq_to(idx, turret_idx) < 13:
                STATE.switch_to(
                    ct,
                    break_tile.state(
                        turret_idx,
                        self.exit_state,
                        replace_with=EntityType.GUNNER,
                        replace_with_extra=direction_to(turret_idx, idx),
                    ),
                )
                return True

        if len(enemy_core_idxs) > 0 and distance_sq_to(best_enemy_core_idx(), turret_idx) < 13:
            return True

    else:
        return False

    return False


def rush_bail(
    ct: Controller,
    self: fog_max.state,
    check_harvester_attack: bool = True,
    check_snipe: bool = True,
    check_empty_sinks: bool = True,
    check_soft_sinks: bool = True,
    check_launchers: bool = True,
    check_initiate_pd_creep: bool = True,
):
    # TODO Added scan, needs to be cleaned up
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
    free_ti_idxs = []

    enemy_harvester_idxs = []

    # includes launchers
    enemy_lb_idxs = []

    ally_ammo_flow = []
    ally_ammo_flow_only = []
    harvester_neighbour_idxs = []

    enemy_empty_soft_sinks = []
    enemy_soft_sinks = []

    for idx in vision_iter():
        if (BITBOARD_ENV[idx] & 0b0010) != 0:
            # Skip walls
            continue

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

        if (BITBOARD_ENEMY[idx] & 0b1000_0000_0000) != 0 and (BITBOARD_ENV[idx] & 0b0100) != 0:
            # Harvester NOTE only cares about titanium harvesters

            # look for neighbours we own, if we do then add to ally_ammo_flow
            enemy_harvester_idxs.append(idx)

        if BITBOARD_ENEMY[idx] == 0 and (BITBOARD_ENV[idx] & 0b0100) != 0:
            # look for free ti and neighbours
            free_ti_idxs.append(idx)

        # Look at enemy sinks
        if BITBOARD_ENEMY[idx] != 0:
            sink = SINKS[idx]

            if sink.get_total_sources() == 0:
                continue

            if HAZARDS[4][idx] != 0 or HAZARDS[3][idx] != 0:
                continue

            soft_sources = sink.bridge_sources | sink.conveyor_sources
            has_ammo_flow = False
            for source_idx in soft_sources:
                ti_flow_node = RESOURCE_FLOW[0][source_idx]
                ti_flow = ti_flow_node.get_flow()
                ti_stall = ti_flow_node.get_stall()
                ref_flow_node = RESOURCE_FLOW[2][source_idx]
                ref_flow = ref_flow_node.get_flow()
                ref_stall = ref_flow_node.get_stall()
                if ti_flow > 0.2 or ref_flow > 0.2 or ti_stall > 0.2 or ref_stall > 0.2:
                    has_ammo_flow = True
                    break

            if (
                has_ammo_flow
                and not break_tile.is_on_cooldown(idx)
                and (BITBOARD_ALLY[idx] & 0b10) == 0
            ):
                # Check its not funding a turret downstream
                if is_funding_ally_turret(idx):
                    continue

                if BITBOARD_ENEMY[idx] == 0:
                    enemy_empty_soft_sinks.append(idx)
                elif (BITBOARD_ENEMY[idx] & 0b1010_0101_1000_0000) != 0:
                    enemy_soft_sinks.append(idx)

    if check_initiate_pd_creep:
        for idx in free_ti_idxs:
            if distance_sq_to(idx, UNIT_INFO.ally_core_idx) < distance_sq_to(
                idx, best_enemy_core_idx()
            ):
                continue

            num_enemy = 0
            num_ally = 0
            for neighbour_idx in cardinal_iter(idx):
                if (BITBOARD_ENEMY[neighbour_idx] & 0b0011_1000) != 0:
                    num_enemy += 1
                elif BITBOARD_ALLY[neighbour_idx] & 0b0001_1000:
                    num_ally += 1

            if num_enemy == 0 and num_ally > 0:
                print("GO TO PD CREEP")
                STATE.switch_to(
                    ct,
                    break_tile.state(
                        idx,
                        build_resource_line.state(
                            idx, best_enemy_core_idx(), self, bail=pd_creep_bail, merge=False
                        ),
                        replace_with=EntityType.HARVESTER,
                    ),
                )
                return True
    else:
        return False

    if check_launchers:
        if (
            can_afford(ct, EntityType.LAUNCHER)
            and distance_sq_to(UNIT_INFO.position, best_enemy_core_idx()) < 25
        ):
            # Look for adjacent enemy bots
            has_adjacent_enemy = False
            ally_tiles = []
            empty_tiles = []
            enemy_tiles = []
            for idx in adjacent_iter(UNIT_INFO.position_idx):
                if (BITBOARD_ENV[idx] & 0b0010) != 0:
                    continue

                if (BITBOARD_ENEMY[idx] & 0b10) != 0:
                    has_adjacent_enemy = True
                    continue

                if COVERAGE[6][idx] > 0:
                    # Do not place in a spot already covered by launcher
                    continue

                num_conveyors = 0
                for n in adjacent_iter(idx):
                    if (BITBOARD_ENEMY[n] & 0b0111_1000_0000) != 0:
                        num_conveyors += 1

                if num_conveyors < 3:
                    continue

                if BITBOARD_ALLY[idx] == 0 and BITBOARD_ENEMY[idx] == 0:
                    empty_tiles.append(idx)
                elif (BITBOARD_ALLY[idx] & 0b1000_0000_0000_0000) != 0 or (
                    BITBOARD_ENEMY[idx] & 0b1000_0000_0000_0000
                ) != 0:
                    empty_tiles.append(idx)
                elif (BITBOARD_ALLY[idx] & 0b1110_0000_0000_0000) != 0:
                    ally_tiles.append(idx)
                elif (BITBOARD_ENEMY[idx] & 0b0010_0101_1000_0000) != 0:
                    enemy_tiles.append(idx)

            if has_adjacent_enemy:
                if len(empty_tiles) > 0:
                    STATE.switch_to(
                        ct,
                        break_tile.state(
                            empty_tiles[0],
                            self,
                            replace_with=EntityType.LAUNCHER,
                            bail=bind_bail(rush_bail, self, check_launchers=False),
                        ),
                    )
                    return True
                if len(ally_tiles) > 0:
                    STATE.switch_to(
                        ct,
                        break_tile.state(
                            ally_tiles[0],
                            self,
                            replace_with=EntityType.LAUNCHER,
                            bail=bind_bail(rush_bail, self, check_launchers=False),
                        ),
                    )
                    return True
                if len(enemy_tiles) > 0:
                    STATE.switch_to(
                        ct,
                        break_tile.state(
                            enemy_tiles[0],
                            self,
                            replace_with=EntityType.LAUNCHER,
                            bail=bind_bail(rush_bail, self, check_launchers=False),
                        ),
                    )
                    return True

    else:
        return False

    if check_harvester_attack:
        # Look for harvesters to attack
        for idx in enemy_harvester_idxs:
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
                print(
                    f"{POSITION_CACHE[neighbour_idx]} - cooldown: {break_tile.is_on_cooldown(neighbour_idx)}"
                )
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

                def harvester_bail(ct: Controller, state: attack_harvester.state):
                    if BITBOARD_ENEMY[state.harvester_idx] == 0:
                        STATE.switch_to(ct, self)
                        return True
                    return rush_bail(ct, self, check_harvester_attack=False)

                atk = attack_harvester.state(
                    idx,
                    self,
                )
                atk.bail = bind_bail(harvester_bail, atk)

                STATE.switch_to(ct, atk)
                return True
    else:
        return False

    enemy_sniper_target_idxs = []

    if check_snipe:
        print(f"snipe afford: {can_afford(ct, EntityType.SENTINEL)}")
        if can_afford(ct, EntityType.SENTINEL):
            for idx in enemy_sniper_target_idxs:
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
                        and distance_sq_to(ally_turret_idx, idx) < 32
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
                    if d2 < 32 and funded:
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
                        break_tile.state(
                            ammo_idx,
                            self,
                            replace_with=EntityType.SENTINEL,
                            replace_with_extra=direction_to(ammo_idx, idx),
                            bail=bind_bail(rush_bail, self, check_snipe=False),
                        ),
                    )
                    return True

    if check_empty_sinks:
        for idx in enemy_empty_soft_sinks:
            if (BITBOARD_ALLY[idx] & 0b0011_1000) == 0:
                if distance_sq_to(idx, best_enemy_core_idx()) < 13:
                    turret_type = EntityType.GUNNER
                elif distance_sq_to(idx, best_enemy_core_idx()) < 32:
                    turret_type = EntityType.SENTINEL
                else:
                    turret_type = EntityType.GUNNER
                direction = direction_to(idx, best_enemy_core_idx())

                def bail_on_not_funded(ct: Controller, state: break_tile.state):
                    if is_funding_ally_turret(state.target_idx):
                        print("FUNDING TURRET, DONT STOP IT")
                        STATE.switch_to(ct, state.exit_state)
                        return True

                    ti_flow = RESOURCE_FLOW[0][state.target_idx]
                    ref_flow = RESOURCE_FLOW[2][state.target_idx]
                    if SINKS[state.target_idx].get_total_sources() == 0 or (
                        ti_flow.get_flow() < 0.20
                        and ref_flow.get_flow() < 0.20
                        and ti_flow.get_stall() < 0.20
                        and ref_flow.get_stall() < 0.20
                    ):
                        print("NOT FUNDED")
                        STATE.switch_to(ct, state.exit_state)
                        return True
                    return rush_bail(ct, self, check_empty_sinks=False)

                STATE.switch_to(
                    ct,
                    break_tile.state(
                        idx,
                        self,
                        replace_with=turret_type,
                        replace_with_extra=direction,
                        bail=bail_on_not_funded,
                    ),
                )
                return True
    else:
        return

    if check_soft_sinks:
        for idx in enemy_soft_sinks:
            if (BITBOARD_ALLY[idx] & 0b0011_1000) == 0:
                turret_type = EntityType.GUNNER
                if distance_sq_to(idx, best_enemy_core_idx()) < 36:
                    turret_type = EntityType.SENTINEL
                direction = direction_to(idx, best_enemy_core_idx())

                def bail_on_not_funded(ct: Controller, state: break_tile.state):
                    if is_funding_ally_turret(state.target_idx):
                        print("FUNDING TURRET, DONT STOP IT")
                        STATE.switch_to(ct, state.exit_state)
                        return True

                    ti_flow = RESOURCE_FLOW[0][state.target_idx]
                    ref_flow = RESOURCE_FLOW[2][state.target_idx]
                    if SINKS[state.target_idx].get_total_sources() == 0 or (
                        ti_flow.get_flow() < 0.20
                        and ref_flow.get_flow() < 0.20
                        and ti_flow.get_stall() < 0.20
                        and ref_flow.get_stall() < 0.20
                    ):
                        print("NOT FUNDED")
                        STATE.switch_to(ct, state.exit_state)
                        return True
                    return rush_bail(ct, self, check_soft_sinks=False)

                STATE.switch_to(
                    ct,
                    break_tile.state(
                        idx,
                        self,
                        replace_with=turret_type,
                        replace_with_extra=direction,
                        bail=bail_on_not_funded,
                    ),
                )
                return True
    else:
        return False

    return False


class attack_empty_sink:
    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            exit_state,
            bail: Callable[[Controller, attack_empty_sink.state], bool] | None = None,
        ):
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

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state

    # ------------- METHODS -------------

    @classmethod
    def example_method(self):
        pass


STATE.register(attack_empty_sink)
