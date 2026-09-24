from typing import Callable, Optional

from cambc import Controller, Direction, EntityType, Environment, Position, Team

from sentinel import get_highest_hierarchy_positions

from botlib import botlib_init, botlib_update

from botlib import (
    CARDINAL_DIRECTION_DELTAS,
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
    SINKS,
    direction_to,
)

from builder.utility.building import direction_to

# State


def init(ct: Controller) -> None:
    # Run the main core logic
    botlib_init(ct, False)
    run(ct)


def run(ct: Controller) -> None:
    botlib_update(None)

    # NOTE BRAND NEW NEEDS TESTING see if I am on enemy harvester

    harvester_sources = SINKS[UNIT_INFO.position_idx].harvester_sources
    enemy_harvester_idxs = []

    # filter for enemy harvester sources
    for harvester_idx in harvester_sources:
        if BITBOARD_ENEMY[harvester_idx] != 0:
            enemy_harvester_idxs.append(harvester_idx)

    print(f"funded by enemy harvesters: {enemy_harvester_idxs}")

    # if we have some then start destroying
    if len(enemy_harvester_idxs) > 0:
        for enemy_harvester_idx in enemy_harvester_idxs:
            enemy_harvester_dir = direction_to(
                UNIT_INFO.position_idx, enemy_harvester_idx
            )
            reverse_enemy_harvester_dir = direction_to(
                enemy_harvester_idx, UNIT_INFO.position_idx
            )
            if ct.get_ammo_amount() == 0:
                if (
                    ct.can_rotate(enemy_harvester_dir)
                    and ct.get_direction() == enemy_harvester_dir
                ):
                    ct.rotate(reverse_enemy_harvester_dir)
                return
            elif ct.get_ammo_amount() > 0:
                if (
                    ct.can_rotate(enemy_harvester_dir)
                    and ct.get_direction() != enemy_harvester_dir
                ):
                    ct.rotate(enemy_harvester_dir)
                if ct.get_direction() == enemy_harvester_dir and ct.can_fire(
                    idx_to_pos(enemy_harvester_idx)
                ):
                    ct.fire(idx_to_pos(enemy_harvester_idx))
                return

    shoot_harvesters = False

    position = ct.get_position()
    x, y = position.x, position.y
    direction = ct.get_direction(ct.get_id())

    target = ct.get_gunner_target()
    attackable_tiles = ct.get_attackable_tiles()

    rotated_attackable_tiles = []

    potential_directions = [
        Direction.NORTH,
        Direction.NORTHEAST,
        Direction.EAST,
        Direction.SOUTHEAST,
        Direction.SOUTH,
        Direction.SOUTHWEST,
        Direction.WEST,
        Direction.NORTHWEST,
    ]

    # potential_directions.remove(direction)

    # TODO rejig hierachy

    target_hierachy = [
        EntityType.BREACH,  # 5
        EntityType.GUNNER,  # 3
        EntityType.SENTINEL,  # 4
        EntityType.LAUNCHER,  # 6
        EntityType.CORE,  # 2
        EntityType.BUILDER_BOT,  # 1
        EntityType.FOUNDRY,  # 12
        EntityType.BRIDGE,  # 10
        EntityType.SPLITTER,  # 8
        EntityType.CONVEYOR,  # 7
        EntityType.ARMOURED_CONVEYOR,  # 9
        # EntityType.HARVESTER,  # 11
        # EntityType.ROAD,  # 13
        # EntityType.BARRIER,  # 14
        # EntityType.MARKER,  # 15
    ]

    # look for own source(s)
    for dx, dy in CARDINAL_DIRECTION_DELTAS:
        nx = x + dx
        ny = y + dy
        neighbour_idx = xy_to_idx(nx, ny)

        if (
            (BITBOARD_ALLY[neighbour_idx] & 0b0001_1000_0000_0000) != 0
            or BITBOARD_ENEMY[neighbour_idx] & 0b0001_1000_0000_0000
        ) != 0:
            # harvester or foundry
            if dy == 1:
                potential_directions.remove(Direction.SOUTH)
                print("removing my source, SOUTH")
            if dy == -1:
                potential_directions.remove(Direction.NORTH)
                print("removing my source, NORTH")
            if dx == 1:
                potential_directions.remove(Direction.EAST)
                print("removing my source, EAST")
            if dx == -1:
                potential_directions.remove(Direction.WEST)
                print("removing my source, WEST")

    ray_pos_list = []
    ray_type_list = []

    # cast the rays, list all the possible targets
    for dir in potential_directions:
        # print(f"dir: {dir}")
        dx, dy = dir.delta()
        # print(f"dx, dy: {dx}, {dy}")
        target_list = []
        type_list = []
        count = 4
        if (
            (dir == Direction.NORTHEAST)
            or (dir == Direction.NORTHWEST)
            or (dir == Direction.SOUTHEAST)
            or (dir == Direction.SOUTHWEST)
        ):
            count = 3
        # print(f"count: {count}")
        for i in range(1, count):
            nx = x + i * dx
            ny = y + i * dy
            # print(f"nx, ny: {nx}, {ny}")
            target_idx = xy_to_idx(nx, ny)

            # if out of bounds, break
            if not_in_bounds(nx, ny) or not ct.is_in_vision(idx_to_pos(target_idx)):
                break
            # if allied tile (not marker or road) or wall, stop checking this direction
            if (
                (BITBOARD_ALLY[target_idx] & 0b0101_1111_1111_1111 != 0)
                and (BITBOARD_ENEMY[target_idx] & 0b0010 == 0)
            ) or (BITBOARD_ENV[target_idx] & 0b0010):
                # NOTE added and no enemy builder bot, should check now
                break

            # TODO check if want this logic, stop ray on harvester
            if not shoot_harvesters and (
                ((BITBOARD_ALLY[target_idx] & 0b0000_1000_0000_0000) != 0)
                or ((BITBOARD_ENEMY[target_idx] & 0b0000_1000_0000_0000) != 0)
            ):
                break

            # if empty, continue, and check the next tile
            if (BITBOARD_ENEMY[target_idx] == 0) and (BITBOARD_ALLY[target_idx] == 0):
                continue
            # if marker, continue
            if (BITBOARD_ENEMY[target_idx] & 0b1000_0000_0000_0000 != 0) or (
                BITBOARD_ALLY[target_idx] & 0b1000_0000_0000_0000 != 0
            ):
                continue
            target_list.append(idx_to_pos(target_idx))
            # TODO, cannot target bots without adding special logic
            if ct.get_tile_building_id(idx_to_pos(target_idx)) is not None:
                type_list.append(
                    ct.get_entity_type(ct.get_tile_building_id(idx_to_pos(target_idx)))
                )
            else:
                type_list.append(None)
            # print("i think the entity is!")
            # print(ct.get_entity_type(ct.get_tile_building_id(idx_to_pos(target_idx))))
            # print(f"at position {idx_to_pos(target_idx)}")
        ray_pos_list.append(target_list)
        ray_type_list.append(type_list)

    # print(f"ray pos list: {ray_pos_list}]")
    # print(f"ray type list: {ray_type_list}]")

    dir_best_pos = []
    dir_best_types = []

    # find best target in each ray

    if len(ray_pos_list) == 0:
        print("no targets?")
        return

    for i in range(len(ray_pos_list)):
        # TODO, check this is fine
        if len(ray_pos_list[i]) == 0:
            dir_best_pos.append(None)
            dir_best_types.append(None)
        else:
            a, b = get_highest_hierarchy_positions(
                ray_pos_list[i], ray_type_list[i], target_hierachy
            )
            if len(a) == 0:
                dir_best_pos.append(None)
                dir_best_types.append(None)
            else:
                dir_best_pos.append(a[0])
                dir_best_types.append(b)

    # compare best target in each ray
    if len(dir_best_pos) > 0:
        c, d = get_highest_hierarchy_positions(
            dir_best_pos, dir_best_types, target_hierachy
        )
    else:
        c, d = [], None
    # print("c, d")
    print(c, d)

    # if only one highest prio, attack in that direction
    if len(c) == 1:
        target_direction = direction_to(pos_to_idx(position), pos_to_idx(c[0]))
        if ct.can_rotate(target_direction) and ct.get_ammo_amount() > 0:
            ct.rotate(target_direction)
            target = ct.get_gunner_target()
        if (
            target is not None
            and ct.can_fire(target)
            and ct.get_direction(ct.get_id()) == target_direction
        ):
            ct.fire(target)

    elif len(c) > 1:
        for pos in c:
            target_direction = direction_to(pos_to_idx(position), pos_to_idx(pos))
            target_position = pos
            if target_direction == direction:
                break
        if ct.can_rotate(target_direction) and ct.get_ammo_amount() > 0:
            ct.rotate(target_direction)
            target = ct.get_gunner_target()

        # TODO, maybe change this if statement vvv
        if (
            target is not None
            and ct.can_fire(target)
            and direction_to(pos_to_idx(position), pos_to_idx(target))
            == target_direction
        ) and ct.get_direction(ct.get_id()) == target_direction:
            ct.fire(target)

        print(f"targeting {target_position}")
        print(f"in direction {target_direction}")
    else:
        print("no targets, or bugged :P")
