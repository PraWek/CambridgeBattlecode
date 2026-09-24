from typing import Callable, Optional

from cambc import Controller, Direction, EntityType, Environment, Position, Team

# State


def get_highest_hierarchy_positions(
    positions: list, entity_types: list, hierarchy: list
) -> list:

    if not positions or not entity_types or not hierarchy:
        return []

    if len(positions) != len(entity_types):
        print("borked positions and entity types")
        return []

    rank_map = {entity: index for index, entity in enumerate(hierarchy)}

    highest_rank = float("inf")
    best_positions = []
    best_entity_type = None

    for pos, e_type in zip(positions, entity_types):
        if e_type in rank_map:
            current_rank = rank_map[e_type]

            if current_rank < highest_rank:
                highest_rank = current_rank
                best_positions = [pos]
                best_entity_type = e_type

            elif current_rank == highest_rank:
                best_positions.append(pos)

    return best_positions, best_entity_type


def init(ct: Controller) -> None:
    # Run the main core logic
    run(ct)


def run(ct: Controller) -> None:

    position = ct.get_position()

    attackable_tiles = ct.get_attackable_tiles()

    candidate_pos_list = []
    candidate_type_list = []

    # gather positions and types
    for pos in attackable_tiles:
        id = ct.get_tile_building_id(pos)
        builder_id = ct.get_tile_builder_bot_id(pos)

        if builder_id is not None and (ct.get_team(builder_id) != ct.get_team()):
            # if there is an enemy bot, add to list and continue
            candidate_pos_list.append(pos)
            candidate_type_list.append(EntityType.BUILDER_BOT)
            continue

        if (
            id is not None and (ct.get_team(id) != ct.get_team())
        ) and builder_id is None:
            # check for enemy tile, if there is a builder bot, it must be ours so skip
            candidate_pos_list.append(pos)
            candidate_type_list.append(ct.get_entity_type(id))

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

    if len(candidate_pos_list) == 0:
        print("NO TARGET FOUND")
        return

    candidate_pos_list, type = get_highest_hierarchy_positions(
        candidate_pos_list, candidate_type_list, target_hierachy
    )

    if len(candidate_pos_list) == 1:
        pos = candidate_pos_list[0]
        print(f"firing at: {pos}")
        print(f"of type: {type}")
        if ct.can_fire(pos):
            ct.fire(pos)
            print("fired!")
        else:
            print("cannot fire this turn")

    elif len(candidate_pos_list) > 1:
        best_d2 = 0
        best_pos = None
        for pos in candidate_pos_list:
            x, y = pos.x, pos.y
            x_0, y_0 = position.x, position.y

            d2 = (x - x_0) * (x - x_0) + (y - y_0) * (y - y_0)
            if d2 > best_d2:
                best_d2 = d2
                best_pos = pos

        if best_pos is not None:
            pos = best_pos
            print(f"firing at: {pos}")
            print(f"of type: {type}")
            if ct.can_fire(pos):
                ct.fire(pos)
                print("fired!")
            else:
                print("cannot fire this turn")
