from botlib import MAP_INFO, UNIT_INFO
from botlib.constants import POSITION_CACHE

from cambc import Controller, Direction, EntityType

from ..data import BOT_PATHING


def move_to(ct: Controller, goal_idx: int, solve_already_done: bool = False, forwards: bool = True):
    if not solve_already_done:
        BOT_PATHING.solve(goal_idx, stop_idx=UNIT_INFO.position_idx)
    if BOT_PATHING.ready():
        if forwards:
            next_tile_idx = BOT_PATHING.get_next_tile(UNIT_INFO.position_idx)
        else:
            next_tile_idx = BOT_PATHING.get_prev_tile(UNIT_INFO.position_idx)
        if next_tile_idx is not None:
            return safe_move(
                ct,
                UNIT_INFO.position.direction_to(POSITION_CACHE[next_tile_idx]),
            )
    return False


def safe_move(ct: Controller, direction: Direction | None) -> bool:
    """Moves in a given direction given that it is safe"""
    if direction is None:
        print(f"NO MOVE!")
        return True

    pos = ct.get_position()
    intent = pos.add(direction)
    if intent.x < 0 or intent.y < 0 or intent.x >= MAP_INFO.width or intent.y >= MAP_INFO.height:
        print(f"FAILED MOVED! {direction}")
        return False
    if (
        ct.is_tile_empty(intent)
        or ct.get_entity_type(ct.get_tile_building_id(intent)) == EntityType.MARKER
    ):
        if ct.can_build_road(intent):
            ct.build_road(intent)
    if ct.can_move(direction):
        ct.move(direction)
        print("MOVED!")
        return True
    print(f"FAILED MOVED! {direction}")
    return False
