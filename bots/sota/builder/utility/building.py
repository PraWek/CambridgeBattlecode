from botlib import MAP_INFO, UNIT_INFO, pos_to_idx, ALLY_BUILDINGS
from botlib.constants import POSITION_CACHE, ENTITY_COST_MAP

from cambc import Controller, Position, Direction, EntityType


def direction_to(start: int | Position, end: int | Position, diagonals_only=False):
    if isinstance(start, int):
        start = POSITION_CACHE[start]

    if isinstance(end, int):
        end = POSITION_CACHE[end]

    if diagonals_only:
        dx = end.x - start.x
        dy = end.y - start.y
        if dx >= 0 and dy >= 0:
            return Direction.SOUTHEAST
        elif dx < 0 and dy >= 0:
            return Direction.SOUTHWEST
        elif dx < 0 and dy < 0:
            return Direction.NORTHWEST
        else:  # dx >= 0 and dy < 0
            return Direction.NORTHEAST
    else:
        return start.direction_to(end)


def can_afford(ct: Controller, entity_type: EntityType):
    """
    Checks if you can afford a given building
    """

    ti, ax = ct.get_global_resources()
    ti_cost, ax_cost = ENTITY_COST_MAP[entity_type]()
    return (ti - ti_cost) >= 0 and (ax - ax_cost) >= 0


def can_build(
    ct: Controller,
    entity_type: EntityType,
    tile: int | Position,
    extra: Direction | int | None = None,
):
    """
    Checks if you can perform the given building action
    """

    if isinstance(extra, int):
        extra = POSITION_CACHE[extra]
    if isinstance(tile, int):
        tile = POSITION_CACHE[tile]
    return ct.can_build(entity_type, tile, extra)


def build_if_can(
    ct: Controller,
    entity_type: EntityType,
    tile: int | Position,
    extra: Direction | Position | int | None = None,
) -> bool:
    """
    Builds the requested building if it can (does not check price)
    Returns true on successful build, false if not
    """

    if isinstance(tile, int):
        tile = POSITION_CACHE[tile]

    if isinstance(extra, int):
        extra = POSITION_CACHE[extra]

    if can_build(ct, entity_type, tile, extra):
        ct.build(entity_type, tile, extra)
        return True
    return False


def build_if_can_afford(
    ct: Controller,
    entity_type: EntityType,
    tile: Position | int,
    extra: Direction | int | None = None,
) -> bool:
    """
    Builds the requested building if it can, checking the price
    Returns true on successful build, false if not
    """

    if isinstance(tile, int):
        tile = POSITION_CACHE[tile]

    if can_afford(ct, entity_type) and can_build(ct, entity_type, tile, extra):
        ct.build(entity_type, tile, extra)
        return True
    return False


def destroy(ct: Controller, tile: Position | int):

    if isinstance(tile, int):
        idx = tile
        tile = POSITION_CACHE[tile]

    if isinstance(tile, Position):
        idx = pos_to_idx(tile)

    if ALLY_BUILDINGS[idx] == 0:
        return True

    if ct.can_destroy(tile):
        ct.destroy(tile)
        return True
    return False
