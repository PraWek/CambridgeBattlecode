"""
Launcher AI
"""

from typing import Callable, Optional

from cambc import Controller, Direction, EntityType, Environment, Position, Team

from botlib import botlib_init, botlib_update

from botlib import (
    CARDINAL_DIRECTION_DELTAS,
    DIRECTION_DELTAS,
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
)


def init(ct: Controller) -> None:
    botlib_init(ct, False)

    # Run the main core logic
    run(ct)


def run(ct: Controller) -> None:
    botlib_update(None)

    bot_target = None

    x, y = ct.get_position()
    for dx, dy in DIRECTION_DELTAS:
        nx = x + dx
        ny = y + dy
        idx = xy_to_idx(nx, ny)
        if (BITBOARD_ENEMY[idx] & 0b0010) != 0:
            bot_target = idx
            break

    if bot_target is None:
        return

    bot_pos = idx_to_pos(bot_target)
    destination = None
    distance = None
    for idx in vision_iter():
        dist = idx_to_pos(idx).distance_squared(bot_pos)
        if ct.can_launch(bot_pos, idx_to_pos(idx)) and (distance is None or distance < dist):
            distance = dist
            destination = idx_to_pos(idx)

    if destination is not None:
        ct.launch(bot_pos, destination)
