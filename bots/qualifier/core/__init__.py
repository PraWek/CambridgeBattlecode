"""
Core AI

NOTE(randomuserhi): Code is optimized to generate efficient byte code for CPython, not for maintainability
"""

from typing import Callable, Optional

from cambc import Controller, Direction, EntityType, Environment, Position, Team

from botlib import botlib_init, botlib_update, vision_iter, get_bot_id
from botlib import BITBOARD_ENEMY, ENEMY_BUILDER_BOT

# State
CORE_MAX_HP = 500

NUM_TO_SPAWN = 3

FIRST_HIT = True

SEEN_BOTS = set()


def init(ct: Controller) -> None:
    botlib_init(ct, False)
    # Run the main core logic
    run(ct)


def run(ct: Controller) -> None:
    botlib_update(None)

    global NUM_TO_SPAWN
    global FIRST_HIT

    position = ct.get_position()
    offsetPos = position.add(Direction.NORTH)

    # if ct.get_team() == Team.A:
    # return

    if ct.can_spawn(position) and NUM_TO_SPAWN > 0:
        ct.spawn_builder(position)
        NUM_TO_SPAWN -= 1

    if ct.get_current_round() > 75:
        ti, ax = ct.get_global_resources()
        bti, _ = ct.get_builder_bot_cost()
        print(f"{bti} {0.05 * ti} {ti}")
        if ct.can_spawn(position) and bti < 0.05 * ti:
            ct.spawn_builder(position)

        for idx in vision_iter():
            if (BITBOARD_ENEMY[idx] & 0b0010) != 0:
                bot_id = get_bot_id(ENEMY_BUILDER_BOT[idx])
                if bot_id not in SEEN_BOTS:
                    if ct.can_spawn(offsetPos):
                        ct.spawn_builder(offsetPos)
                        SEEN_BOTS.add(bot_id)

    if CORE_MAX_HP != ct.get_hp() and FIRST_HIT:
        if ct.can_spawn(offsetPos):
            FIRST_HIT = False
            ct.spawn_builder(offsetPos)
            print("hit, spawning defender")
