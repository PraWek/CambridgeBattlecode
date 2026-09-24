"""
Core AI
"""

from typing import Callable, Optional

from cambc import Controller, Direction, EntityType, Environment, Position, Team

from botlib import botlib_init, botlib_update, vision_iter, get_bot_id, pos_to_idx
from botlib import BITBOARD_ENEMY, ENEMY_BUILDER_BOT, HAZARDS

# State
CORE_PREV_HP = 500

TICK = 0

NUM_DEFENDER = 1
NUM_ATTACKER = 2
NUM_ECON = 1

FIRST_HIT = 0

SEEN_BOTS = set()


def init(ct: Controller) -> None:
    botlib_init(ct, False)
    # Run the main core logic
    run(ct)


def run(ct: Controller) -> None:
    botlib_update(None)

    global TICK, NUM_ATTACKER, NUM_DEFENDER, NUM_GENERIC, NUM_ECON, CORE_PREV_HP
    global FIRST_HIT

    TICK += 1

    position = ct.get_position()

    offsetPos = [
        position.add(Direction.NORTH),  # ECON
        position.add(Direction.SOUTH),  # DEFENDER
        position.add(Direction.EAST),  # ATTACKER
    ]

    # If offset pos is covered by launcher, we cannot use it - set it to safe position
    for i in range(len(offsetPos)):
        if HAZARDS[6][pos_to_idx(offsetPos[i])] > 0:
            offsetPos[i] = position

    if ct.can_spawn(offsetPos[2]) and NUM_ATTACKER > 0:
        ct.spawn_builder(offsetPos[2])
        NUM_ATTACKER -= 1
    elif ct.can_spawn(offsetPos[1]) and NUM_DEFENDER > 0:
        ct.spawn_builder(offsetPos[1])
        NUM_DEFENDER -= 1
    elif ct.can_spawn(offsetPos[0]) and NUM_ECON > 0:
        ct.spawn_builder(offsetPos[0])
        NUM_ECON -= 1

    if ct.get_current_round() > 75:  # 75
        ti, ax = ct.get_global_resources()
        bti, _ = ct.get_builder_bot_cost()
        if ct.can_spawn(position) and bti < 0.25 * ti:
            ct.spawn_builder(position)

        for idx in vision_iter():
            if (BITBOARD_ENEMY[idx] & 0b0010) != 0:
                bot_id = get_bot_id(ENEMY_BUILDER_BOT[idx])
                if bot_id not in SEEN_BOTS:
                    if ct.can_spawn(offsetPos[1]):
                        ct.spawn_builder(offsetPos[1])
                        SEEN_BOTS.add(bot_id)

    if CORE_PREV_HP != ct.get_hp() and TICK > FIRST_HIT:
        if ct.can_spawn(offsetPos[1]):
            FIRST_HIT = TICK + 50  # 50 round cooldown
            ct.spawn_builder(offsetPos[1])
            print("hit, spawning defender")
            CORE_PREV_HP = ct.get_hp()
