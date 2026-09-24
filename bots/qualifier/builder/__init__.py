"""
Builder Bot AI
"""

from cambc import Controller, Team

from botlib import (
    botlib_init,
    botlib_update,
    StateManager,
)

# ------------- STATE -------------

STATE = StateManager()

# ------------- MAIN LOGIC -------------

from .states.mastermind import master

from botlib import vision_iter, BITBOARD_ENV, UNIT_INFO, INCOME_INFO

from .data import ORES_COMPLETED, BOT_PATHING, RESOURCE_PATHING, DISTANCE_FIELD


def init(ct: Controller) -> None:
    # Init map manager
    botlib_init(ct)

    # pad bot solves to get extra information on surrounding tiles for
    # backup movement when stuck
    BOT_PATHING.init(default_inflate_value=9)
    RESOURCE_PATHING.init()
    DISTANCE_FIELD.init()

    # Run the main bot logic
    run(ct)


def run(ct: Controller) -> None:
    # Update the map information
    # Takes around ~ 0.5-0.7ms (400-700 microsecnds)
    start = ct.get_cpu_time_elapsed()
    botlib_update(STATE)
    print(f"botlib = {ct.get_cpu_time_elapsed() - start}")

    if STATE.current == -1:
        STATE.switch_to(ct, master.state())

    # TODO
    # if STATE.current == -1:
    #     for idx in vision_iter():
    #         if (BITBOARD_ENV[idx] & 0b0100) != 0 and not ORES_COMPLETED[idx]:
    #             STATE.switch_to_and_run(ct, gather_resource.path_to_ore, idx)
    #             return

    print(f"state = {STATE.current_state_name}")
    STATE._run(ct)
