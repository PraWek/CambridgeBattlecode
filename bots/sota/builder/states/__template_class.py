from __future__ import annotations
from typing import Callable

from cambc import Controller, Direction, EntityType, Position

from botlib import State
from .. import STATE

from botlib import MAP_INFO, UNIT_INFO
from botlib import best_enemy_core_idx

from botlib.constants import POSITION_CACHE, DIRECTION_CACHE

from ..utility.movement import move_to

from ..data import BOT_PATHING

# ------------- STATES -------------


class state_class:
    class state:
        STATE_ID: int = -1

        def __init__(
            self, exit_state, bail: Callable[[Controller, state_class.state], bool] | None = None
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


STATE.register(state_class)
