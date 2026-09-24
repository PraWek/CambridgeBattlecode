from cambc import Controller, Direction, EntityType, Position

from botlib import State
from .. import STATE

from botlib import MAP_INFO, UNIT_INFO, INCOME_INFO, BITBOARD_GRID
from botlib import best_enemy_core_idx

from botlib.constants import POSITION_CACHE, DIRECTION_CACHE

from ..utility.movement import move_to

from ..data import BOT_PATHING, DISTANCE_FIELD

# ------------- STATES -------------


from .fundamentals import path_to, fog_max

from . import master_econ, master_rush, master_def, master_generic

import random


class master:
    class state:
        STATE_ID: int = -1

        def __init__(self):
            pass

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

        if ct.get_current_round() < 3:
            # First bots is always a rush bot
            STATE.switch_to(ct, rush_master.state(self))
        elif ct.get_current_round() < 10:
            STATE.switch_to(
                ct,
                generic_master.state(self, local=20, follow_lines_only=False),
            )
        elif UNIT_INFO.position_idx == UNIT_INFO.ally_core_idx:
            score = random.random()
            if score < 0.5:
                STATE.switch_to(
                    ct, rush_master.state(self, local=5, follow_lines_only=(random.random() > 0.5))
                )
            else:
                # STATE.switch_to(ct, def_master.state(self))
                STATE.switch_to(
                    ct, generic_master.state(self, follow_lines_only=(random.random() > 0.5))
                )
        else:
            # If core spawned us off center then it must be to defend something
            STATE.switch_to(
                ct,
                generic_master.state(
                    self,
                    econ=False,
                    follow_lines_only=False,
                    local=5,
                    turns_till_attack=None if (random.random() > 0.5) else 50,
                ),
            )

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state


STATE.register(master)


class rush_master:
    class state:
        STATE_ID: int = -1

        def __init__(self, exit_state, local=None, follow_lines_only: bool = False):
            self.exit_state = exit_state
            self.follow_lines_only = follow_lines_only
            self.local = local
            self.fm = None

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

        self.fm = fog_max.state(self, bail=rush_master.bail)
        STATE.switch_to(
            ct,
            path_to.state(
                best_enemy_core_idx(),
                16,
                self.fm,
                check_reachability=False,
                bail=rush_master.path_bail,
            ),
        )

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state

    @classmethod
    def path_bail(cls, ct: Controller, path: path_to.state):
        self = cls.current_state

        path.target_idx = best_enemy_core_idx()
        return master_rush.rush_bail(ct, self.fm)

    @classmethod
    def bail(cls, ct: Controller, fog_max: fog_max.state):
        self = cls.current_state

        mask = 0

        lines = (
            BITBOARD_GRID.enemy[7]
            | BITBOARD_GRID.enemy[8]
            | BITBOARD_GRID.enemy[9]
            | BITBOARD_GRID.enemy[10]
        )
        for _ in range(2):
            lines |= (
                (lines << 1) | (lines >> 1) | (lines << MAP_INFO.width) | (lines >> MAP_INFO.width)
            )
        lines &= DISTANCE_FIELD.full_mask
        if lines != 0 and self.follow_lines_only:
            mask |= lines

        if self.local is not None:
            DISTANCE_FIELD.solver.use_buildings = False
            DISTANCE_FIELD.solver.use_enemy_core = False
            DISTANCE_FIELD.solve(best_enemy_core_idx(), stop_cost=self.local)
            DISTANCE_FIELD.solver.use_buildings = True
            DISTANCE_FIELD.solver.use_enemy_core = True
            if DISTANCE_FIELD.ready():
                mask |= DISTANCE_FIELD.visited

        if mask != 0:
            fog_max.mask = ~mask
        else:
            fog_max.mask = 0

        return master_rush.rush_bail(ct, fog_max)


STATE.register(rush_master)


class def_master:
    class state:
        STATE_ID: int = -1

        def __init__(self, exit_state):
            self.exit_state = exit_state

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

        STATE.switch_to(ct, fog_max.state(self, bail=def_master.bail))

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state

    @classmethod
    def bail(cls, ct: Controller, fog_max: fog_max.state):
        self = cls.current_state

        DISTANCE_FIELD.solver.use_buildings = False
        DISTANCE_FIELD.solve(UNIT_INFO.ally_core_idx, stop_cost=10)
        DISTANCE_FIELD.solver.use_buildings = True
        if DISTANCE_FIELD.ready():
            fog_max.mask = ~DISTANCE_FIELD.visited

        return master_def.def_bail(ct, fog_max)


STATE.register(def_master)


class econ_master:
    class state:
        STATE_ID: int = -1

        def __init__(self, exit_state, local: int | None = None):
            self.exit_state = exit_state
            self.local = local

            self.init = False

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        seed_idx = None
        if not self.init:
            DISTANCE_FIELD.solver.use_buildings = False
            DISTANCE_FIELD.solve(UNIT_INFO.ally_core_idx, stop_cost=5)
            DISTANCE_FIELD.solver.use_buildings = True
            if DISTANCE_FIELD.ready():
                self.init = True
                if DISTANCE_FIELD.archive_size > 0:
                    bits: int = DISTANCE_FIELD.frontier_archives[0][DISTANCE_FIELD.archive_size - 1]
                    k = random.randint(0, bits.bit_count() - 1)
                    for _ in range(k):
                        bits &= bits - 1
                    seed_idx = (bits & -bits).bit_length() - 1

        STATE.switch_to(ct, fog_max.state(self, bail=econ_master.bail, seed_idx=seed_idx))

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state

    @classmethod
    def bail(cls, ct: Controller, fog_max: fog_max.state):
        self = cls.current_state

        if self.local is not None:
            DISTANCE_FIELD.solver.use_buildings = False
            DISTANCE_FIELD.solve(UNIT_INFO.ally_core_idx, stop_cost=self.local)
            DISTANCE_FIELD.solver.use_buildings = True
            if DISTANCE_FIELD.ready():
                fog_max.mask = ~DISTANCE_FIELD.visited
        else:
            fog_max.mask = 0

        return master_econ.econ_bail(ct, self, fog_max)


STATE.register(econ_master)


class generic_master:
    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            exit_state,
            local: int | None = None,
            econ: bool = True,
            follow_lines_only: bool = False,
            turns_till_attack=None,
        ):
            self.exit_state = exit_state
            self.local = local
            self.econ = econ
            self.follow_lines_only = follow_lines_only

            self.init = False

            self.visited = 0

            self.turns_till_attack = turns_till_attack

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        seed_idx = None
        if not self.init:
            DISTANCE_FIELD.solver.use_buildings = False
            DISTANCE_FIELD.solve(UNIT_INFO.ally_core_idx, stop_cost=5)
            DISTANCE_FIELD.solver.use_buildings = True
            if DISTANCE_FIELD.ready():
                self.init = True
                if DISTANCE_FIELD.archive_size > 0:
                    bits: int = DISTANCE_FIELD.frontier_archives[0][DISTANCE_FIELD.archive_size - 1]
                    k = random.randint(0, bits.bit_count() - 1)
                    for _ in range(k):
                        bits &= bits - 1
                    seed_idx = (bits & -bits).bit_length() - 1

        STATE.switch_to(ct, fog_max.state(self, bail=generic_master.bail, seed_idx=seed_idx))

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state

    @classmethod
    def bail(cls, ct: Controller, fog_max: fog_max.state):
        self = cls.current_state

        if self.turns_till_attack is not None and STATE.tick > self.turns_till_attack:
            STATE.switch_to(
                ct, rush_master.state(self, local=5, follow_lines_only=(random.random() > 0.5))
            )
            return

        self.visited |= UNIT_INFO.vision_mask

        mask = 0

        lines = (
            BITBOARD_GRID.ally[7]
            | BITBOARD_GRID.ally[8]
            | BITBOARD_GRID.ally[9]
            | BITBOARD_GRID.ally[10]
        )
        for _ in range(2):
            lines |= (
                (lines << 1) | (lines >> 1) | (lines << MAP_INFO.width) | (lines >> MAP_INFO.width)
            )
        lines &= DISTANCE_FIELD.full_mask
        if lines != 0 and self.follow_lines_only:
            mask |= lines

        if self.local is not None:
            DISTANCE_FIELD.solver.use_buildings = False
            DISTANCE_FIELD.solve(UNIT_INFO.ally_core_idx, stop_cost=self.local)
            DISTANCE_FIELD.solver.use_buildings = True
            if DISTANCE_FIELD.ready():
                mask |= DISTANCE_FIELD.visited

        if mask != 0:
            fog_max.mask = ~mask
        else:
            fog_max.mask = 0

        return master_generic.generic_bail(ct, self, fog_max)


STATE.register(generic_master)
