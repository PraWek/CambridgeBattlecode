"""
Main Bot AI

NOTE(randomuserhi): Code is optimized to generate efficient byte code for CPython, not for maintainability
"""

import tracemalloc

from cambc import Controller, Direction, EntityType, Environment, Position, Team

import builder
import core
import sentinel
import gunner
import launcher

# Maps entity type to its correspoding controller
CONTROLLER_MAP = {
    EntityType.CORE: (core.run, core.init),
    EntityType.BUILDER_BOT: (builder.run, builder.init),
    EntityType.SENTINEL: (sentinel.run, sentinel.init),
    EntityType.GUNNER: (gunner.run, gunner.init),
    EntityType.LAUNCHER: (launcher.run, launcher.init),
}

RUN = None


class Player:
    def run(self, ct: Controller) -> None:
        global RUN

        if RUN is not None:
            RUN(ct)
        else:
            ent_type = ct.get_entity_type()
            if ent_type in CONTROLLER_MAP:
                run, init = CONTROLLER_MAP[ent_type]
                init(ct)
                RUN = run
