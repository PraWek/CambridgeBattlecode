from cambc import Controller, EntityType

from base import BaseBot
from builder_bot import BuilderBot
from core_bot import CoreBot
from gunner_bot import GunnerBot
from intruder_bot import IntruderBot
from launcher_bot import LauncherBot
from spawn_orders import read_spawn_assignment

class Player:
    def __init__(self) -> None:
        """Initialize the lazy role-specific bot holder for this game entity."""
        self.initialized = False
        self.bot: BaseBot | None = None
        self.builder_role = None
        self.builder_direction = None

    def run(self, c: Controller) -> None:
        """Initialize the entity role once and delegate its current turn."""
        self.init_once(c)
        if self.bot is not None:
            self.bot.run(c)

    def init_once(self, c: Controller) -> None:
        """Create the bot implementation corresponding to this entity's type."""
        if self.initialized:
            return
        entity_type: EntityType = c.get_entity_type()
        if entity_type == EntityType.CORE:
            self.bot = CoreBot(c.get_map_width(), c.get_map_height())
        elif entity_type == EntityType.BUILDER_BOT:
            # Remember the handoff before constructing a potentially expensive
            # cache. A CPU interruption must not lose the selected role.
            if self.builder_role is None:
                intruder, self.builder_direction = read_spawn_assignment(c)
                self.builder_role = IntruderBot if intruder else BuilderBot
            self.bot = self.builder_role(c.get_map_width(), c.get_map_height())
            if self.builder_role is BuilderBot and self.builder_direction is not None:
                self.bot.work_direction = self.builder_direction
        elif entity_type == EntityType.GUNNER:
            self.bot = GunnerBot(c.get_map_width(), c.get_map_height())
        elif entity_type == EntityType.LAUNCHER:
            self.bot = LauncherBot(c.get_map_width(), c.get_map_height())
        # Construction can itself be interrupted by the engine's CPU limit.
        # Retry next turn until a complete role object has been assigned.
        self.initialized = True
