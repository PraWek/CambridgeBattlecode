from cambc import Controller, EntityType

from base import BaseBot
from builder_bot import BuilderBot
from core_bot import CoreBot
from gunner_bot import GunnerBot

class Player:
    def __init__(self) -> None:
        """Initialize the lazy role-specific bot holder for this game entity."""
        self.initialized = False
        self.bot: BaseBot | None = None

    def run(self, c: Controller) -> None:
        """Initialize the entity role once and delegate its current turn."""
        self.init_once(c)
        if self.bot is not None:
            self.bot.run(c)

    def init_once(self, c: Controller) -> None:
        """Create the bot implementation corresponding to this entity's type."""
        if self.initialized:
            return
        self.initialized = True

        entity_type: EntityType = c.get_entity_type()
        if entity_type == EntityType.CORE:
            self.bot = CoreBot(c.get_map_width(), c.get_map_height())
        elif entity_type == EntityType.BUILDER_BOT:
            self.bot = BuilderBot(c.get_map_width(), c.get_map_height())
        elif entity_type == EntityType.GUNNER:
            self.bot = GunnerBot(c.get_map_width(), c.get_map_height())
