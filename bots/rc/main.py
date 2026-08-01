from cambc import Controller, EntityType

from builder_bot import BuilderBot
from core_bot import CoreBot

class Player:
    def __init__(self) -> None:
        self.initialized = False
        self.bot: CoreBot | BuilderBot | None = None

    def run(self, c: Controller) -> None:
        self.init_once(c)
        self.bot.run(c)

    def init_once(self, c: Controller) -> None:
        if self.initialized:
            return
        self.initialized = True

        entity_type: EntityType = c.get_entity_type()
        if entity_type == EntityType.CORE:
            self.bot = CoreBot(c.get_map_width(), c.get_map_height())
        elif entity_type == EntityType.BUILDER_BOT:
            self.bot = BuilderBot(c.get_map_width(), c.get_map_height())