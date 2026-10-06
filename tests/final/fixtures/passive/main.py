"""Economic diagnostic only: allow 300 uncontested turns, then stop the replay."""
from cambc import EntityType


class Player:
    def run(self, controller):
        if controller.get_entity_type() == EntityType.CORE and controller.get_current_round() >= 300:
            controller.resign()
