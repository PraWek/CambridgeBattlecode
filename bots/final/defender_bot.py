"""Keep the core alive while the forward batteries are being established."""

from cambc import EntityType


class DefenderBot:
    def __init__(self, width, height):
        self.core = None

    def run(self, controller):
        if self.core is None:
            for entity_id in controller.get_nearby_buildings():
                if controller.get_entity_type(entity_id) == EntityType.CORE and controller.get_team(entity_id) == controller.get_team():
                    self.core = controller.get_position(entity_id)
                    break
        current = controller.get_position()
        if self.core is not None and controller.can_heal(current):
            controller.heal(current)
