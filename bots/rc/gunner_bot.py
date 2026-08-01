from cambc import Controller, Direction, EntityType, Position

from base import BaseBot
from constants import MARKER_KIND_ENEMY
from geometry import decode_marker


class GunnerBot(BaseBot):
    def run(self, controller: Controller) -> None:
        """Fire at an available target or rotate toward a shared enemy marker."""
        target = controller.get_gunner_target()
        if target is not None and controller.can_fire(target):
            controller.fire(target)
            return

        marker_target = self.read_enemy_marker(controller)
        if marker_target is None:
            return
        desired = controller.get_position().direction_to(marker_target)
        if desired != Direction.CENTRE and desired != controller.get_direction() and controller.can_rotate(desired):
            controller.rotate(desired)

    def read_enemy_marker(self, controller: Controller) -> Position | None:
        """Return the first nearby marker that contains an enemy position."""
        for entity_id in controller.get_nearby_entities():
            if controller.get_entity_type(entity_id) != EntityType.MARKER:
                continue
            try:
                kind, pos, _ = decode_marker(controller.get_marker_value(entity_id))
            except Exception:
                continue
            if kind == MARKER_KIND_ENEMY:
                return pos
        return None
