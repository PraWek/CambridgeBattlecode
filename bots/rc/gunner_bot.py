from cambc import Controller, Direction, Position

from base import BaseBot
from constants import MARKER_KIND_ENEMY
from geometry import decode_marker


class GunnerBot(BaseBot):
    def run(self, controller: Controller) -> None:
        """Fire at an available target or rotate toward a shared enemy marker."""
        if self._scan_turn(controller, read_markers=True):
            return
        target = controller.get_gunner_target()
        if target is not None:
            target = self.tile_cache.canonicalize(target)
        if target is not None and controller.can_fire(target):
            controller.fire(target)
            return

        marker_target = self.read_enemy_marker()
        if marker_target is None:
            return
        desired = self.get_cached_position().direction_to(marker_target)
        current_direction = (
            None if self.entity_id is None
            else self.tile_cache.entity_direction(self.entity_id)
        )
        if desired != Direction.CENTRE and desired != current_direction and controller.can_rotate(desired):
            controller.rotate(desired)

    def read_enemy_marker(self) -> Position | None:
        """Return the first nearby marker that contains an enemy position."""
        for entity_id in self.tile_cache.marker_ids():
            marker_value = self.tile_cache.marker_values.get(entity_id)
            if marker_value is None:
                continue
            try:
                kind, pos, _ = decode_marker(
                    marker_value,
                    self.tile_cache.position_at,
                )
            except Exception:
                continue
            if kind == MARKER_KIND_ENEMY:
                return pos
        return None
