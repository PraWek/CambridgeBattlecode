"""Launcher behaviour used by an IntruderBot to cross a blocked route."""

from cambc import Controller, EntityType, Position

from base import BaseBot
from constants import DIRECTIONS, MARKER_KIND_INTRUDER_LAUNCH
from geometry import decode_marker


class LauncherBot(BaseBot):
    """Launch the adjacent intruder to the cached landing tile in its order marker."""

    def run(self, controller: Controller) -> None:
        """Read one launch order, throw its adjacent friendly builder, then clear it."""
        # Combat launchers must react immediately, including their first turn.
        # No terrain/symmetry scan is necessary for a legal throw.
        if self.throw_enemy(controller):
            return
        if self._scan_turn(controller, read_markers=True):
            return

        landing, marker_pos, source = self.read_launch_order()
        if landing is None or marker_pos is None or source is None:
            return
        for bot_pos, bot_id in self.tile_cache.visible_builder_ids.items():
            if bot_pos != source:
                continue
            if self.tile_cache.entity_team(bot_id) != self.team:
                continue
            if max(abs(bot_pos.x - self.get_cached_position().x), abs(bot_pos.y - self.get_cached_position().y)) > 1:
                continue
            if not controller.can_launch(bot_pos, landing):
                continue
            controller.launch(bot_pos, landing)
            self.clear_launch_order(controller, marker_pos)
            return

    def throw_enemy(self, controller: Controller) -> bool:
        origin = controller.get_position()
        team = controller.get_team()
        for direction in DIRECTIONS:
            source = origin.add(direction)
            if not self.in_bounds(source):
                continue
            entity_id = controller.get_tile_builder_bot_id(source)
            if entity_id is None or controller.get_team(entity_id) == team:
                continue
            targets = [self.tile_cache.offset(origin, dx, dy)
                       for dx in range(-5, 6) for dy in range(-5, 6)
                       if dx * dx + dy * dy <= 26]
            targets = [pos for pos in targets if pos is not None]
            targets.sort(key=lambda pos: source.distance_squared(pos), reverse=True)
            for target in targets:
                if controller.can_launch(source, target):
                    controller.launch(source, target)
                    return True
        return False

    def read_launch_order(self) -> tuple[Position | None, Position | None, Position | None]:
        """Return the landing tile and marker tile from a visible intruder order."""
        for marker_id in self.tile_cache.marker_ids():
            value = self.tile_cache.marker_values.get(marker_id)
            marker_pos = self.tile_cache.entity_position(marker_id)
            if value is None or marker_pos is None:
                continue
            try:
                kind, landing, source_code = decode_marker(
                    value,
                    self.tile_cache.position_at,
                )
            except Exception:
                continue
            if kind == MARKER_KIND_INTRUDER_LAUNCH and 1 <= source_code <= len(DIRECTIONS):
                source = self.tile_cache.neighbor(self.get_cached_position(), DIRECTIONS[source_code - 1])
                return landing, marker_pos, source
        return None, None, None

    def clear_launch_order(self, controller: Controller, marker_pos: Position) -> None:
        """Remove a completed friendly order so this launcher cannot repeat it."""
        if not controller.can_destroy(marker_pos):
            return
        controller.destroy(marker_pos)
        self.tile_cache.forget_building(marker_pos)
