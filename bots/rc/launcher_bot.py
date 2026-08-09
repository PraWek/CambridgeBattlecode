"""Launcher behaviour used by an IntruderBot to cross a blocked route."""

from cambc import Controller, EntityType, Position

from base import BaseBot
from constants import MARKER_KIND_INTRUDER_LAUNCH
from geometry import decode_marker


class LauncherBot(BaseBot):
    """Launch the adjacent intruder to the cached landing tile in its order marker."""

    def run(self, controller: Controller) -> None:
        """Read one launch order, throw its adjacent friendly builder, then clear it."""
        self._scan_turn(controller, read_markers=True)
        landing, marker_pos = self.read_launch_order()
        if landing is None or marker_pos is None:
            return
        for bot_pos, bot_id in self.tile_cache.visible_builder_ids.items():
            if self.tile_cache.entity_team(bot_id) != self.team:
                continue
            if max(abs(bot_pos.x - self.get_cached_position().x), abs(bot_pos.y - self.get_cached_position().y)) > 1:
                continue
            if not controller.can_launch(bot_pos, landing):
                continue
            controller.launch(bot_pos, landing)
            self.clear_launch_order(controller, marker_pos)
            return

    def read_launch_order(self) -> tuple[Position | None, Position | None]:
        """Return the landing tile and marker tile from a visible intruder order."""
        for marker_id in self.tile_cache.marker_ids():
            value = self.tile_cache.marker_values.get(marker_id)
            marker_pos = self.tile_cache.entity_position(marker_id)
            if value is None or marker_pos is None:
                continue
            try:
                kind, landing, _ = decode_marker(value)
            except Exception:
                continue
            if kind == MARKER_KIND_INTRUDER_LAUNCH:
                return landing, marker_pos
        return None, None

    def clear_launch_order(self, controller: Controller, marker_pos: Position) -> None:
        """Remove a completed friendly order so this launcher cannot repeat it."""
        if not controller.can_destroy(marker_pos):
            return
        controller.destroy(marker_pos)
        self.tile_cache.forget_building(marker_pos)
