from cambc import Controller, EntityType, Environment, Position, Team

from constants import PASSABLE_BUILDINGS
from tile_cache import TileCache


class BaseBot:

    def __init__(self, map_width: int, map_height: int) -> None:
        """Store immutable map dimensions shared by every bot role."""
        self.map_width = map_width
        self.map_height = map_height
        self.tile_cache = TileCache(map_width, map_height)
        self.entity_id: int | None = None
        self.team: Team | None = None
        self.current_position: Position | None = None

        self.max_cpu_cost = 0
        self.rolling_avg_cpu_cost = 0

    def _scan_turn(
            self,
            controller: Controller,
            read_markers: bool = False,
    ) -> bool:
        """Fill the cache and report when the role must yield this turn."""
        if self.entity_id is None:
            self.entity_id = controller.get_id()
        self.tile_cache.scan_turn(controller, self.entity_id)
        if self.tile_cache.scan_incomplete_this_turn:
            return True
        self.current_position = self.tile_cache.current_position
        if self.team is None:
            self.team = self.tile_cache.entity_team(self.entity_id)
            if self.team is None:
                self.team = controller.get_team()
        if read_markers:
            if self.tile_cache.symmetry_confirmed_this_turn:
                return True
            if not self.tile_cache.cache_friendly_marker_values(controller, self.team):
                return True
        return self.tile_cache.symmetry_confirmed_this_turn

    def get_cached_position(self) -> Position:
        """Return this entity's start-of-turn position without an API call."""
        if self.current_position is None:
            raise RuntimeError("tile cache was not scanned before reading position")
        return self.current_position

    def is_cached_tile_passable(self, pos: Position) -> bool:
        """Return whether the latest cached tile state permits a builder step."""
        if self.tile_cache.environment_at(pos) == Environment.WALL:
            return False
        building = self.tile_cache.building_at(pos)
        if building is None:
            return False
        building_type, building_team = building
        if building_type not in PASSABLE_BUILDINGS:
            return False
        if building_type == EntityType.CORE and building_team != self.team:
            return False
        return self.tile_cache.builder_id_at(pos) is None

    def run(self, c: Controller) -> None:
        """Execute common per-turn accounting; subclasses extend this behavior."""
        # Log CPU costs
        cpu_cost = c.get_cpu_time_elapsed()
        if cpu_cost > self.max_cpu_cost:
            self.max_cpu_cost = cpu_cost
        self.rolling_avg_cpu_cost = (self.rolling_avg_cpu_cost * 39 + cpu_cost) / 40
        print(f"[{self}] avg cpu: {self.rolling_avg_cpu_cost}")
        print(f"[{self}] max cpu: {self.max_cpu_cost}")

    def in_bounds(self, pos: Position) -> bool:
        """Return whether ``pos`` lies inside the current map."""
        return 0 <= pos.x < self.map_width and 0 <= pos.y < self.map_height
