"""Keep the core alive while the forward batteries are being established."""

from cambc import EntityType
from constants import DIRECTIONS
from tile_cache import TileCache


class DefenderBot:
    def __init__(self, width, height):
        self.core = None
        self.cache = TileCache(width, height)
        self.visits = {}

    def run(self, controller):
        if self.core is None:
            for entity_id in controller.get_nearby_buildings():
                if controller.get_entity_type(entity_id) == EntityType.CORE and controller.get_team(entity_id) == controller.get_team():
                    self.core = controller.get_position(entity_id)
                    break
        current = controller.get_position()
        if self.core is None:
            return
        if self.shield_core(controller, current):
            return
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tile = self.cache.offset(self.core, dx, dy)
                if tile is not None and current.distance_squared(tile) <= 2 and controller.can_heal(tile):
                    controller.heal(tile)
                    return
        if max(abs(current.x-self.core.x), abs(current.y-self.core.y)) <= 1:
            return
        # A hostile launcher does not change our assignment. Walk back to a
        # repair position; repeated visits keep a wall detour from cycling.
        self.visits[current] = self.visits.get(current, 0) + 1
        steps = []
        for direction in DIRECTIONS:
            tile = self.cache.neighbor(current, direction)
            if tile is not None:
                steps.append((tile.distance_squared(self.core) + 8*self.visits.get(tile, 0), direction, tile))
        for _, direction, tile in sorted(steps, key=lambda v: v[0]):
            if controller.can_move(direction):
                controller.move(direction)
                return
            if controller.can_build_road(tile):
                controller.build_road(tile)
                if controller.can_move(direction):
                    controller.move(direction)
                return

    def shield_core(self, controller, current):
        """Put a replaceable barrier in a gunner ray before it reaches core."""
        team = controller.get_team()
        for entity_id in controller.get_nearby_units():
            if controller.get_team(entity_id) == team or controller.get_entity_type(entity_id) != EntityType.GUNNER:
                continue
            origin = controller.get_position(entity_id)
            facing = controller.get_direction(entity_id)
            pos = origin
            ray = []
            for _ in range(3):
                pos = self.cache.neighbor(pos, facing)
                if pos is None or origin.distance_squared(pos) > 13:
                    break
                if max(abs(pos.x-self.core.x), abs(pos.y-self.core.y)) <= 1:
                    for shield in reversed(ray):
                        if shield == current or current.distance_squared(shield) > 2:
                            continue
                        old = controller.get_tile_building_id(shield)
                        if old is not None:
                            if controller.get_team(old) != team:
                                continue
                            kind = controller.get_entity_type(old)
                            if kind == EntityType.BARRIER:
                                if controller.can_heal(shield):
                                    controller.heal(shield)
                                    return True
                                continue
                            if kind != EntityType.ROAD or not controller.can_destroy(shield):
                                continue
                            controller.destroy(shield)
                        if controller.can_build_barrier(shield):
                            controller.build_barrier(shield)
                            return True
                    break
                ray.append(pos)
        return False
