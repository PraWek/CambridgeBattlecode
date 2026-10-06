from cambc import Controller, Direction, EntityType, Position

from base import BaseBot
from constants import MARKER_KIND_ENEMY
from geometry import decode_marker


class GunnerBot(BaseBot):
    def run(self, controller: Controller) -> None:
        """Fire at an available target or rotate toward a shared enemy marker."""
        # Firing must not wait for terrain scanning or symmetry inference.
        # can_fire checks geometry, not allegiance; a road or allied builder
        # in the ray must never be mistaken for a target.
        team = controller.get_team()
        candidates = []
        for tile in controller.get_attackable_tiles():
            entity_id = controller.get_tile_builder_bot_id(tile)
            if entity_id is None:
                entity_id = controller.get_tile_building_id(tile)
            if entity_id is None or controller.get_team(entity_id) == team:
                continue
            kind = controller.get_entity_type(entity_id)
            priority = 3 if kind == EntityType.CORE else 2 if kind in (
                EntityType.GUNNER, EntityType.SENTINEL, EntityType.BREACH,
            ) else 1
            if controller.can_fire(tile):
                candidates.append((priority, tile))
        if candidates:
            controller.fire(max(candidates, key=lambda item: item[0])[1])
            return
        if self._scan_turn(controller, read_markers=True):
            return

        # A mine-fed gunner may need to turn from its ammo intake toward a
        # raider, counter-battery or the source it is capturing.
        priorities = {EntityType.CORE: 100, EntityType.GUNNER: 120,
                      EntityType.SENTINEL: 120, EntityType.BREACH: 120,
                      EntityType.BUILDER_BOT: 110, EntityType.HARVESTER: 20,
                      EntityType.LAUNCHER: 70}
        ammo = controller.get_ammo_amount()
        facing = self.tile_cache.entity_direction(self.entity_id)
        if ammo == 0 and facing is not None:
            source = self.tile_cache.neighbor(self.current_position, facing)
            building = self.tile_cache.building_at(source) if source is not None else None
            if building is not None and building[0] == EntityType.HARVESTER:
                # Firing toward a captured mine closes that ammo port. Reopen
                # it after the magazine empties instead of remaining stuck.
                reload_direction = facing.opposite()
                if controller.can_rotate(reload_direction):
                    controller.rotate(reload_direction)
                    return
        targets = []
        for entity_id in self.tile_cache.visible_entity_ids:
            if self.tile_cache.entity_team(entity_id) == team:
                continue
            kind = self.tile_cache.entity_type(entity_id)
            priority = priorities.get(kind, 0)
            pos = self.tile_cache.entity_position(entity_id)
            if not priority or pos is None or self.current_position.distance_squared(pos) > 13:
                continue
            occupant = self.tile_cache.builder_id_at(pos)
            if occupant is not None and self.tile_cache.entity_team(occupant) == team:
                continue
            if kind == EntityType.HARVESTER and ammo < 4:
                continue
            direction = self.current_position.direction_to(pos)
            if ammo > 0 and controller.can_rotate(direction) and controller.can_fire_from(
                    self.current_position, direction, EntityType.GUNNER, pos):
                targets.append((priority, direction))
        if targets:
            controller.rotate(max(targets, key=lambda v: v[0])[1])
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
