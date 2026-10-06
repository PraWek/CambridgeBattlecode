"""Fire immediately at the most valuable enemy in the sentinel's cone."""

from cambc import EntityType


class SentinelBot:
    def run(self, controller):
        team = controller.get_team()
        targets = []
        priorities = {EntityType.CORE: 100, EntityType.SENTINEL: 120,
                      EntityType.GUNNER: 120, EntityType.LAUNCHER: 60,
                      EntityType.BUILDER_BOT: 110, EntityType.HARVESTER: 10}
        for pos in controller.get_attackable_tiles():
            entity_id = controller.get_tile_builder_bot_id(pos)
            if entity_id is None:
                entity_id = controller.get_tile_building_id(pos)
            if entity_id is not None and controller.get_team(entity_id) != team:
                priority = priorities.get(controller.get_entity_type(entity_id), 0)
                if priority:
                    targets.append((priority, pos))
        targets.sort(key=lambda item: item[0], reverse=True)
        for _, pos in targets:
            if controller.can_fire(pos):
                controller.fire(pos)
                return
