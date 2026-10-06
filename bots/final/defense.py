"""Local anti-infiltration actions shared by economy builders."""

from cambc import EntityType, GameConstants
from constants import DIRECTIONS, ORTHOGONAL_DIRECTIONS

TURRETS = {EntityType.GUNNER, EntityType.SENTINEL, EntityType.BREACH}


def defend_economy(bot, controller, current):
    cache = bot.tile_cache
    # Repairing the core wins time without discarding the active conveyor job.
    core = bot.core_pos
    if core is not None and current.distance_squared(core) <= 8:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tile = cache.offset(core, dx, dy)
                if tile is not None and controller.can_heal(tile):
                    controller.heal(tile)
                    bot.last_progress_round = bot.rounds_alive
                    return True
    # Use a second output of the threatened mine for counter-battery fire.
    for entity_id in cache.visible_entity_ids:
        if cache.entity_team(entity_id) == bot.team or cache.entity_type(entity_id) not in TURRETS:
            continue
        turret = cache.entity_position(entity_id)
        for direction in ORTHOGONAL_DIRECTIONS:
            ore = cache.neighbor(turret, direction)
            building = cache.building_at(ore) if ore is not None else None
            if building != (EntityType.HARVESTER, bot.team):
                continue
            for output in ORTHOGONAL_DIRECTIONS:
                site = cache.neighbor(ore, output)
                if site is None or current.distance_squared(site) > 2:
                    continue
                facing = site.direction_to(turret)
                for kind in (EntityType.GUNNER, EntityType.SENTINEL):
                    if site.add(facing) == ore:
                        continue
                    if not controller.can_fire_from(site, facing, kind, turret):
                        continue
                    if cache.building_at(site) == (EntityType.ROAD, bot.team) and controller.can_destroy(site):
                        controller.destroy(site)
                        cache.forget_building(site)
                    if controller.can_build(kind, site, facing) and controller.can_fire_from(site, facing, kind, turret):
                        entity_id = controller.build(kind, site, facing)
                        cache.remember_building(site, entity_id, kind, bot.team, direction=facing)
                        return True

    enemies = [pos for pos, entity_id in cache.visible_builder_ids.items()
               if cache.entity_team(entity_id) != bot.team
               and current.distance_squared(pos) <= 8
               and (pos.distance_squared(bot.core_pos) <= 8 or any(
                   pos.distance_squared(ore) <= 8
                   for ore, building in cache.buildings.items()
                   if building == (EntityType.HARVESTER, bot.team)))]
    if not enemies:
        return False
    if controller.get_global_resources()[0] < controller.get_launcher_cost()[0] + 6 * controller.get_conveyor_cost()[0]:
        return False
    launchers = [cache.entity_position(entity_id) for entity_id in cache.visible_entity_ids
                 if cache.entity_team(entity_id) == bot.team
                 and cache.entity_type(entity_id) == EntityType.LAUNCHER]
    enemies = [pos for pos in enemies if not any(pos.distance_squared(p) <= 2 for p in launchers)]
    for enemy in enemies:
        for direction in DIRECTIONS:
            site = cache.neighbor(current, direction)
            if site is None or site.distance_squared(enemy) > 2:
                continue
            if cache.building_at(site) == (EntityType.ROAD, bot.team) and controller.can_destroy(site):
                controller.destroy(site)
                cache.forget_building(site)
            if controller.can_build_launcher(site):
                entity_id = controller.build_launcher(site)
                cache.remember_building(site, entity_id, EntityType.LAUNCHER, bot.team)
                bot.last_progress_round = bot.rounds_alive
                return True
    return False


def ore_feeds_enemy(bot, ore):
    for direction in ORTHOGONAL_DIRECTIONS:
        site = bot.tile_cache.neighbor(ore, direction)
        building = bot.tile_cache.building_at(site) if site is not None else None
        if building and building[1] != bot.team and building[0] in TURRETS:
            return True
    return False
