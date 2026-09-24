"""Exploit local ammunition sources before committing to a long supply route."""

from cambc import EntityType, Environment, GameConstants
from constants import DIRECTIONS, ORTHOGONAL_DIRECTIONS


def execute_plan(bot, controller, current, plan):
    site, ore, kind, facing, feed = plan
    cache = bot.tile_cache
    mine = cache.building_at(ore)
    if feed != site and cache.building_at(feed) != (EntityType.BRIDGE, bot.team):
        if current == feed or current.distance_squared(feed) > 2:
            bot.advance_towards_revisitable_target(controller, bot.construction_approach(feed))
            return True
        old = cache.building_at(feed)
        if old is not None:
            if old != (EntityType.ROAD, bot.team):
                bot.siege_plan = None
                return False
            if controller.can_destroy(feed):
                controller.destroy(feed)
                cache.forget_building(feed)
        if controller.can_build_bridge(feed, site):
            entity_id = controller.build_bridge(feed, site)
            cache.remember_building(feed, entity_id, EntityType.BRIDGE, bot.team)
        return True
    building = cache.building_at(site)
    if building is not None and building[0] != EntityType.ROAD:
        bot.siege_plan = None
        return False
    if current == site or current.distance_squared(site) > 2:
        approaches = [cache.neighbor(site, d) for d in DIRECTIONS]
        approaches = [p for p in approaches if p is not None and bot.is_roadable_position(p)
                      ]
        if approaches:
            approach = min(approaches, key=lambda p: current.distance_squared(p))
            bot.advance_towards_revisitable_target(controller, approach)
            return True
        bot.siege_plan = None
        return False
    if building is not None:
        if building[1] != bot.team:
            bot.siege_plan = None
            return False
        if controller.can_destroy(site):
            controller.destroy(site)
            cache.forget_building(site)
    if controller.can_build(kind, site, facing):
        entity_id = controller.build(kind, site, facing)
        cache.remember_building(site, entity_id, kind, bot.team, direction=facing)
        bot.siege_site, bot.siege_ore = site, ore
        bot.siege_feed = feed
        bot.siege_plan = None
    return True


def try_siege(bot, controller, current):
    cache = bot.tile_cache
    target = bot.destination
    if target is None or not bot.destination_is_confirmed_core:
        return False
    site = getattr(bot, 'siege_site', None)
    if site is not None:
        building = cache.building_at(site)
        if building is None or building[1] != bot.team or building[0] not in (EntityType.GUNNER, EntityType.SENTINEL):
            bot.siege_site = None
        else:
            if building[0] == EntityType.GUNNER and clear_firing_lane(bot, controller, current, site):
                return True
            if controller.can_heal(site):
                controller.heal(site)
                return True
            ore = bot.siege_ore
            feed = bot.siege_feed
            if feed != site and cache.building_at(feed) != (EntityType.BRIDGE, bot.team):
                if current == feed or current.distance_squared(feed) > 2:
                    bot.advance_towards_revisitable_target(controller, bot.construction_approach(feed))
                    return True
                old = cache.building_at(feed)
                if old is not None:
                    if old != (EntityType.ROAD, bot.team):
                        bot.siege_site = None
                        return False
                    if controller.can_destroy(feed):
                        controller.destroy(feed)
                        cache.forget_building(feed)
                if controller.can_build_bridge(feed, site):
                    entity_id = controller.build_bridge(feed, site)
                    cache.remember_building(feed, entity_id, EntityType.BRIDGE, bot.team)
                return True
            if cache.building_at(ore) is None:
                if current.distance_squared(ore) > 2:
                    bot.advance_towards_revisitable_target(controller, bot.construction_approach(ore))
                elif controller.can_build_harvester(ore):
                    entity_id = controller.build_harvester(ore)
                    cache.remember_building(ore, entity_id, EntityType.HARVESTER, bot.team)
                    bot.owned_supply_ores.add(ore)
                return True
            # A single gunner can be outhealed by core defenders. Keep the
            # first battery firing while constructing a second independent one.
            bot.siege_site = None

    plan = getattr(bot, 'siege_plan', None)
    if plan is not None:
        return execute_plan(bot, controller, current, plan)

    core_tiles = [cache.offset(target, dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    core_tiles = [p for p in core_tiles if p is not None]
    core_tiles.sort(key=lambda pos: current.distance_squared(pos))
    # One short bridge makes a gunner as cheap as a sentinel, but a single Ti
    # mine supplies its 2-Ti shots every turn instead of one 10-Ti shot per four.
    sources = []
    for ore in cache.visible_tiles:
        if cache.environment_at(ore) != Environment.ORE_TITANIUM:
            continue
        mine = cache.building_at(ore)
        if mine is not None and (mine[0] != EntityType.HARVESTER or (mine[1] == bot.team and ore not in bot.owned_supply_ores)):
            continue
        for d in ORTHOGONAL_DIRECTIONS:
            feed = cache.neighbor(ore, d)
            if buildable(bot, feed):
                sources.append((ore, feed))
    shots = {}
    for tile in core_tiles:
        for facing in DIRECTIONS:
            dx, dy = facing.delta()
            for distance in range(1, 4 if facing in ORTHOGONAL_DIRECTIONS else 3):
                site = cache.offset(tile, -dx * distance, -dy * distance)
                if not buildable(bot, site):
                    continue
                for ore, feed in sources:
                    if site.distance_squared(feed) <= 9 and site.add(facing) != feed:
                        key = (site, facing, feed)
                        shots[key] = (current.distance_squared(site) + current.distance_squared(feed),
                                      site, ore, facing, feed, tile)
    ordered = sorted(shots.values(), key=lambda item: (item[0], item[1].x, item[1].y))
    for _, site, ore, facing, feed, tile in ordered[:12]:
        if controller.can_fire_from(site, facing, EntityType.GUNNER, tile):
            bot.siege_plan = site, ore, EntityType.GUNNER, facing, feed
            return execute_plan(bot, controller, current, bot.siege_plan)
    candidates = []
    for site in tuple(cache.visible_tiles):
        if site is None or cache.environment_at(site) != Environment.EMPTY:
            continue
        building = cache.building_at(site)
        if building is not None and not (building[0] == EntityType.ROAD and building[1] == bot.team):
            continue
        for facing in ORTHOGONAL_DIRECTIONS:
            ore = cache.neighbor(site, facing)
            if ore is None or cache.environment_at(ore) != Environment.ORE_TITANIUM:
                continue
            mine = cache.building_at(ore)
            # The economy's mines retain all their outputs.
            if mine is not None and (mine[0] != EntityType.HARVESTER or (mine[1] == bot.team and ore not in bot.owned_supply_ores)):
                continue
            if min(site.distance_squared(p) for p in core_tiles) > 32:
                continue
            candidates.append((mine is None, site, ore))
    candidates.sort(key=lambda item: (item[0], current.distance_squared(item[1]), item[1].x, item[1].y))
    for unmined, site, ore in candidates[:6]:
        for kind in (EntityType.GUNNER, EntityType.SENTINEL):
            for target_tile in core_tiles[:3]:
                facing = site.direction_to(target_tile)
                # A turret cannot receive ammunition through its output side.
                if site.add(facing) == ore:
                    continue
                if not controller.can_fire_from(site, facing, kind, target_tile):
                    continue
                bot.siege_plan = site, ore, kind, facing, site
                return execute_plan(bot, controller, current, bot.siege_plan)
    # Explore the ammunition source before choosing a disconnected gunner.
    if bot.gunner_id is None:
        ores = [p for p, env in cache.environments.items()
                if env == Environment.ORE_TITANIUM and p.distance_squared(target) <= 64
                and p not in cache.visible_tiles
                and (cache.building_at(p) is None or cache.building_at(p)[1] != bot.team)]
        if ores:
            ore = min(ores, key=lambda p: current.distance_squared(p))
            approach = bot.construction_approach(ore)
            if approach is not None:
                bot.advance_towards_revisitable_target(controller, approach)
                return True
        if controller.get_current_round() < 100:
            probes = [cache.offset(target, dx, dy) for dx, dy in
                      ((-5, -4), (0, -6), (5, -4), (6, 0), (5, 4), (0, 6), (-5, 4), (-6, 0))]
            probes = [p for p in probes if p is not None and p not in cache.observed_tiles
                      and cache.environment_at(p) != Environment.WALL]
            if probes:
                probe = min(probes, key=lambda p: current.distance_squared(p))
                bot.advance_towards_unvisited_target(controller, probe)
                return True
    return False


def buildable(bot, site):
    if site is None or bot.tile_cache.environment_at(site) != Environment.EMPTY:
        return False
    building = bot.tile_cache.building_at(site)
    if building is not None and building != (EntityType.ROAD, bot.team):
        return False
    for entity_id in bot.tile_cache.visible_entity_ids:
        if bot.tile_cache.entity_type(entity_id) != EntityType.GUNNER or bot.tile_cache.entity_team(entity_id) != bot.team:
            continue
        origin = bot.tile_cache.entity_position(entity_id)
        facing = bot.tile_cache.entity_direction(entity_id)
        if facing is not None and origin.direction_to(site) == facing:
            dx, dy = abs(site.x - origin.x), abs(site.y - origin.y)
            if (dx == 0 or dy == 0 or dx == dy) and site.distance_squared(origin) <= 13:
                return False
    return True


def clear_firing_lane(bot, controller, current, site):
    cache = bot.tile_cache
    direction = cache.conveyor_directions.get(site)
    entity_id = cache.building_id_at(site)
    if entity_id is not None:
        direction = cache.entity_direction(entity_id)
    if direction is None:
        return False
    lane = []
    pos = site
    for _ in range(3 if direction in ORTHOGONAL_DIRECTIONS else 2):
        pos = cache.neighbor(pos, direction)
        if pos is None:
            break
        lane.append(pos)
    for pos in lane:
        if cache.building_at(pos) == (EntityType.ROAD, bot.team):
            if controller.can_destroy(pos):
                controller.destroy(pos)
                cache.forget_building(pos)
                continue
            approaches = [cache.neighbor(pos, d) for d in DIRECTIONS]
            approaches = [p for p in approaches if p is not None and p not in lane
                          and bot.is_roadable_position(p)]
            if approaches:
                bot.advance_towards_revisitable_target(controller, min(approaches, key=lambda p: current.distance_squared(p)))
                return True
    if current in lane:
        for direction in DIRECTIONS:
            step = cache.neighbor(current, direction)
            if step is not None and step not in lane and bot.try_move_step(controller, direction):
                return True
    return False
