"""Ammo-first batteries using RC movement and the shared map cache.

Adapt sota/qualifier's use of enemy mine outputs without taking economy lanes.
"""
from cambc import Direction, EntityType, Environment
from constants import DIRECTIONS, ORTHOGONAL_DIRECTIONS

TURRETS = (EntityType.GUNNER, EntityType.SENTINEL)


def buildable(bot, site):
    if site is None or bot.tile_cache.environment_at(site) != Environment.EMPTY:
        return False
    building = bot.tile_cache.building_at(site)
    return building is None or building == (EntityType.ROAD, bot.team) or (
        building[1] != bot.team and building[0] in
        (EntityType.ROAD, EntityType.CONVEYOR, EntityType.BRIDGE, EntityType.SPLITTER))


def approach(bot, controller, current, site):
    if current == site:
        bot.vacate_gunner_site(controller, site)
        return False
    if current.distance_squared(site) > 2:
        target = bot.construction_approach(site)
        if target is not None:
            bot.advance_towards_revisitable_target(controller, target)
        return False
    return True


def execute_plan(bot, controller, current, plan):
    site, ore, kind, facing = plan
    cache = bot.tile_cache
    mine = cache.building_at(ore)
    old = cache.building_at(site)
    if old is not None and old[1] != bot.team and buildable(bot, site):
        if current != site:
            bot.advance_towards_revisitable_target(controller, site)
        elif controller.can_fire(site):
            entity_id = cache.building_id_at(site)
            hp = controller.get_hp(entity_id)
            controller.fire(site)
            if hp <= 2:
                cache.forget_building(site)
        return True
    if old is not None and old != (EntityType.ROAD, bot.team):
        if old == (kind, bot.team):
            bot.siege_site, bot.siege_ore = site, ore
        bot.siege_plan = None
        return False
    if mine is not None and mine[0] != EntityType.HARVESTER:
        bot.siege_plan = None
        return False
    if mine == (EntityType.HARVESTER, bot.team) and ore not in bot.owned_supply_ores:
        bot.siege_plan = None
        return False
    if mine is None:
        if not approach(bot, controller, current, ore):
            return True
        if controller.can_build_harvester(ore):
            entity_id = controller.build_harvester(ore)
            cache.remember_building(ore, entity_id, EntityType.HARVESTER, bot.team)
            bot.owned_supply_ores.add(ore)
        return True
    if not approach(bot, controller, current, site):
        return True
    if old is not None:
        if not controller.can_destroy(site):
            return True
        controller.destroy(site)
        cache.forget_building(site)
    if controller.can_build(kind, site, facing):
        entity_id = controller.build(kind, site, facing)
        cache.remember_building(site, entity_id, kind, bot.team, direction=facing)
        bot.siege_site, bot.siege_ore = site, ore
        bot.siege_plan = None
    return True


def try_siege(bot, controller, current):
    cache = bot.tile_cache
    target = bot.destination
    if target is None or not bot.destination_is_confirmed_core:
        return False
    now = controller.get_current_round()
    for entity_id in cache.visible_entity_ids:
        if cache.entity_type(entity_id) in TURRETS and cache.entity_team(entity_id) == bot.team:
            pos = cache.entity_position(entity_id)
            if controller.can_heal(pos):
                controller.heal(pos)
                return True
    if not hasattr(bot, 'siege_rejected'):
        bot.siege_rejected = {}
        bot.siege_plan = None
        bot.siege_site = None
        bot.siege_progress = (current, now)
    site = bot.siege_site
    if site is not None:
        building = cache.building_at(site)
        if building is not None and building[1] == bot.team and building[0] in TURRETS:
            if building[0] == EntityType.GUNNER and clear_firing_lane(bot, controller, current, site):
                return True
            if controller.can_heal(site):
                controller.heal(site)
                return True
            ore = bot.siege_ore
            if cache.building_at(ore) is None:
                if approach(bot, controller, current, ore) and controller.can_build_harvester(ore):
                    entity_id = controller.build_harvester(ore)
                    cache.remember_building(ore, entity_id, EntityType.HARVESTER, bot.team)
                    bot.owned_supply_ores.add(ore)
                return True
        bot.siege_site = None

    plan = bot.siege_plan
    if plan is not None:
        previous, since = bot.siege_progress
        if current != previous:
            bot.siege_progress = (current, now)
        elif now - since > 12:
            # Lost collisions or a changed obstacle must not freeze the attack.
            bot.siege_rejected[plan[0]] = now + 32
            bot.siege_plan = None
        if bot.siege_plan is not None:
            return execute_plan(bot, controller, current, plan)

    core_tiles = [cache.offset(target, dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    core_tiles = [p for p in core_tiles if p is not None]
    core_tiles.sort(key=lambda p: current.distance_squared(p))
    candidates = []
    for ore in sorted(cache.visible_tiles, key=lambda p: (p.x, p.y)):
        if cache.environment_at(ore) != Environment.ORE_TITANIUM:
            continue
        mine = cache.building_at(ore)
        if mine is not None and (mine[0] != EntityType.HARVESTER or
                                (mine[1] == bot.team and ore not in bot.owned_supply_ores)):
            continue
        if ore.distance_squared(target) > 85:
            continue
        for direction in ORTHOGONAL_DIRECTIONS:
            site = cache.neighbor(ore, direction)
            if not buildable(bot, site) or bot.siege_rejected.get(site, 0) > now:
                continue
            obstruction = cache.building_at(site)
            penalty = 18 if obstruction is not None and obstruction[1] != bot.team else 0
            # Capture a hostile source even when it cannot yet reach the core.
            # The gunner loads facing away, then chooses its own nearby target.
            if mine is not None and mine[1] != bot.team:
                candidates.append((35 + current.distance_squared(site) + penalty,
                                   site, ore, EntityType.GUNNER, direction, None))
            for kind, radius in ((EntityType.GUNNER, 13), (EntityType.SENTINEL, 32)):
                for goal in core_tiles:
                    if site.distance_squared(goal) > radius:
                        continue
                    facing = site.direction_to(goal)
                    if facing == Direction.CENTRE or site.add(facing) == ore:
                        continue
                    dx, dy = abs(goal.x-site.x), abs(goal.y-site.y)
                    if kind == EntityType.GUNNER and dx and dy and dx != dy:
                        continue
                    score = current.distance_squared(site) + penalty + (12 if mine is None else 0) + (8 if kind == EntityType.SENTINEL else 0)
                    candidates.append((score, site, ore, kind, facing, goal))
    candidates.sort(key=lambda v: v[0])
    for _, site, ore, kind, facing, goal in candidates[:20]:
        if goal is None or controller.can_fire_from(site, facing, kind, goal):
            bot.siege_plan = site, ore, kind, facing
            bot.siege_progress = (current, now)
            return execute_plan(bot, controller, current, bot.siege_plan)
    # Survey the source before falling back to RC's longer bridge supply plan.
    hidden = [p for p, env in cache.environments.items()
              if env == Environment.ORE_TITANIUM and p.distance_squared(target) <= 85
              and p not in cache.observed_tiles and bot.is_available_supply_ore(p)]
    if hidden:
        ore = min(hidden, key=lambda p: current.distance_squared(p))
        pos = bot.construction_approach(ore)
        if pos is not None:
            bot.advance_towards_revisitable_target(controller, pos)
            return True
    return False


def clear_firing_lane(bot, controller, current, site):
    cache = bot.tile_cache
    entity_id = cache.building_id_at(site)
    direction = cache.entity_direction(entity_id) if entity_id is not None else None
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
        if cache.building_at(pos) == (EntityType.ROAD, bot.team) and controller.can_destroy(pos):
            controller.destroy(pos)
            cache.forget_building(pos)
    if current in lane:
        for direction in DIRECTIONS:
            step = cache.neighbor(current, direction)
            if step is not None and step not in lane and bot.try_move_step(controller, direction):
                return True
    return False
