"""Build a complete, short Ti + raw Ax -> refined Ax transaction.

Raw ore stays disabled in the general conveyor planner. These dedicated
inputs terminate at a foundry; only its refined output joins the core tree.
"""
from cambc import EntityType, Environment
from constants import DIRECTIONS, ORTHOGONAL_DIRECTIONS


def available(bot, pos, source=None, factory=False):
    cache = bot.tile_cache
    if pos is None or cache.environment_at(pos) != Environment.EMPTY:
        return False
    building = cache.building_at(pos)
    if building is None or building == (EntityType.ROAD, bot.team):
        return True
    if factory and building == (EntityType.FOUNDRY, bot.team):
        return True
    # A spare port already sealed by a barrier can be reused. Productive
    # conveyors, including another worker's lanes, remain untouched.
    if source is None:
        return False
    if building == (EntityType.BARRIER, bot.team):
        return True
    return (cache.environment_at(source) == Environment.ORE_TITANIUM
            and building[1] == bot.team and building[0] in (EntityType.CONVEYOR, EntityType.BRIDGE)
            and getattr(bot, 'refinery_network_loads', {}).get(pos, 99) <= 1)


def inputs(bot, ore, foundry):
    cache = bot.tile_cache
    if ore.distance_squared(foundry) == 1:
        return [(None, None)]
    result = []
    for direction in ORTHOGONAL_DIRECTIONS:
        feed = cache.neighbor(ore, direction)
        old = cache.building_at(feed) if feed is not None else None
        reuse = (old == (EntityType.BRIDGE, bot.team) and bot.known_bridge_targets.get(feed) == foundry) or (
            old == (EntityType.CONVEYOR, bot.team) and cache.neighbor(feed, cache.entity_direction(cache.building_id_at(feed))) == foundry)
        if (available(bot, feed, ore) or reuse) and feed != foundry and feed.distance_squared(foundry) <= 9:
            # An adjacent conveyor rejects output back from the foundry.
            # A bridge beside it would accept that output into its input lane.
            kind = EntityType.CONVEYOR if feed.distance_squared(foundry) == 1 else EntityType.BRIDGE
            result.append((feed, kind))
    return result


def axionite_inputs(bot, ore, foundry):
    short = inputs(bot, ore, foundry)
    if short:
        return [(feed, kind, None, None) for feed, kind in short]
    cache = bot.tile_cache
    routes = []
    for direction in ORTHOGONAL_DIRECTIONS:
        feed = cache.neighbor(ore, direction)
        if not available(bot, feed, ore):
            continue
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if not 0 < dx*dx+dy*dy <= 9:
                    continue
                relay = cache.offset(foundry, dx, dy)
                if relay is None or relay == feed or relay.distance_squared(feed) > 9 or not available(bot, relay):
                    continue
                kind = EntityType.CONVEYOR if dx*dx+dy*dy == 1 else EntityType.BRIDGE
                routes.append((feed, EntityType.BRIDGE, relay, kind))
    return routes[:12]


def plans(bot, current):
    cache = bot.tile_cache
    ax_ores = sorted((p for p in cache.observed_tiles
                     if cache.environment_at(p) == Environment.ORE_AXIONITE
                     and cache.building_at(p) is None),
                    key=lambda p: current.distance_squared(p))
    titanium = [p for p, building in cache.buildings.items()
                if building == (EntityType.HARVESTER, bot.team)
                and cache.environment_at(p) == Environment.ORE_TITANIUM]
    network = bot.known_connected_network()
    bot.refinery_network_loads = bot.known_network_loads(network)
    existing = {p for p, building in cache.buildings.items() if building == (EntityType.FOUNDRY, bot.team)}
    for ax in ax_ores:
        for ti in sorted(titanium, key=lambda p: p.distance_squared(ax)):
            if ti.distance_squared(ax) > 121:
                continue
            sites = [cache.offset(ti, dx, dy) for dx in range(-4, 5) for dy in range(-4, 5)
                     if dx*dx+dy*dy <= 16]
            sites = sorted((p for p in sites if available(bot, p, factory=True)
                            and (not existing or p in existing) and p.distance_squared(ax) <= 49),
                           key=lambda p: (p.distance_squared(ax) > 16, current.distance_squared(p)))
            for site in sites:
                yield
                ax_inputs, ti_inputs = axionite_inputs(bot, ax, site), inputs(bot, ti, site)
                if not ax_inputs or not ti_inputs:
                    continue
                # Each planned impassable building needs a known approach.
                if not any(bot.traversable_for_planning(None, p)
                           for d in DIRECTIONS if (p := cache.neighbor(site, d)) is not None):
                    continue
                for output_dir in ORTHOGONAL_DIRECTIONS:
                    out = cache.neighbor(site, output_dir)
                    reuse_output = (cache.building_at(out) == (EntityType.BRIDGE, bot.team)
                                    and bot.known_bridge_targets.get(out) in network) if out is not None else False
                    if not (available(bot, out) or reuse_output) or out.distance_squared(ax) == 1:
                        continue
                    receivers = [p for p in network if p.distance_squared(out) <= 9 and p != out]
                    receivers.sort(key=lambda p: bot.core_distance(p))
                    if not receivers:
                        continue
                    for ax_feed, ax_kind, ax_relay, relay_kind in ax_inputs:
                        for ti_feed, ti_kind in ti_inputs:
                            used = [p for p in (site, out, ax_feed, ti_feed, ax_relay) if p is not None]
                            if len(used) != len(set(used)):
                                continue
                            # Do not let raw Ax enter the titanium input.
                            if ti_feed is not None and ti_feed.distance_squared(ax) == 1:
                                continue
                            receiver = next((p for p in receivers if p not in used), None)
                            if receiver is None:
                                continue
                            steps = [(out, EntityType.BRIDGE, receiver, 'output'),
                                     (site, EntityType.FOUNDRY, None, 'foundry')]
                            if ti_feed is not None:
                                steps.append((ti_feed, ti_kind, site if ti_kind == EntityType.BRIDGE else ti_feed.direction_to(site), 'titanium'))
                            if ax_feed is not None:
                                if ax_relay is not None:
                                    steps.append((ax_relay, relay_kind, site if relay_kind == EntityType.BRIDGE else ax_relay.direction_to(site), 'axionite'))
                                sink = ax_relay or site
                                steps.append((ax_feed, ax_kind, sink if ax_kind == EntityType.BRIDGE else ax_feed.direction_to(sink), 'axionite'))
                            steps.append((ax, EntityType.HARVESTER, None, 'mine'))
                            return steps
    return None


def try_refinery(bot, controller, current):
    now = controller.get_current_round()
    project = getattr(bot, 'refinery_project', None)
    if project is None:
        if now < getattr(bot, 'refinery_retry', 100) or controller.get_global_resources()[0] < 500:
            return False
        search = getattr(bot, 'refinery_search', None)
        if search is None:
            search = bot.refinery_search = plans(bot, current)
        # Search resumes while normal mining and exploration continue.
        for _ in range(6):
            if controller.get_cpu_time_elapsed() >= 1100:
                break
            try:
                next(search)
            except StopIteration as result:
                bot.refinery_search = None
                bot.refinery_retry = now + 32
                if result.value is None:
                    return False
                bot.refinery_project = project = result.value
                bot.refinery_index = 0
                bot.refinery_progress = now
                break
        if project is None:
            return False
    if now - bot.refinery_progress > 80:
        bot.refinery_project = None
        bot.refinery_retry = now + 64
        return False
    cache = bot.tile_cache
    while bot.refinery_index < len(project):
        pos, kind, extra, role = project[bot.refinery_index]
        building = cache.building_at(pos)
        matches = building == (kind, bot.team)
        if matches and kind == EntityType.BRIDGE:
            matches = bot.known_bridge_targets.get(pos) == extra
        if matches and kind == EntityType.CONVEYOR:
            matches = cache.entity_direction(cache.building_id_at(pos)) == extra
        if matches:
            bot.refinery_index += 1
            continue
        replace_input = (role == 'titanium' and building is not None and building[1] == bot.team
                         and building[0] in (EntityType.BRIDGE, EntityType.CONVEYOR)
                         and bot.known_network_loads(bot.known_connected_network()).get(pos, 99) <= 1)
        if building is not None and not replace_input and building not in ((EntityType.ROAD, bot.team), (EntityType.BARRIER, bot.team)):
            bot.refinery_project = None
            return False
        if current.distance_squared(pos) > 2 or (current == pos and kind != EntityType.CONVEYOR):
            approaches = {p for d in DIRECTIONS if (p := cache.neighbor(pos, d)) is not None
                          and bot.traversable_for_planning(None, p)}
            path = bot.ore_walk(current, approaches) if approaches else []
            if path:
                bot.try_move_scout_step(controller, current, current.direction_to(path[0]))
            return True
        if building is not None:
            if not controller.can_destroy(pos):
                return True
            controller.destroy(pos)
            cache.forget_building(pos)
            bot.network_memory.forget(pos)
        if not controller.can_build(kind, pos, extra):
            return True
        entity_id = controller.build(kind, pos, extra)
        cache.remember_building(pos, entity_id, kind, bot.team,
                                direction=extra if kind == EntityType.CONVEYOR else None)
        if kind == EntityType.BRIDGE:
            bot.known_bridge_targets[pos] = extra
            bot.known_bridge_ids[pos] = entity_id
        bot.refinery_progress = now
        bot.last_progress_round = bot.rounds_alive
        bot.connected_network_cache = None
        bot.refinery_index += 1
        return True
    bot.refinery_project = None
    bot.refinery_retry = now + 256
    return False
