"""Regression checks for replay failures: interrupted searches and broken trunks."""
import sys
import unittest

from cambc import Direction, EntityType, Environment, Team

from tests.nexus.test_cache_and_planning import (
    NEXUS, TileCache, construction_access, exploration, load_module,
)


budget_module = load_module('nexus_budget_test', 'planning_budget.py')
module_names = tuple(path.stem for path in NEXUS.glob('*.py'))
shadowed = {name: sys.modules.get(name) for name in module_names}
for name in module_names:
    sys.modules.pop(name, None)
sys.path.insert(0, str(NEXUS))
try:
    builder_module = load_module('nexus_builder_slices_test', 'builder_bot.py')
    base_module = sys.modules['base']
    player_module = load_module('nexus_player_slices_test', 'main.py')
finally:
    sys.path.remove(str(NEXUS))
    for name in module_names:
        sys.modules.pop(name, None)
    sys.modules.update((name, value) for name, value in shadowed.items() if value is not None)


class Clock:
    elapsed = 0

    def get_cpu_time_elapsed(self):
        return self.elapsed


class PlanningSliceTests(unittest.TestCase):
    def test_failed_generator_is_retried_instead_of_caching_none(self):
        for error in (ValueError('broken route'), BaseException('interrupted')):
            with self.subTest(error=type(error).__name__):
                budget = budget_module.PlanningBudget()
                memory = budget_module.SearchMemory(budget)
                starts = []

                def search():
                    starts.append(1)
                    yield
                    if len(starts) == 1:
                        raise error
                    return ['route']

                budget.begin_turn(Clock())
                with self.assertRaises(type(error)) as raised:
                    memory.run('route', search)
                self.assertIs(raised.exception, error)
                budget.begin_turn(Clock())
                self.assertEqual(memory.run('route', search), ['route'])
                self.assertEqual(starts, [1, 1])

    def test_changing_version_does_not_restart_pending_search(self):
        budget = budget_module.PlanningBudget()
        memory = budget_module.SearchMemory(budget)
        starts = []

        def search():
            starts.append(1)
            for _ in range(25):
                yield
            return 'done'

        for version in range(5):
            budget.begin_turn(Clock())
            try:
                result = memory.run('route', search, version=version)
                break
            except RuntimeError:
                self.assertTrue(budget.pending)
        self.assertEqual(result, 'done')
        self.assertEqual(starts, [1])
        budget.begin_turn(Clock())
        with self.assertRaises(RuntimeError):
            memory.run('route', search, version=99)
        self.assertEqual(starts, [1, 1])

    def test_builder_does_not_swallow_unrelated_error_during_planning(self):
        bot = builder_module.BuilderBot(5, 5)

        def broken_turn(controller):
            bot.planning_budget.pending = True
            raise ValueError('broken route')

        bot.run_turn = broken_turn
        with self.assertRaisesRegex(ValueError, 'broken route'):
            bot.run(Clock())

    def test_interrupted_role_initialization_is_retried(self):
        from unittest.mock import patch
        player = player_module.Player()

        class Controller:
            def get_entity_type(self): return EntityType.BUILDER_BOT
            def get_map_width(self): return 50
            def get_map_height(self): return 50
            def get_position(self): return TileCache(1, 1).position_at(0, 0)
            def get_team(self): return Team.A
            def get_nearby_buildings(self): return []

        role = object()
        with patch.object(player_module, 'BuilderBot', side_effect=[RuntimeError('interrupted'), role]):
            with self.assertRaises(RuntimeError):
                player.init_once(Controller())
            self.assertFalse(player.initialized)
            player.init_once(Controller())
        self.assertTrue(player.initialized)
        self.assertIs(player.bot, role)

    def test_partial_scans_deliver_all_discoveries_to_role(self):
        bot = base_module.BaseBot(5, 5)
        cache = bot.tile_cache
        first, second = cache.position_at(0, 0), cache.position_at(1, 0)
        bot.entity_id, bot.team = 1, Team.A
        batches = iter(({first}, {second}, set()))

        def scan(controller, own_id):
            cache.newly_observed_tiles = next(batches)
            cache.scan_incomplete_this_turn = first in cache.newly_observed_tiles
            cache.current_position = cache.position_at(2, 2)

        cache.scan_turn = scan
        self.assertTrue(bot._scan_turn(None))
        self.assertFalse(bot._scan_turn(None))
        self.assertEqual(cache.newly_observed_tiles, {first, second})
        self.assertFalse(bot._scan_turn(None))
        self.assertEqual(cache.newly_observed_tiles, set())

    def test_search_resumes_without_restarting_and_caches_result(self):
        budget = budget_module.PlanningBudget()
        memory = budget_module.SearchMemory(budget)
        events = []

        def search():
            events.append('start')
            for index in range(40):
                events.append(index)
                yield
            return 123

        for turn in range(5):
            budget.begin_turn(Clock())
            try:
                result = memory.run('route', search)
                break
            except RuntimeError:
                self.assertTrue(budget.pending)
        self.assertEqual(turn, 3)
        self.assertEqual(result, 123)
        self.assertEqual(events, ['start'] + list(range(40)))
        self.assertEqual(memory.run('route', search), 123)

    def test_time_guard_yields_before_advancing_search(self):
        clock = Clock()
        clock.elapsed = 1200
        budget = budget_module.PlanningBudget()
        budget.begin_turn(clock)
        memory = budget_module.SearchMemory(budget)
        events = []

        def search():
            events.append('advanced')
            yield
            return 7

        with self.assertRaises(RuntimeError):
            memory.run('route', search)
        self.assertEqual(events, [])
        clock.elapsed = 0
        budget.begin_turn(clock)
        self.assertEqual(memory.run('route', search), 7)
        self.assertFalse(budget.pending)

    def test_frontier_beyond_single_turn_limit_is_not_discarded(self):
        cache = TileCache(500, 1)
        start, target = cache.position_at(0, 0), cache.position_at(499, 0)
        budget = budget_module.PlanningBudget()
        memory = budget_module.SearchMemory(budget)
        for _ in range(5):
            budget.begin_turn(Clock())
            try:
                path = memory.run('frontier', lambda: exploration.frontier_search_steps(
                    start, {target}, cache.neighbor, lambda pos: True,
                    [Direction.EAST, Direction.WEST],
                ))
                break
            except RuntimeError:
                self.assertTrue(budget.pending)
        self.assertEqual(len(path), 499)
        self.assertEqual(path[-1], target)
        self.assertEqual(len(set(path)), len(path))

    def test_new_terrain_during_access_search_invalidates_completed_cache(self):
        cache = TileCache(250, 2)
        start = cache.position_at(0, 0)
        revealed = cache.position_at(0, 1)
        terrain = {cache.position_at(x, 0): Environment.EMPTY for x in range(250)}
        access = construction_access.ConstructionAccess()
        budget = budget_module.PlanningBudget()
        directions = [Direction.EAST, Direction.WEST, Direction.SOUTH, Direction.NORTH]
        budget.begin_turn(Clock())
        with self.assertRaises(RuntimeError):
            access.reachable(start, terrain, set(), cache.neighbor, directions, budget)
        terrain[revealed] = Environment.EMPTY
        budget.begin_turn(Clock())
        access.reachable(start, terrain, set(), cache.neighbor, directions, budget)
        # The second call finishes the old queue, but must not label that old
        # component as already covering the newly observed cell behind it.
        self.assertNotEqual(access.signature[0], len(terrain))
        result = access.reachable(start, terrain, set(), cache.neighbor, directions)
        self.assertIn(revealed, result)


class ConstructionOrderTests(unittest.TestCase):
    def setUp(self):
        self.bot = builder_module.BuilderBot(10, 7)
        self.bot.team = Team.A
        self.bot.core_pos = self.bot.tile_cache.position_at(7, 3)
        self.p = self.bot.tile_cache.position_at
        self.bot.known_env.update({self.p(x, y): Environment.EMPTY
                                   for x in range(10) for y in range(7)})

    def test_connectivity_does_not_mutate_cached_core_footprint(self):
        bot = self.bot
        footprint = set(bot.core_receiver_tiles())
        source = self.p(5, 3)
        bot.known_buildings[source] = (EntityType.CONVEYOR, Team.A)
        bot.known_conveyor_directions[source] = Direction.EAST
        self.assertIn(source, bot.known_connected_network())
        self.assertEqual(bot.core_receiver_tiles(), footprint)
        self.assertFalse(bot.is_core_receiver_tile(source))
        bot.known_buildings[source] = None
        bot.connected_network_cache = None
        self.assertNotIn(source, bot.known_connected_network())

    def test_bridge_cannot_feed_the_output_side_of_a_conveyor(self):
        bot = self.bot
        for y in (1, 2, 3):
            tile = self.p(5, y)
            bot.known_buildings[tile] = (EntityType.CONVEYOR, Team.A)
            bot.known_conveyor_directions[tile] = Direction.SOUTH if y < 3 else Direction.EAST
        bridge, landing = self.p(5, 4), self.p(5, 1)
        bot.known_buildings[bridge] = (EntityType.BRIDGE, Team.A)
        bot.known_bridge_targets[bridge] = landing
        self.assertIn(landing, bot.known_connected_network())
        self.assertNotIn(bridge, bot.known_connected_network())

    def test_ore_route_resumes_past_old_64_node_limit(self):
        bot = builder_module.BuilderBot(50, 3)
        p = bot.tile_cache.position_at
        bot.known_env.update({p(x, y): Environment.EMPTY if y != 1 or x == 49 else Environment.WALL
                              for x in range(50) for y in range(3)})
        for turn in range(10):
            bot.planning_budget.begin_turn(Clock())
            try:
                path = bot.ore_walk(p(0, 0), {p(0, 2)})
                break
            except RuntimeError:
                self.assertTrue(bot.planning_budget.pending)
        self.assertGreater(turn, 0)
        self.assertEqual(path[-1], p(0, 2))
        self.assertGreaterEqual(len(path), 98)
        self.assertTrue(all(bot.known_env[tile] == Environment.EMPTY for tile in path))

    def test_branch_execution_starts_at_sink_and_returns_to_mine(self):
        bot = self.bot
        ore = self.p(1, 3)
        bot.known_env[ore] = Environment.ORE_TITANIUM
        tiles = [self.p(x, 3) for x in range(2, 6)]
        directions = {tile: Direction.EAST for tile in tiles}
        bot.connection_plan = lambda controller, target: (
            tiles[0], tiles, directions, {}, self.p(6, 3),
        )
        self.assertTrue(bot.assign_connection_target(None, tiles[0], ore))
        self.assertEqual(bot.path[-len(tiles):], list(reversed(tiles)))
        self.assertTrue(bot.bridge_build_is_deferred(tiles[0]))
        self.assertFalse(bot.bridge_build_is_deferred(tiles[-1]))
        for tile in reversed(tiles[1:]):
            bot.known_buildings[tile] = (EntityType.CONVEYOR, Team.A)
            bot.known_conveyor_directions[tile] = Direction.EAST
        bot.connected_network_cache = None
        self.assertFalse(bot.bridge_build_is_deferred(tiles[0]))

    def test_bridge_waits_for_landing_to_reach_core(self):
        bot = self.bot
        source, landing = self.p(2, 3), self.p(5, 3)
        bot.bridge_targets[source] = landing
        bot.conveyor_path_tiles.add(source)
        self.assertTrue(bot.bridge_build_is_deferred(source))
        bot.known_buildings[landing] = (EntityType.CONVEYOR, Team.A)
        bot.known_conveyor_directions[landing] = Direction.EAST
        bot.connected_network_cache = None
        self.assertFalse(bot.bridge_build_is_deferred(source))

    def test_empty_frontier_shortlist_still_tries_reachable_boundary(self):
        bot = self.bot
        bot.current_position = self.p(0, 0)
        target = self.p(9, 6)
        bot.observed_tiles.update(set(bot.known_env) - {target})
        bot.scout_frontier = {target}
        bot.scout_frontier_initialized = True
        bot.choose_right_hand_scout_step = lambda *args, **kwargs: None
        bot.scout_frontier_has_known_entry = lambda pos: False
        selected, path = bot.choose_scout_plan(None)
        self.assertEqual(selected, target)
        self.assertEqual(path[-1], target)

    def test_construction_walk_resumes_long_detour(self):
        bot = builder_module.BuilderBot(250, 1)
        p = bot.tile_cache.position_at
        bot.known_env.update({p(x, 0): Environment.EMPTY for x in range(250)})
        for _ in range(3):
            bot.planning_budget.begin_turn(Clock())
            try:
                path = bot.connection_walk(p(0, 0), {p(249, 0)})
                break
            except RuntimeError:
                self.assertTrue(bot.planning_budget.pending)
        self.assertEqual(len(path), 249)


if __name__ == '__main__':
    unittest.main()
