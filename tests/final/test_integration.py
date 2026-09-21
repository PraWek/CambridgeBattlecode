"""Contracts between Nexus's economy and RC's combat roles."""

import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

from cambc import Direction, EntityType, Environment, GameConstants, Position, Team


ROOT = Path(__file__).resolve().parents[2]
BOT = ROOT / "bots" / "final"
names = [p.stem for p in BOT.glob("*.py")]
previous = {name: sys.modules.pop(name, None) for name in names}
sys.path.insert(0, str(BOT))
try:
    player = importlib.import_module("main")
    core = importlib.import_module("core_bot")
    orders = importlib.import_module("spawn_orders")
    fleet = importlib.import_module("fleet")
    combat = importlib.import_module("combat_navigation")
    geometry = importlib.import_module("geometry")
    constants = importlib.import_module("constants")
finally:
    sys.path.remove(str(BOT))
    for name in names:
        sys.modules.pop(name, None)
    sys.modules.update((name, module) for name, module in previous.items() if module is not None)


class CoreController:
    def __init__(self):
        self.round = 1
        self.units = 1
        self.next_id = 10
        self.spawned = []
        self.markers = {}
        self.dead = set()
        self.destroyed = []
        self.resources = (10000, 0)

    def get_id(self): return 1
    def get_team(self, entity_id=None): return Team.A
    def get_position(self, entity_id=None): return Position(5, 5)
    def get_unit_count(self): return self.units
    def get_current_round(self): return self.round
    def can_spawn(self, pos): return self.units < GameConstants.MAX_TEAM_UNITS
    def can_place_marker(self, pos): return True
    def can_destroy(self, pos): return pos in self.markers
    def get_global_resources(self): return self.resources
    def get_harvester_cost(self): return (100, 0)
    def get_conveyor_cost(self): return (3, 0)
    def get_builder_bot_cost(self): return (50, 0)

    def spawn_builder(self, pos):
        self.next_id += 1
        self.spawned.append((self.next_id, pos))
        self.units += 1
        return self.next_id

    def place_marker(self, pos, value):
        self.next_id += 1
        self.markers[pos] = (self.next_id, value)
        return self.next_id

    def destroy(self, pos):
        self.destroyed.append(pos)
        del self.markers[pos]

    def get_hp(self, entity_id):
        if entity_id in self.dead:
            raise ValueError("Unknown id")
        raise ValueError("Position out of vision range")


def newborn(marker_kind=None, marker_team=Team.A, payload=0):
    c = Mock()
    c.get_entity_type.side_effect = lambda entity_id=None: (
        EntityType.BUILDER_BOT if entity_id is None else EntityType.MARKER
    )
    c.get_position.return_value = Position(6, 6)
    c.get_team.side_effect = lambda entity_id=None: Team.A if entity_id is None else marker_team
    c.get_map_width.return_value = 12
    c.get_map_height.return_value = 12
    c.get_nearby_buildings.return_value = [] if marker_kind is None else [42]
    if marker_kind is not None:
        c.get_marker_value.return_value = geometry.encode_marker(marker_kind, Position(6, 6), payload)
    return c


class SpawnIntegrationTests(unittest.TestCase):
    def test_opening_keeps_intruder_and_four_economic_sectors(self):
        c = CoreController()
        bot = core.CoreBot(12, 12)
        for _ in range(5):
            bot.run(c)
            c.round += 1
        self.assertEqual(len(bot.economy_builder_ids), 4)
        self.assertNotIn(bot.intruder_id, bot.economy_builder_ids)
        self.assertEqual(bot.intruder_id, c.spawned[0][0])
        self.assertEqual(bot.initial_spawned_directions, set(constants.BUILDER_WORK_DIRECTIONS))
        self.assertFalse(bot.intruder_order_active)

    def test_turrets_do_not_consume_economic_worker_quota(self):
        c = CoreController()
        c.units = 25
        bot = core.CoreBot(12, 12)
        bot.core_pos = bot.tile_cache.position_at(5, 5)
        self.assertTrue(bot.try_spawn_missing_builder(c))
        self.assertEqual(len(bot.economy_builder_ids), 1)

    def test_fog_keeps_worker_but_death_enables_replacement(self):
        c = CoreController()
        bot = core.CoreBot(12, 12)
        bot.core_pos = bot.tile_cache.position_at(5, 5)
        bot.initial_spawned_directions.update(constants.BUILDER_WORK_DIRECTIONS)
        bot.economy_builder_ids = [20, 21, 22, 23]
        bot.refresh_fleet(c)
        self.assertEqual(bot.economy_builder_ids, [20, 21, 22, 23])
        c.dead.add(21)
        bot.refresh_fleet(c)
        self.assertEqual(bot.economy_builder_ids, [20, 22, 23])
        self.assertTrue(bot.try_spawn_missing_builder(c))

    def test_unit_cap_applies_to_both_roles(self):
        c = CoreController()
        c.units = GameConstants.MAX_TEAM_UNITS
        bot = core.CoreBot(12, 12)
        bot.core_pos = bot.tile_cache.position_at(5, 5)
        self.assertFalse(bot.try_spawn_missing_builder(c))
        self.assertFalse(bot.try_spawn_intruder(c))

    def test_combat_replacement_preserves_construction_reserve(self):
        c = CoreController()
        bot = core.CoreBot(12, 12)
        bot.core_pos = bot.tile_cache.position_at(5, 5)
        bot.intruders_spawned = 1
        bot.intruder_spawn_round = 1
        self.assertFalse(bot.try_spawn_intruder(c))
        c.round = 200
        c.resources = (173, 0)
        self.assertFalse(bot.try_spawn_intruder(c))
        c.resources = (174, 0)
        self.assertTrue(bot.try_spawn_intruder(c))

    def test_intruder_marker_must_be_friendly_and_addressed(self):
        c = newborn(constants.MARKER_KIND_SPAWN_INTRUDER)
        p = player.Player()
        p.init_once(c)
        self.assertIsInstance(p.bot, player.IntruderBot)
        c = newborn(constants.MARKER_KIND_SPAWN_INTRUDER, Team.B)
        p = player.Player()
        p.init_once(c)
        self.assertIsInstance(p.bot, player.BuilderBot)
        c = newborn(constants.MARKER_KIND_SPAWN_INTRUDER)
        c.get_position.return_value = Position(5, 6)
        self.assertEqual(orders.read_spawn_assignment(c), (False, None))

    def test_worker_captures_fallback_sector_before_cache_scan(self):
        c = newborn(constants.MARKER_KIND_SPAWN_DIRECTION, payload=2)
        p = player.Player()
        p.init_once(c)
        self.assertIsInstance(p.bot, player.BuilderBot)
        self.assertEqual(p.bot.work_direction, Direction.EAST)

    def test_interrupted_initialization_remembers_role_after_marker_removal(self):
        class InterruptedIntruder:
            attempts = 0

            def __init__(self, width, height):
                type(self).attempts += 1
                if self.attempts == 1:
                    raise RuntimeError("simulated CPU interruption")

        c = newborn(constants.MARKER_KIND_SPAWN_INTRUDER)
        p = player.Player()
        with patch.object(player, "IntruderBot", InterruptedIntruder):
            with self.assertRaises(RuntimeError):
                p.init_once(c)
            self.assertFalse(p.initialized)
            c.get_nearby_buildings.return_value = []
            p.init_once(c)
        self.assertIsInstance(p.bot, InterruptedIntruder)
        self.assertTrue(p.initialized)

    def test_stationary_roles_are_dispatched(self):
        for kind, cls in ((EntityType.GUNNER, player.GunnerBot),
                          (EntityType.LAUNCHER, player.LauncherBot),
                          (EntityType.CORE, player.CoreBot)):
            c = newborn()
            c.get_entity_type.side_effect = None
            c.get_entity_type.return_value = kind
            p = player.Player()
            p.init_once(c)
            self.assertIsInstance(p.bot, cls)


class CombatIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.bot = player.IntruderBot(12, 12)
        self.bot.team = Team.A
        self.pos = self.bot.tile_cache.position_at(5, 5)
        self.bot.current_position = self.pos

    def remember(self, pos, kind, team=Team.A, direction=Direction.EAST):
        self.bot.known_env[pos] = Environment.EMPTY
        self.bot.tile_cache.remember_building(pos, 100 + pos.x * 12 + pos.y, kind, team, direction=direction)

    def test_economic_transport_cannot_be_repurposed_for_combat(self):
        for kind in (EntityType.CONVEYOR, EntityType.BRIDGE,
                     EntityType.SPLITTER, EntityType.ARMOURED_CONVEYOR):
            with self.subTest(kind=kind):
                self.remember(self.pos, kind)
                c = Mock()
                self.assertFalse(self.bot.is_supply_tile(self.pos))
                self.assertFalse(self.bot.is_supply_bridge_landing(self.pos))
                self.assertFalse(self.bot.is_gunner_site_locally_viable(self.pos))
                self.assertFalse(self.bot.clear_walkable_tile(c, self.pos))
                c.destroy.assert_not_called()

    def test_economic_harvester_is_not_diverted_to_gunner(self):
        self.remember(self.pos, EntityType.HARVESTER)
        self.assertFalse(self.bot.is_available_supply_ore(self.pos))
        self.remember(self.pos, EntityType.HARVESTER, Team.B)
        self.assertTrue(self.bot.is_available_supply_ore(self.pos))
        self.remember(self.pos, EntityType.HARVESTER)
        self.bot.owned_supply_ores.add(self.pos)
        self.assertTrue(self.bot.is_available_supply_ore(self.pos))

    def test_own_supply_and_enemy_logistics_remain_replaceable(self):
        self.remember(self.pos, EntityType.CONVEYOR)
        self.bot.owned_supply_tiles.add(self.pos)
        c = Mock()
        c.can_destroy.return_value = True
        self.assertTrue(self.bot.clear_walkable_tile(c, self.pos))
        c.destroy.assert_called_once_with(self.pos)
        self.remember(self.pos, EntityType.CONVEYOR, Team.B)
        self.assertTrue(self.bot.is_supply_tile(self.pos))
        c.get_hp.return_value = GameConstants.BUILDER_BOT_ATTACK_DAMAGE
        c.can_fire.return_value = True
        self.bot.clear_walkable_tile(c, self.pos)
        c.fire.assert_called_once_with(self.pos)

    def test_new_economic_obstacle_invalidates_committed_supply_route(self):
        self.remember(self.pos, EntityType.CONVEYOR)
        self.bot.supply_path = [self.pos]
        self.bot.supply_directions[self.pos] = Direction.EAST
        c = Mock()
        self.assertFalse(self.bot.build_supply_tile(c, self.pos))
        self.assertEqual(self.bot.supply_path, [])
        c.destroy.assert_not_called()

    def test_combat_astar_resumes_across_turns(self):
        state = self.bot.a_star_state("test")
        start = self.bot.tile_cache.position_at(0, 0)
        goal = self.bot.tile_cache.position_at(11, 0)
        path = []
        for _ in range(20):
            path = combat.a_star_to_any(None, start, {goal},
                                       lambda c, p: p.y == 0,
                                       self.bot.tile_cache.neighbor,
                                       max_expansions=1, state=state)
            if path:
                break
            self.assertTrue(state.pending)
        self.assertEqual(path[-1], goal)
        self.assertEqual(len(path), 11)
        self.assertFalse(state.pending)

    def test_launcher_only_throws_addressed_intruder(self):
        bot = player.LauncherBot(12, 12)
        bot.team = Team.A
        bot.current_position = bot.tile_cache.position_at(5, 5)
        source = bot.tile_cache.position_at(6, 5)
        worker = bot.tile_cache.position_at(4, 5)
        landing = bot.tile_cache.position_at(8, 5)
        marker = bot.tile_cache.position_at(5, 6)
        bot.tile_cache.visible_builder_ids = {worker: 20, source: 21}
        bot.tile_cache.entity_team = lambda _: Team.A
        bot._scan_turn = lambda *a, **k: False
        bot.read_launch_order = lambda: (landing, marker, source)
        c = Mock()
        c.can_launch.return_value = True
        c.can_destroy.return_value = True
        bot.run(c)
        c.launch.assert_called_once_with(source, landing)
        c.destroy.assert_called_once_with(marker)

    def test_launcher_marker_round_trip_includes_source(self):
        bot = player.LauncherBot(12, 12)
        bot.current_position = bot.tile_cache.position_at(5, 5)
        marker = bot.tile_cache.position_at(5, 6)
        landing = bot.tile_cache.position_at(8, 5)
        value = geometry.encode_marker(constants.MARKER_KIND_INTRUDER_LAUNCH, landing,
                                       constants.DIRECTIONS.index(Direction.EAST) + 1)
        bot.tile_cache.remember_building(marker, 42, EntityType.MARKER, Team.A, marker_value=value)
        bot.tile_cache.marker_ids = lambda: (42,)
        self.assertEqual(bot.read_launch_order(), (landing, marker, Position(6, 5)))

    def test_intruder_waits_for_launcher_scan_and_cancels_expired_order(self):
        bot = self.bot
        bot.waiting_launcher_origin = self.pos
        bot.waiting_launcher_round = 10
        bot.waiting_launch_marker = bot.tile_cache.position_at(5, 6)
        c = Mock()
        c.get_current_round.return_value = 15
        self.assertTrue(bot.wait_for_launcher(c, self.pos))
        c.destroy.assert_not_called()
        c.get_current_round.return_value = 10 + constants.LAUNCH_WAIT_ROUNDS + 1
        c.can_destroy.return_value = True
        self.assertFalse(bot.wait_for_launcher(c, self.pos))
        c.destroy.assert_called_once()

    def test_gunner_fires_at_engine_validated_target(self):
        bot = player.GunnerBot(12, 12)
        bot._scan_turn = lambda *a, **k: False
        c = Mock()
        c.get_gunner_target.return_value = Position(5, 5)
        c.can_fire.return_value = True
        bot.run(c)
        c.fire.assert_called_once_with(Position(5, 5))


if __name__ == "__main__":
    unittest.main()
