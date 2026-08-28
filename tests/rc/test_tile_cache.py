"""Regression tests for the RC tile cache."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from cambc import Direction, EntityType, Environment, Position, Team


RC_BOT_DIRECTORY = Path(__file__).resolve().parents[2] / "bots" / "rc"
if str(RC_BOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(RC_BOT_DIRECTORY))

from base import BaseBot
from core_bot import CoreBot
from intruder_bot import IntruderBot
from navigation import a_star_to_any
from tile_cache import TileCache


class _ControllerWhoseFirstScanExhaustsTheBudget:
    """Minimal controller that leaves no scan call for its own entity on turn one."""

    def __init__(self, environments: dict[Position, Environment]) -> None:
        self.environments = environments

    def get_nearby_tiles(self) -> list[Position]:
        return list(self.environments)

    def get_tile_env(self, pos: Position) -> Environment:
        return self.environments[pos]

    @staticmethod
    def get_nearby_units() -> list[int]:
        return [1]

    @staticmethod
    def get_nearby_buildings() -> list[int]:
        return []

    @staticmethod
    def get_position(entity_id: int) -> Position:
        assert entity_id == 1
        return Position(2, 2)

    @staticmethod
    def get_entity_type(entity_id: int) -> EntityType:
        assert entity_id == 1
        return EntityType.BUILDER_BOT

    @staticmethod
    def get_team(entity_id: int) -> Team:
        assert entity_id == 1
        return Team.A


class _CrowdedEntityController(_ControllerWhoseFirstScanExhaustsTheBudget):
    """Expose enough dynamic builders to defer part of the entity refresh."""

    @staticmethod
    def get_nearby_units() -> list[int]:
        return [1, 2, 3]

    @staticmethod
    def get_position(entity_id: int) -> Position:
        return Position(entity_id, 2)

    @staticmethod
    def get_entity_type(entity_id: int) -> EntityType:
        return EntityType.BUILDER_BOT

    @staticmethod
    def get_team(entity_id: int) -> Team:
        return Team.A


class _FirstTurnCoreController:
    """Controller double that permits only the bootstrap Intruder actions."""

    def __init__(self) -> None:
        self.marker_positions: list[Position] = []
        self.spawned_positions: list[Position] = []
        self.scanned = False

    @staticmethod
    def get_id() -> int:
        return 7

    @staticmethod
    def get_team() -> Team:
        return Team.A

    @staticmethod
    def get_position() -> Position:
        return Position(1, 1)

    @staticmethod
    def get_unit_count() -> int:
        return 1

    @staticmethod
    def get_current_round() -> int:
        return 1

    def can_place_marker(self, pos: Position) -> bool:
        return pos == Position(3, 3)

    def place_marker(self, pos: Position, _value: int) -> int:
        self.marker_positions.append(pos)
        return 20

    @staticmethod
    def can_spawn(pos: Position) -> bool:
        return pos == Position(2, 2)

    def spawn_builder(self, pos: Position) -> None:
        self.spawned_positions.append(pos)

    def get_nearby_tiles(self) -> list[Position]:
        self.scanned = True
        raise AssertionError("the first-turn Intruder spawn must not scan tiles")


class TileCacheSymmetryTests(unittest.TestCase):
    def test_symmetry_evidence_survives_partial_scan(self) -> None:
        """Terrain read before a partial scan still proves symmetry next turn.

        The cache budget is 7 calls.  This controller exposes 6 new terrain
        cells, so ``get_nearby_tiles`` + terrain reads consumes all 7 calls.
        The first scan ends before ``_update_symmetry``
        runs.  A later complete scan sees no new terrain, but must still use
        those 6 observations and retain only vertical symmetry.
        """
        # This terrain is vertically symmetric, but deliberately not
        # horizontal, rotational, or diagonal.
        environments = {
            Position(1, 1): Environment.WALL,
            Position(3, 1): Environment.WALL,
            Position(1, 4): Environment.EMPTY,
            Position(3, 4): Environment.EMPTY,
            Position(0, 0): Environment.EMPTY,
            Position(4, 0): Environment.EMPTY,
        }
        controller = _ControllerWhoseFirstScanExhaustsTheBudget(environments)
        cache = TileCache(5, 6)

        cache.scan_turn(controller, own_id=1)

        self.assertTrue(cache.scan_incomplete_this_turn)
        self.assertIsNone(cache.confirmed_symmetry)
        self.assertEqual(len(cache._pending_symmetry_tiles), len(environments))

        cache.scan_turn(controller, own_id=1)

        self.assertFalse(cache.scan_incomplete_this_turn)
        self.assertEqual(cache.confirmed_symmetry, "vertical")
        self.assertEqual(cache.possible_symmetries, {"vertical"})
        self.assertFalse(cache._pending_symmetry_tiles)

        # A fresh direct observation after confirmation changes
        # ``observed_tiles``.  Historical backfill must continue from its
        # stable cursor rather than iterating that mutable set.
        controller.environments[Position(0, 1)] = Environment.EMPTY
        historical_index = cache._symmetry_backfill_historical_index

        cache.scan_turn(controller, own_id=1)
        self.assertEqual(cache._symmetry_backfill_historical_index, historical_index)

        cache.scan_turn(controller, own_id=1)
        self.assertEqual(
            cache._symmetry_backfill_historical_index,
            historical_index + 1,
        )

    def test_entity_budget_keeps_terrain_and_own_position_cached(self) -> None:
        """A crowded dynamic scan records the bot position before yielding."""
        cache = TileCache(5, 6)
        controller = _CrowdedEntityController({})

        cache.scan_turn(controller, own_id=1)

        self.assertTrue(cache.scan_incomplete_this_turn)
        self.assertTrue(cache.role_cache_ready_this_turn)
        self.assertEqual(cache.current_position, Position(1, 2))

    def test_base_bot_yields_after_partial_entity_scan(self) -> None:
        """No role action may use a partially rebuilt dynamic cache."""
        controller = _CrowdedEntityController({})
        controller.get_id = lambda: 1
        bot = BaseBot(5, 6)

        self.assertTrue(bot._scan_turn(controller))
        self.assertIsNone(bot.current_position)
        self.assertEqual(bot.tile_cache.current_position, Position(1, 2))

    def test_entity_scan_resumes_its_existing_queue_on_the_next_turn(self) -> None:
        """A crowded view eventually completes instead of restarting at its first ID."""
        cache = TileCache(5, 6)
        controller = _CrowdedEntityController({})

        cache.scan_turn(controller, own_id=1)
        self.assertTrue(cache.scan_incomplete_this_turn)
        self.assertEqual(cache._entity_scan_cursor, 1)

        cache.scan_turn(controller, own_id=1)

        self.assertTrue(cache.scan_incomplete_this_turn)
        self.assertEqual(cache._entity_scan_cursor, 2)

        cache.scan_turn(controller, own_id=1)

        self.assertFalse(cache.scan_incomplete_this_turn)
        self.assertFalse(cache._entity_scan_pending)
        self.assertEqual(cache.visible_entity_ids, {1, 2, 3})


class TileCacheNavigationTests(unittest.TestCase):
    def test_a_star_returns_preallocated_neighbor_positions(self) -> None:
        """A* must expand through TileCache rather than Position.add()."""
        cache = TileCache(3, 1)
        start = cache._positions[0][0]
        goal = cache._positions[2][0]
        walkable = {cache._positions[x][0] for x in range(3)}

        path = a_star_to_any(
            None,
            start,
            {goal},
            lambda _controller, pos: pos in walkable,
            cache.neighbor,
        )

        self.assertEqual(path, [cache._positions[1][0], goal])
        self.assertIs(path[0], cache._positions[1][0])
        self.assertIs(path[1], goal)

    def test_intruder_exploration_does_not_use_a_star(self) -> None:
        """Exploration stays local even when the opposing Core is confirmed."""
        bot = IntruderBot(5, 5)
        current = bot.tile_cache._positions[1][1]
        target = bot.tile_cache._positions[3][3]
        bot.get_cached_position = lambda: current
        bot.move_towards = lambda *args, **kwargs: False
        bot.start_exploration_wall_following = lambda *args, **kwargs: False
        bot.finish_unvisited_movement = lambda *args, **kwargs: False

        for confirmed_core in (False, True):
            bot.destination_is_confirmed_core = confirmed_core
            with patch("intruder_bot.a_star_to_any") as search:
                self.assertFalse(bot.advance_towards_unvisited_target(None, target))

            search.assert_not_called()

    def test_intruder_selects_nearest_core_edge_and_direct_firing_ray(self) -> None:
        """The first valid shot is checked beside the Intruder, not globally ranked."""
        bot = IntruderBot(20, 20)
        bot.destination = bot.tile_cache.position_at(10, 16)
        bot.destination_is_confirmed_core = True
        current = bot.tile_cache.position_at(13, 14)
        assert current is not None
        bot.known_env.update({
            pos: Environment.EMPTY
            for column in bot.tile_cache._positions
            for pos in column
        })

        calls: list[tuple[Position, Direction, EntityType, Position]] = []

        class _GunnerController:
            @staticmethod
            def can_fire_from(
                    site: Position,
                    facing: Direction,
                    entity_type: EntityType,
                    target: Position,
            ) -> bool:
                calls.append((site, facing, entity_type, target))
                return True

        site_data = bot.choose_gunner_site(_GunnerController(), current)

        expected_site = bot.tile_cache.position_at(13, 13)
        expected_target = bot.tile_cache.position_at(11, 15)
        self.assertEqual(site_data, (expected_site, Direction.SOUTHWEST))
        self.assertEqual(
            calls,
            [(expected_site, Direction.SOUTHWEST, EntityType.GUNNER, expected_target)],
        )

    def test_intruder_checks_next_near_core_edge_when_first_target_fails(self) -> None:
        """A failed target resumes through its rays before the next nearby edge."""
        bot = IntruderBot(20, 20)
        bot.destination = bot.tile_cache.position_at(10, 16)
        bot.destination_is_confirmed_core = True
        current = bot.tile_cache.position_at(13, 14)
        assert current is not None
        bot.known_env.update({
            pos: Environment.EMPTY
            for column in bot.tile_cache._positions
            for pos in column
        })

        first_target = bot.tile_cache.position_at(11, 15)
        second_target = bot.tile_cache.position_at(11, 16)
        assert first_target is not None
        assert second_target is not None
        checked_targets: list[Position] = []

        class _GunnerController:
            @staticmethod
            def can_fire_from(
                    _site: Position,
                    _facing: Direction,
                    _entity_type: EntityType,
                    target: Position,
            ) -> bool:
                checked_targets.append(target)
                return target == second_target

        controller = _GunnerController()
        self.assertIsNone(bot.choose_gunner_site(controller, current))
        self.assertTrue(bot.gunner_site_search_pending())
        self.assertIsNone(bot.choose_gunner_site(controller, current))
        self.assertTrue(bot.gunner_site_search_pending())
        self.assertIsNotNone(bot.choose_gunner_site(controller, current))
        self.assertEqual(set(checked_targets[:-1]), {first_target})
        self.assertEqual(checked_targets[-1], second_target)

    def test_intruder_keeps_untried_gunner_candidates_after_seven_engine_checks(self) -> None:
        """A seven-call Gunner probe budget yields without abandoning the search."""
        bot = IntruderBot(20, 20)
        bot.destination = bot.tile_cache.position_at(10, 16)
        bot.destination_is_confirmed_core = True
        current = bot.tile_cache.position_at(13, 14)
        assert current is not None
        bot.known_env.update({
            pos: Environment.EMPTY
            for column in bot.tile_cache._positions
            for pos in column
        })

        calls: list[tuple[Position, Direction, EntityType, Position]] = []

        class _GunnerController:
            @staticmethod
            def can_fire_from(
                    site: Position,
                    facing: Direction,
                    entity_type: EntityType,
                    target: Position,
            ) -> bool:
                calls.append((site, facing, entity_type, target))
                return False

        controller = _GunnerController()
        self.assertIsNone(bot.choose_gunner_site(controller, current))
        self.assertEqual(len(calls), 7)
        first_cursor = bot.gunner_site_candidate_cursor
        self.assertTrue(bot.gunner_site_search_pending())

        self.assertIsNone(bot.choose_gunner_site(controller, current))
        self.assertEqual(len(calls), 14)
        self.assertGreater(bot.gunner_site_candidate_cursor, first_cursor)
        self.assertTrue(bot.gunner_site_search_pending())

    def test_intruder_fans_out_after_the_three_nearest_core_edges(self) -> None:
        """Fallback targets are ordered by their distance from the direct approach."""
        bot = IntruderBot(20, 20)
        bot.destination = bot.tile_cache.position_at(10, 16)
        bot.destination_is_confirmed_core = True
        current = bot.tile_cache.position_at(13, 14)
        assert current is not None

        self.assertEqual(
            bot.ordered_gunner_targets(current),
            tuple(
                bot.tile_cache.position_at(x, y)
                for x, y in (
                    (11, 15),
                    (11, 16),
                    (10, 15),
                    (9, 17),
                    (9, 15),
                    (11, 17),
                    (9, 16),
                    (10, 17),
                )
            ),
        )

    def test_intruder_replans_supply_for_a_new_cheaper_titanium_deposit(self) -> None:
        """A later, nearer deposit replaces the provisional supply endpoint."""
        bot = IntruderBot(30, 30)
        gunner = bot.tile_cache.position_at(12, 12)
        old_ore = bot.tile_cache.position_at(22, 17)
        new_ore = bot.tile_cache.position_at(16, 17)
        old_tile = bot.tile_cache.position_at(21, 17)
        new_tile = bot.tile_cache.position_at(15, 17)
        assert all((gunner, old_ore, new_ore, old_tile, new_tile))
        bot.gunner_site = gunner
        bot.gunner_direction = Direction.EAST
        bot.known_env[old_ore] = Environment.ORE_TITANIUM
        old_plan = ([old_tile], {old_tile: Direction.WEST}, {}, 30)
        new_plan = ([new_tile], {new_tile: Direction.WEST}, {}, 12)
        bot.store_supply_plan(old_ore, old_plan)

        # This is the state immediately after the turn-137 scan: the route
        # to (22, 17) already exists and the nearer (16, 17) has just entered
        # the cache.
        bot.known_env[new_ore] = Environment.ORE_TITANIUM
        bot.plan_supply_route = lambda ore: new_plan if ore is new_ore else None

        self.assertTrue(bot.reconsider_supply_plan())
        self.assertIs(bot.supply_ore, new_ore)
        self.assertEqual(bot.supply_path, [new_tile])
        self.assertEqual(bot.supply_plan_cost, 12)


class CoreBootstrapTests(unittest.TestCase):
    def test_core_spawns_intruder_before_the_first_cache_scan(self) -> None:
        controller = _FirstTurnCoreController()
        bot = CoreBot(8, 8)

        bot.run(controller)

        self.assertFalse(controller.scanned)
        self.assertEqual(controller.marker_positions, [Position(3, 3)])
        self.assertEqual(controller.spawned_positions, [Position(2, 2)])
        self.assertTrue(bot.intruder_spawned)
        self.assertIs(bot.core_pos, bot.tile_cache.position_at(1, 1))


if __name__ == "__main__":
    unittest.main()
