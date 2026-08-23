"""Regression tests for the RC tile cache."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from cambc import EntityType, Environment, Position, Team


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

    def test_entity_budget_keeps_terrain_and_own_position_usable(self) -> None:
        """A crowded dynamic scan may defer entities without freezing the role."""
        cache = TileCache(5, 6)
        controller = _CrowdedEntityController({})

        cache.scan_turn(controller, own_id=1)

        self.assertTrue(cache.scan_incomplete_this_turn)
        self.assertTrue(cache.role_cache_ready_this_turn)
        self.assertEqual(cache.current_position, Position(1, 2))

    def test_base_bot_continues_after_partial_entity_scan(self) -> None:
        """The role runs once terrain and its own position are cached."""
        controller = _CrowdedEntityController({})
        controller.get_id = lambda: 1
        bot = BaseBot(5, 6)

        self.assertFalse(bot._scan_turn(controller))
        self.assertEqual(bot.get_cached_position(), Position(1, 2))


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

    def test_intruder_prepares_gunner_candidates_incrementally(self) -> None:
        """Gunner-site enumeration must not become one repeatable TLE turn."""
        bot = IntruderBot(10, 10)
        bot.destination = bot.tile_cache.position_at(5, 5)
        bot.destination_is_confirmed_core = True

        bot.choose_gunner_site(None)

        self.assertIsNone(bot.gunner_site_candidates)
        self.assertIsNotNone(bot.gunner_candidate_targets)

        # Eight Core-edge tiles have twenty position/facing/distance probes
        # each.  The bounded preparer consumes 48 per call, then retains the
        # completed (possibly empty) ranking for validation next turn.
        for _ in range(3):
            bot.choose_gunner_site(None)

        self.assertEqual(bot.gunner_site_candidates, [])
        self.assertIsNone(bot.gunner_candidate_targets)


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
