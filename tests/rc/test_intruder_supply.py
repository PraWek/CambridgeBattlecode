"""Exercise supply construction from either end of an Intruder's route."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from cambc import Direction, EntityType, Environment, Position, Team

RC_BOT_DIRECTORY = Path(__file__).resolve().parents[2] / "bots" / "rc"
if str(RC_BOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(RC_BOT_DIRECTORY))

from intruder_bot import IntruderBot


class _SupplyController:
    """Enforce action range, one build per turn, and walkable destinations."""

    def __init__(self, bot: IntruderBot) -> None:
        self.bot = bot
        self.action_used = False
        self.move_used = False
        self.built: list[tuple[EntityType, Position, object]] = []
        self.moves: list[Position] = []

    def turn(self) -> None:
        self.action_used = False
        self.move_used = False
        self.bot.supply_gunner(self, self.bot.current_position)

    def can_build(self, pos: Position) -> bool:
        return (
            not self.action_used
            and self.bot.current_position.distance_squared(pos) <= 2
            and self.bot.known_buildings.get(pos) is None
        )

    def build(self, kind: EntityType, pos: Position, output=None) -> int:
        assert self.can_build(pos)
        self.action_used = True
        self.built.append((kind, pos, output))
        return 1000 + len(self.built)

    def can_build_harvester(self, pos: Position) -> bool:
        return self.can_build(pos) and self.bot.known_env[pos] == Environment.ORE_TITANIUM

    def build_harvester(self, pos: Position) -> int:
        return self.build(EntityType.HARVESTER, pos)

    def can_build_conveyor(self, pos: Position, direction: Direction) -> bool:
        return self.can_build(pos) and self.bot.known_env[pos] == Environment.EMPTY

    def build_conveyor(self, pos: Position, direction: Direction) -> int:
        return self.build(EntityType.CONVEYOR, pos, direction)

    def can_build_road(self, pos: Position) -> bool:
        return self.can_build_conveyor(pos, Direction.NORTH)

    def build_road(self, pos: Position) -> int:
        return self.build(EntityType.ROAD, pos)

    def can_build_bridge(self, pos: Position, target: Position) -> bool:
        return (
            self.can_build(pos)
            and pos != self.bot.current_position
            and pos.distance_squared(target) <= 9
        )

    def build_bridge(self, pos: Position, target: Position) -> int:
        assert self.can_build_bridge(pos, target)
        return self.build(EntityType.BRIDGE, pos, target)

    def can_destroy(self, pos: Position) -> bool:
        return (
            not self.action_used
            and self.bot.current_position.distance_squared(pos) <= 2
            and self.bot.known_buildings.get(pos) is not None
        )

    def destroy(self, pos: Position) -> None:
        assert self.can_destroy(pos)
        self.action_used = True

    def can_move(self, direction: Direction) -> bool:
        target = self.bot.tile_cache.neighbor(self.bot.current_position, direction)
        return not self.move_used and target is not None and self.bot.is_cached_tile_passable(target)

    def move(self, direction: Direction) -> None:
        assert self.can_move(direction)
        self.move_used = True
        self.bot.current_position = self.bot.tile_cache.neighbor(self.bot.current_position, direction)
        self.moves.append(self.bot.current_position)


class IntruderSupplyTests(unittest.TestCase):
    def make_route(self, start: tuple[int, int], bridge: bool = False) -> IntruderBot:
        bot = IntruderBot(14, 10)
        bot.team = Team.A
        bot.known_env.update({
            pos: Environment.EMPTY
            for column in bot.tile_cache._positions
            for pos in column
        })
        pos = bot.tile_cache.position_at
        bot.current_position = pos(*start)
        bot.gunner_site = pos(1, 4)
        bot.gunner_direction = Direction.WEST
        ore = pos(11, 4)
        bot.known_env[ore] = Environment.ORE_TITANIUM
        path = [pos(x, 4) for x in range(2, 11) if not bridge or x not in (5, 6)]
        bridges = {pos(7, 4): pos(4, 4)} if bridge else {}
        directions = {tile: Direction.WEST for tile in path if tile not in bridges}
        bot.store_supply_plan(ore, (path, directions, bridges, 30))
        if bridge:
            bot.known_env[pos(5, 4)] = Environment.WALL
            bot.known_env[pos(6, 4)] = Environment.WALL
        return bot

    def remember_conveyor(self, bot: IntruderBot, x: int) -> None:
        bot.tile_cache.remember_building(
            bot.tile_cache.position_at(x, 4), x, EntityType.CONVEYOR, Team.A,
            direction=Direction.WEST,
        )

    def finish(self, bot: IntruderBot, controller: _SupplyController) -> None:
        for _ in range(100):
            controller.turn()
            if bot.mode == "complete":
                break
        self.assertEqual(bot.mode, "complete")
        self.assertTrue(all(bot.supply_tile_complete(tile) for tile in bot.supply_path))
        self.assertEqual(bot.known_buildings[bot.supply_ore][0], EntityType.HARVESTER)
        for kind, _, output in controller.built:
            if kind == EntityType.CONVEYOR:
                self.assertEqual(output, Direction.WEST)

    def test_near_ore_builds_back_to_gunner_and_does_not_return_for_mine(self) -> None:
        bot = self.make_route((10, 5))  # Off the route, as after scouting.
        controller = _SupplyController(bot)
        self.finish(bot, controller)
        conveyors = [pos.x for kind, pos, _ in controller.built if kind == EntityType.CONVEYOR]
        self.assertEqual(conveyors, list(range(10, 1, -1)))
        self.assertEqual(controller.built[0][0], EntityType.HARVESTER)
        self.assertEqual(bot.current_position.x, 2)

    def test_near_gunner_builds_toward_ore(self) -> None:
        bot = self.make_route((2, 4))
        controller = _SupplyController(bot)
        self.finish(bot, controller)
        conveyors = [pos.x for kind, pos, _ in controller.built if kind == EntityType.CONVEYOR]
        self.assertEqual(conveyors, list(range(2, 11)))
        self.assertEqual(controller.built[-1][0], EntityType.HARVESTER)

    def test_starry_night_turn_140_approaches_ore_instead_of_returning(self) -> None:
        bot = IntruderBot(50, 41)
        bot.team = Team.B
        pos = bot.tile_cache.position_at
        bot.current_position = pos(43, 27)
        bot.gunner_site = pos(22, 27)
        ore = pos(41, 31)
        bot.known_env.update({
            tile: Environment.EMPTY
            for column in bot.tile_cache._positions
            for tile in column
        })
        bot.known_env[ore] = Environment.ORE_TITANIUM
        path = [pos(x, 27) for x in range(23, 42)] + [pos(41, y) for y in range(28, 31)]
        directions = {
            tile: tile.direction_to(bot.gunner_site if i == 0 else path[i - 1])
            for i, tile in enumerate(path)
        }
        bot.store_supply_plan(ore, (path, directions, {}, 66))
        controller = _SupplyController(bot)
        controller.turn()
        self.assertEqual(bot.supply_build_step, -1)
        self.assertLess(bot.current_position.distance_squared(ore), pos(43, 27).distance_squared(ore))
        self.assertGreater(bot.current_position.y, 27)

    def test_compares_against_built_line_end_instead_of_gunner(self) -> None:
        bot = self.make_route((7, 4))
        for x in range(2, 7):
            self.remember_conveyor(bot, x)
        controller = _SupplyController(bot)
        self.finish(bot, controller)
        conveyors = [pos.x for kind, pos, _ in controller.built if kind == EntityType.CONVEYOR]
        self.assertEqual(conveyors, [7, 8, 9, 10])

    def test_reverse_build_joins_existing_prefix_and_reuses_suffix(self) -> None:
        bot = self.make_route((10, 4))
        for x in (2, 3, 9, 10):
            self.remember_conveyor(bot, x)
        controller = _SupplyController(bot)
        self.finish(bot, controller)
        conveyors = [pos.x for kind, pos, _ in controller.built if kind == EntityType.CONVEYOR]
        self.assertEqual(conveyors, [8, 7, 6, 5, 4])
        self.assertEqual(bot.current_position.x, 4)

    def test_equal_distances_choose_gunner_side(self) -> None:
        bot = self.make_route((6, 4))
        controller = _SupplyController(bot)
        self.finish(bot, controller)
        conveyors = [pos.x for kind, pos, _ in controller.built if kind == EntityType.CONVEYOR]
        self.assertEqual(conveyors, list(range(2, 11)))

    def test_new_plan_and_reset_clear_previous_build_direction(self) -> None:
        bot = self.make_route((10, 4))
        bot.choose_supply_build_direction(bot.current_position)
        self.assertEqual(bot.supply_build_step, -1)
        bot.store_supply_plan(bot.supply_ore, (bot.supply_path, bot.supply_directions, {}, 30))
        self.assertIsNone(bot.supply_build_step)
        bot.current_position = bot.supply_path[0]
        bot.choose_supply_build_direction(bot.current_position)
        self.assertEqual(bot.supply_build_step, 1)
        bot.reset_supply_plan()
        self.assertIsNone(bot.supply_build_step)

    def test_bridge_keeps_ore_to_gunner_output_in_both_build_directions(self) -> None:
        for start in ((2, 4), (10, 4), (7, 4)):
            with self.subTest(start=start):
                bot = self.make_route(start, bridge=True)
                controller = _SupplyController(bot)
                # A Launcher can land only on existing transport.  Seed roads
                # at both endpoints, then simulate just the launch movement.
                for x in (4, 7):
                    bot.tile_cache.remember_building(
                        bot.tile_cache.position_at(x, 4), x, EntityType.ROAD, Team.A,
                    )

                def launch(_controller, current, direction, destination, exact_landing=False):
                    self.assertTrue(exact_landing)
                    self.assertTrue(bot.is_cached_tile_passable(destination))
                    bot.current_position = destination
                    return True

                with patch.object(bot, "start_launcher_crossing", side_effect=launch):
                    self.finish(bot, controller)
                self.assertEqual(
                    [(pos, target) for kind, pos, target in controller.built if kind == EntityType.BRIDGE],
                    [(bot.tile_cache.position_at(7, 4), bot.tile_cache.position_at(4, 4))],
                )

    def test_reverse_bridge_with_bare_landing_uses_walkable_detour(self) -> None:
        bot = self.make_route((10, 4), bridge=True)
        controller = _SupplyController(bot)
        with patch.object(bot, "start_launcher_crossing") as launch:
            self.finish(bot, controller)
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
