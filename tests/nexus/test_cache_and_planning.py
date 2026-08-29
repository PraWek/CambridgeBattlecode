from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

from cambc import Direction, EntityType, Position, Team


ROOT = Path(__file__).resolve().parents[2]
NEXUS = ROOT / "bots" / "nexus"


def load_module(alias: str, filename: str):
    spec = importlib.util.spec_from_file_location(alias, NEXUS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exploration = load_module("nexus_exploration_test", "exploration.py")
economy = load_module("nexus_economy_test", "economy.py")
network_planner = load_module("nexus_network_planner_test", "network_planner.py")
orders = load_module("nexus_orders_test", "orders.py")
tile_cache_module = load_module("nexus_tile_cache_test", "tile_cache.py")
TileCache = tile_cache_module.TileCache


class NexusPositionOwnershipTests(unittest.TestCase):
    def test_public_coordinate_accessors_share_one_pool(self) -> None:
        cache = TileCache(4, 3)
        external = Position(2, 1)

        self.assertIs(cache.canonicalize(external), cache.position_at(2, 1))
        self.assertIs(
            cache.neighbor(cache.position_at(1, 1), Direction.EAST),
            cache.position_at(2, 1),
        )
        self.assertIsNone(cache.offset(cache.position_at(0, 0), -1, 0))

    def test_nexus_allocates_positions_only_in_tile_cache_pool(self) -> None:
        violations: list[str] = []
        for source in NEXUS.glob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            parents: dict[ast.AST, ast.AST] = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "Position"
                ):
                    continue
                owner = parents.get(node)
                while owner is not None and not isinstance(owner, ast.FunctionDef):
                    owner = parents.get(owner)
                allowed = source.name == "tile_cache.py" and getattr(owner, "name", None) == "__init__"
                if not allowed:
                    violations.append(f"{source.name}:{node.lineno}")
        self.assertEqual(violations, [])

    def test_explicit_symmetry_uses_canonical_coordinates(self) -> None:
        cache = TileCache(5, 4)
        source = cache.position_at(1, 2)
        mirror = cache.mirrored_position(source, "vertical")

        self.assertIs(mirror, cache.position_at(3, 2))

    def test_crowded_core_scan_eventually_reads_all_order_markers(self) -> None:
        class Controller:
            @staticmethod
            def get_nearby_tiles() -> list[Position]:
                return []

            @staticmethod
            def get_nearby_units() -> list[int]:
                return [1, 2, 3, 4]

            @staticmethod
            def get_nearby_buildings() -> list[int]:
                return [10, 11, 12, 13]

            @staticmethod
            def get_position(entity_id: int) -> Position:
                return Position(entity_id % 5, entity_id // 5)

            @staticmethod
            def get_entity_type(entity_id: int) -> EntityType:
                return EntityType.BUILDER_BOT if entity_id < 10 else EntityType.MARKER

            @staticmethod
            def get_team(_entity_id: int) -> Team:
                return Team.A

            @staticmethod
            def get_marker_value(entity_id: int) -> int:
                return entity_id * 100

        cache = TileCache(5, 5)
        controller = Controller()
        completed = False
        for _ in range(20):
            cache.scan_turn(controller, own_id=1)
            if cache.scan_incomplete_this_turn:
                continue
            if cache.cache_friendly_marker_values(controller, Team.A):
                completed = True
                break

        self.assertTrue(completed)
        self.assertEqual(set(cache.marker_values), {10, 11, 12, 13})


class NexusExplorationTests(unittest.TestCase):
    def test_spawn_order_is_read_only_by_the_newborn_on_its_encoded_tile(self) -> None:
        cache = TileCache(5, 5)
        current = cache.position_at(2, 1)
        next_spawn = cache.position_at(3, 2)
        assert current is not None and next_spawn is not None
        records = [(7, next_spawn, 2), (7, current, 1)]

        selected = orders.spawn_order_for(
            records,
            current,
            {7, 8, 9},
            {1: Direction.NORTH, 2: Direction.EAST},
        )

        self.assertEqual(selected, (7, current, Direction.NORTH))

    def test_friendly_transport_never_poisoned_as_static_obstacle(self) -> None:
        self.assertFalse(
            exploration.is_static_step_obstacle(
                terrain_blocked=False,
                building_present=True,
                building_passable=True,
            )
        )
        self.assertTrue(
            exploration.is_static_step_obstacle(
                terrain_blocked=True,
                building_present=False,
                building_passable=False,
            )
        )

    def test_information_gain_precedes_revisit_noise(self) -> None:
        cache = TileCache(3, 3)
        current = cache.position_at(1, 1)
        east = cache.position_at(2, 1)
        north = cache.position_at(1, 0)
        assert current is not None and east is not None and north is not None

        selected = exploration.choose_information_gain_step(
            current=current,
            directions=(Direction.NORTH, Direction.EAST),
            neighbor=cache.neighbor,
            viable=lambda _pos: True,
            vision_gain=lambda pos: 9 if pos is east else 3,
            total_visits={east: 2, north: 0},
            recent_visits={},
            avoided=lambda _pos: False,
            forward_progress=lambda _pos: 0,
            sweep_bias=lambda _current, _candidate: 0,
            heading=Direction.NORTH,
            require_new_vision=True,
        )

        self.assertEqual(selected, (Direction.EAST, east))

    def test_heading_breaks_equal_information_ties(self) -> None:
        cache = TileCache(3, 3)
        current = cache.position_at(1, 1)
        east = cache.position_at(2, 1)
        assert current is not None and east is not None

        selected = exploration.choose_information_gain_step(
            current=current,
            directions=(Direction.NORTH, Direction.EAST),
            neighbor=cache.neighbor,
            viable=lambda _pos: True,
            vision_gain=lambda _pos: 5,
            total_visits={},
            recent_visits={},
            avoided=lambda _pos: False,
            forward_progress=lambda _pos: 0,
            sweep_bias=lambda _current, _candidate: 0,
            heading=Direction.EAST,
            require_new_vision=True,
        )

        self.assertEqual(selected, (Direction.EAST, east))

    def test_sector_progress_prevents_scouts_collapsing_into_one_quadrant(self) -> None:
        cache = TileCache(5, 3)
        current = cache.position_at(2, 1)
        east = cache.position_at(3, 1)
        west = cache.position_at(1, 1)
        assert current is not None and east is not None and west is not None

        selected = exploration.choose_information_gain_step(
            current=current,
            directions=(Direction.WEST, Direction.EAST),
            neighbor=cache.neighbor,
            viable=lambda _pos: True,
            vision_gain=lambda pos: 12 if pos is west else 8,
            total_visits={},
            recent_visits={},
            avoided=lambda _pos: False,
            forward_progress=lambda pos: pos.x - 2,
            sweep_bias=lambda _current, _candidate: 0,
            heading=Direction.EAST,
            require_new_vision=True,
        )

        self.assertEqual(selected, (Direction.EAST, east))

    def test_stall_recycling_requires_no_movement_and_no_useful_action(self) -> None:
        self.assertTrue(
            exploration.should_recycle_stalled_builder(96, 96, 96)
        )
        self.assertFalse(
            exploration.should_recycle_stalled_builder(96, 0, 96)
        )
        self.assertFalse(
            exploration.should_recycle_stalled_builder(3, 200, 96)
        )


class NexusCapacityTests(unittest.TestCase):
    def test_active_line_uses_exact_physical_capacity(self) -> None:
        decisions = [
            network_planner.starts_new_line(count, count > 0, 4)
            for count in range(9)
        ]
        self.assertEqual(
            decisions,
            [True, False, False, False, True, False, False, False, True],
        )

    def test_sector_core_entries_do_not_overlap(self) -> None:
        entries = [
            set(network_planner.sector_entry_offsets(direction))
            for direction in (
                Direction.NORTH,
                Direction.EAST,
                Direction.SOUTH,
                Direction.WEST,
            )
        ]
        self.assertEqual(sum(map(len, entries)), len(set().union(*entries)))

    def test_dedicated_route_tree_excludes_every_transport_lane(self) -> None:
        cache = TileCache(5, 1)
        core = cache.position_at(0, 0)
        owned_open = cache.position_at(1, 0)
        owned_full = cache.position_at(2, 0)
        foreign = cache.position_at(3, 0)
        assert all((core, owned_open, owned_full, foreign))

        network = {owned_open, owned_full, foreign}
        tree = network_planner.dedicated_route_tree({core})

        self.assertEqual(tree, {core})
        self.assertTrue(tree.isdisjoint(network))

class NexusEconomyTests(unittest.TestCase):
    def test_builder_budget_expands_gradually(self) -> None:
        desired = lambda turn: economy.desired_builder_count(
            turn, 4, 8, 100, 100,
        )

        self.assertEqual(desired(99), 4)
        self.assertEqual(desired(100), 5)
        self.assertEqual(desired(199), 5)
        self.assertEqual(desired(200), 6)
        self.assertEqual(desired(400), 8)


if __name__ == "__main__":
    unittest.main()
