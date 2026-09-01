from __future__ import annotations

import ast
import importlib.util
import sys
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
network_memory = load_module("nexus_network_memory_test", "network_memory.py")
sys.path.insert(0, str(NEXUS))
try:
    navigation = load_module("nexus_navigation_test", "navigation.py")
finally:
    sys.path.remove(str(NEXUS))
    for generic_module in ("constants", "geometry"):
        loaded = sys.modules.get(generic_module)
        if loaded is not None and str(NEXUS) in str(getattr(loaded, "__file__", "")):
            del sys.modules[generic_module]
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
    def test_enemy_approach_progress_becomes_positive_beyond_home_core(self) -> None:
        origin = Position(2, 8)
        enemy = Position(2, 1)

        self.assertEqual(
            exploration.target_approach_progress(origin, enemy, Position(2, 6)),
            2,
        )
        self.assertLess(
            exploration.target_approach_progress(origin, enemy, Position(2, 9)),
            0,
        )

    def test_sweep_patrol_chooses_reachable_far_tile_not_unreachable_candidate(self) -> None:
        cache = TileCache(5, 2)
        start = cache.position_at(0, 0)
        far = cache.position_at(4, 0)
        assert start is not None and far is not None

        path = navigation.breadth_first_sweep_path(
            start,
            traversable_fn=lambda pos: pos.y == 0,
            neighbor_fn=cache.neighbor,
            # Patrol endpoints remain far apart even after the remote end has
            # accumulated more visits than a neighbouring tile.
            visit_counts={far: 99},
            movement_directions=(Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST),
            max_expansions=16,
        )

        self.assertEqual(path[-1], far)
        self.assertEqual(len(path), 4)

    def test_cardinal_spawn_does_not_depend_on_handoff_marker(self) -> None:
        cache = TileCache(3, 3)
        preferred = cache.position_at(1, 0)
        fallback = cache.position_at(2, 1)
        assert preferred is not None and fallback is not None

        self.assertFalse(orders.spawn_needs_handoff(preferred, preferred))
        self.assertTrue(orders.spawn_needs_handoff(fallback, preferred))

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

    def test_two_tile_patrol_is_recycled_even_after_network_discovery(self) -> None:
        self.assertTrue(
            exploration.should_recycle_exhausted_scout(
                240, 0, 0, 12,
                240, 12, 24, 12,
                has_pending_repairs=False,
            )
        )
        self.assertFalse(
            exploration.should_recycle_exhausted_scout(
                240, 0, 0, 12,
                240, 12, 24, 12,
                has_pending_repairs=True,
            )
        )


class NexusCapacityTests(unittest.TestCase):
    def test_recovered_mine_can_merge_only_into_a_line_with_downstream_capacity(self) -> None:
        cache = TileCache(5, 2)
        upper = [cache.position_at(x, 0) for x in range(1, 5)]
        lower = [cache.position_at(x, 1) for x in range(1, 5)]
        core_upper = cache.position_at(0, 0)
        core_lower = cache.position_at(0, 1)
        assert all(upper + lower) and core_upper is not None and core_lower is not None
        receiver = {
            upper[index]: upper[index - 1] if index else core_upper
            for index in range(4)
        }
        receiver.update({
            lower[index]: lower[index - 1] if index else core_lower
            for index in range(4)
        })
        loads = {pos: 4 for pos in upper}
        loads.update({pos: 2 for pos in lower})

        merge_tree = network_planner.residual_capacity_tree(
            set(upper + lower),
            loads,
            receiver.get,
            lambda pos: pos in {core_upper, core_lower},
            capacity=4,
        )

        self.assertTrue(set(lower).issubset(merge_tree))
        self.assertTrue(set(upper).isdisjoint(merge_tree))

    def test_bridge_search_crosses_room_wall_instead_of_exhausting_open_room(self) -> None:
        cache = TileCache(9, 5)
        starts = [cache.position_at(8, 0), cache.position_at(8, 4)]
        anchor = cache.position_at(0, 0)
        blocked = {
            cache.position_at(4, y)
            for y in range(4)
        }
        assert all(starts) and anchor is not None and None not in blocked

        def bridge_crosses_block(current, direction, distance):
            dx, dy = direction.delta()
            return any(
                cache.offset(current, dx * step, dy * step) in blocked
                for step in range(1, distance)
            )

        result = network_planner.minimum_cost_bridge_route(
            starts,
            {anchor},
            [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST],
            cache.neighbor,
            cache.offset,
            lambda pos: pos not in blocked,
            bridge_crosses_block,
            lambda _anchor, _source: True,
            conveyor_cost=3,
            bridge_cost=20,
            max_jump_distance=3,
            max_expansions=32,
        )

        self.assertIsNotNone(result)
        nodes, bridge_targets, cost = result
        self.assertIn(nodes[0], starts)
        self.assertEqual(nodes[-1], anchor)
        self.assertTrue(bridge_targets)
        self.assertLess(cost, 48)

    def test_bridge_search_skips_a_forbidden_core_entry(self) -> None:
        cache = TileCache(5, 2)
        start = cache.position_at(4, 0)
        forbidden = cache.position_at(2, 0)
        allowed = cache.position_at(0, 0)
        assert start is not None and forbidden is not None and allowed is not None

        result = network_planner.minimum_cost_bridge_route(
            [start],
            {forbidden, allowed},
            [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST],
            cache.neighbor,
            cache.offset,
            lambda _pos: True,
            lambda _current, _direction, _distance: False,
            lambda anchor, _source: anchor == allowed,
            conveyor_cost=3,
            bridge_cost=20,
            max_jump_distance=3,
            max_expansions=32,
        )

        self.assertIsNotNone(result)
        nodes, _, _ = result
        self.assertEqual(nodes[-1], allowed)
        self.assertNotIn(forbidden, nodes)

    def test_min_cost_flow_prefers_cheaper_residual_augmentation(self) -> None:
        cache = TileCache(5, 3)
        start = cache.position_at(0, 1)
        upper_anchor = cache.position_at(4, 0)
        lower_anchor = cache.position_at(4, 2)
        assert all((start, upper_anchor, lower_anchor))

        result = network_planner.minimum_cost_flow_augmentation(
            [start],
            {upper_anchor, lower_anchor},
            [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST],
            cache.neighbor,
            cache.offset,
            lambda _pos: True,
            lambda _pos, _direction, _distance: False,
            lambda _receiver, _source: True,
            conveyor_cost=3,
            bridge_cost=20,
            max_jump_distance=3,
            max_expansions=64,
            conveyor_cost_fn=lambda pos, _direction: 1 if pos.y == 2 else 3,
            anchor_cost_fn=lambda anchor: 0 if anchor is lower_anchor else 3,
            minimum_conveyor_cost=1,
        )

        self.assertIsNotNone(result)
        nodes, _, _ = result
        self.assertEqual(nodes[-1], lower_anchor)

    def test_min_cost_flow_never_uses_forbidden_existing_direction(self) -> None:
        cache = TileCache(4, 2)
        start = cache.position_at(0, 0)
        anchor = cache.position_at(3, 0)
        blocked = cache.position_at(1, 0)
        assert all((start, anchor, blocked))

        result = network_planner.minimum_cost_flow_augmentation(
            [start],
            {anchor},
            [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST],
            cache.neighbor,
            cache.offset,
            lambda _pos: True,
            lambda _pos, _direction, _distance: False,
            lambda _receiver, _source: True,
            conveyor_cost=3,
            bridge_cost=20,
            max_jump_distance=3,
            max_expansions=32,
            edge_usable_fn=lambda pos, direction: not (
                pos is blocked and direction == Direction.EAST
            ),
        )

        self.assertIsNotNone(result)
        nodes, _, _ = result
        self.assertNotEqual(nodes, [start, blocked, cache.position_at(2, 0), anchor])

    def test_capacity_relief_requires_every_output_to_be_overloaded(self) -> None:
        self.assertTrue(
            network_planner.receivers_need_capacity_relief([(5, False)], 4)
        )
        self.assertTrue(
            network_planner.receivers_need_capacity_relief([(4, True)], 4)
        )
        self.assertFalse(
            network_planner.receivers_need_capacity_relief(
                [(5, False), (2, False)],
                4,
            )
        )

    def test_max_flow_respects_one_shared_transport_bottleneck(self) -> None:
        cache = TileCache(5, 3)
        first = cache.position_at(1, 1)
        second = cache.position_at(2, 1)
        core = cache.position_at(3, 1)
        mines = [cache.position_at(0, y) for y in range(3)]
        assert all((first, second, core, *mines))
        receivers = {first: second, second: core}

        value, loads, served = network_planner.transport_max_flow(
            {first, second, core},
            {mine: [first] for mine in mines},
            receivers.get,
            lambda pos: pos is core,
            tile_capacity=2,
        )

        self.assertEqual(value, 2)
        self.assertEqual(loads[first], 2)
        self.assertEqual(loads[second], 2)
        self.assertEqual(len(served), 2)

    def test_max_flow_balances_harvesters_across_independent_lanes(self) -> None:
        cache = TileCache(4, 3)
        upper = cache.position_at(1, 0)
        lower = cache.position_at(1, 2)
        upper_core = cache.position_at(2, 0)
        lower_core = cache.position_at(2, 2)
        mine_a = cache.position_at(0, 1)
        mine_b = cache.position_at(1, 1)
        assert all((upper, lower, upper_core, lower_core, mine_a, mine_b))
        receivers = {upper: upper_core, lower: lower_core}

        value, loads, served = network_planner.transport_max_flow(
            {upper, lower, upper_core, lower_core},
            {mine_a: [upper, lower], mine_b: [upper, lower]},
            receivers.get,
            lambda pos: pos in {upper_core, lower_core},
            tile_capacity=1,
        )

        self.assertEqual(value, 2)
        self.assertEqual(loads[upper], 1)
        self.assertEqual(loads[lower], 1)
        self.assertEqual(served, {mine_a, mine_b})

    def test_builder_rejects_a_virtual_branch_without_flow_gain(self) -> None:
        generic_modules = (
            "base",
            "constants",
            "exploration",
            "geometry",
            "navigation",
            "network_memory",
            "network_planner",
            "orders",
            "tile_cache",
        )
        shadowed = {name: sys.modules.get(name) for name in generic_modules}
        for name in generic_modules:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(NEXUS))
        try:
            builder_module = load_module("nexus_builder_flow_test", "builder_bot.py")
        finally:
            sys.path.remove(str(NEXUS))
            for name in generic_modules:
                sys.modules.pop(name, None)
            sys.modules.update(
                (name, module)
                for name, module in shadowed.items()
                if module is not None
            )
        bot = builder_module.BuilderBot(8, 5)
        bot.team = Team.A
        bot.core_pos = bot.tile_cache.position_at(6, 2)
        existing_mine = bot.tile_cache.position_at(3, 2)
        existing_lane = bot.tile_cache.position_at(4, 2)
        candidate = bot.tile_cache.position_at(3, 1)
        new_lane = bot.tile_cache.position_at(4, 1)
        assert all((existing_mine, existing_lane, candidate, new_lane))
        bot.known_buildings[existing_mine] = (EntityType.HARVESTER, Team.A)
        bot.known_buildings[existing_lane] = (EntityType.CONVEYOR, Team.A)
        bot.known_conveyor_directions[existing_lane] = Direction.EAST
        network = bot.core_receiver_tiles() | {existing_lane}

        self.assertTrue(
            bot.plan_raises_transport_flow(
                candidate,
                network,
                [new_lane],
                {new_lane: Direction.EAST},
                {},
                bot.core_entry_tiles(),
            )
        )
        self.assertFalse(
            bot.plan_raises_transport_flow(
                candidate,
                network,
                [new_lane],
                {new_lane: Direction.WEST},
                {},
                bot.core_entry_tiles(),
            )
        )

    def test_bridge_execution_returns_to_sources_after_building_landing(self) -> None:
        cache = TileCache(5, 2)
        source = cache.position_at(0, 0)
        detour_a = cache.position_at(1, 1)
        detour_b = cache.position_at(2, 1)
        landing = cache.position_at(3, 0)
        downstream = cache.position_at(4, 0)
        assert all((source, detour_a, detour_b, landing, downstream))
        forward = [source, detour_a, detour_b, landing, downstream]

        path, activation = network_planner.bridge_safe_execution_path(
            forward,
            {source},
        )

        self.assertEqual(activation, len(forward))
        self.assertEqual(
            path,
            forward + [landing, detour_b, detour_a, source],
        )

    def test_repointed_transport_cannot_be_repaired_from_stale_blueprint(self) -> None:
        cache = TileCache(2, 1)
        lane = cache.position_at(1, 0)
        assert lane is not None
        memory = network_memory.NetworkMemory(4)
        memory.remember(lane, EntityType.CONVEYOR, Direction.WEST)
        memory.audit(lane, None, False, None, None)

        memory.forget(lane)

        self.assertNotIn(lane, memory.blueprint)
        self.assertNotIn(lane, memory.damaged_tiles)

    def test_observed_allied_transport_is_available_for_replacement_patrol(self) -> None:
        cache = TileCache(2, 1)
        lane = cache.position_at(1, 0)
        assert lane is not None
        memory = network_memory.NetworkMemory(4)

        memory.remember(lane, EntityType.CONVEYOR, Direction.WEST)

        self.assertEqual(memory.patrol_tiles(), {lane})

    def test_connection_replan_queues_only_dropped_unconnected_tiles(self) -> None:
        cache = TileCache(4, 1)
        reused = cache.position_at(0, 0)
        dropped = cache.position_at(1, 0)
        rescued = cache.position_at(2, 0)
        assert all((reused, dropped, rescued))
        memory = network_memory.NetworkMemory(4)
        for tile in (reused, dropped, rescued):
            memory.record_unfinished_owned_tile(tile)

        memory.replan_owned_branch({reused}, {rescued})

        self.assertEqual(memory.unfinished_owned_tiles, {reused})
        self.assertEqual(memory.abandoned_owned_tiles, {dropped})

    def test_new_plan_rescues_previously_abandoned_transport(self) -> None:
        cache = TileCache(2, 1)
        lane = cache.position_at(1, 0)
        assert lane is not None
        memory = network_memory.NetworkMemory(4)
        memory.record_unfinished_owned_tile(lane)
        memory.abandon_owned_branch(set())

        memory.replan_owned_branch({lane}, set())

        self.assertNotIn(lane, memory.abandoned_owned_tiles)
        self.assertIn(lane, memory.unfinished_owned_tiles)

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
