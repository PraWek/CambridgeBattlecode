from __future__ import annotations

import unittest

from analyze_nexus_replay import (
    Entity,
    final_network_metrics,
    grouped,
    packed_integers,
    read_varint,
    reachable_builder_vision,
)


class NexusReplayAnalyzerTests(unittest.TestCase):
    def test_varint_reader_handles_multiple_bytes(self) -> None:
        self.assertEqual(read_varint(bytes((0xAC, 0x02)), 0), (300, 2))

    def test_packed_enum_field_is_decoded(self) -> None:
        # field 1, length-delimited, with four packed enum values
        row = grouped(bytes((0x0A, 0x04, 0x00, 0x01, 0x02, 0x03)))
        self.assertEqual(packed_integers(row, 1), [0, 1, 2, 3])

    def test_reachable_vision_does_not_cross_solid_wall(self) -> None:
        environments = [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]

        visible = reachable_builder_vision((0, 1), environments, radius_sq=0)

        self.assertEqual(visible, {(0, 0), (0, 1), (0, 2)})

    def test_final_network_reports_only_unfinished_bridge_landings(self) -> None:
        entities = {
            1: Entity(1, 0, (2, 2), "bridge", bridge_target=(2, 5)),
            2: Entity(2, 0, (5, 2), "bridge", bridge_target=(5, 5)),
            3: Entity(3, 0, (5, 5), "conveyor", direction=5),
        }

        metrics = final_network_metrics(0, entities, (5, 7))

        self.assertEqual(metrics["dangling_bridge_tiles"], [(2, 2)])

    def test_final_network_flow_caps_a_shared_root_at_four_harvesters(self) -> None:
        entities = {
            1: Entity(1, 0, (1, 1), "conveyor", direction=3),
            2: Entity(2, 0, (2, 1), "conveyor", direction=3),
        }
        for index, position in enumerate(
            ((0, 1), (1, 0), (1, 2), (2, 0), (2, 2)),
            start=10,
        ):
            entities[index] = Entity(index, 0, position, "harvester")

        metrics = final_network_metrics(0, entities, (4, 1))

        self.assertEqual(metrics["connected_harvesters"], 5)
        self.assertEqual(metrics["max_flow_harvesters"], 4)
        self.assertEqual(len(metrics["capacity_blocked_harvesters"]), 1)


if __name__ == "__main__":
    unittest.main()
