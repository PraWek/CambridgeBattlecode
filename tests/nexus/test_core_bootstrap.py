from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

from cambc import Position


ROOT = Path(__file__).resolve().parents[2]
NEXUS = ROOT / "bots" / "nexus"
GENERIC_MODULES = (
    "base",
    "constants",
    "economy",
    "geometry",
    "orders",
    "tile_cache",
)


def load_core_module():
    previous = {name: sys.modules.get(name) for name in GENERIC_MODULES}
    sys.path.insert(0, str(NEXUS))
    try:
        spec = importlib.util.spec_from_file_location(
            "nexus_core_bootstrap_test",
            NEXUS / "core_bot.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(NEXUS))
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


core_bot_module = load_core_module()


class NexusCoreBootstrapTests(unittest.TestCase):
    def test_first_cardinal_builder_spawns_before_full_cache_scan(self) -> None:
        class Controller:
            spawned: list[Position] = []

            @staticmethod
            def get_id() -> int:
                return 1

            @staticmethod
            def get_position(_entity_id: int) -> Position:
                return Position(5, 5)

            @staticmethod
            def get_unit_count() -> int:
                return 1

            @staticmethod
            def get_current_round() -> int:
                return 1

            @staticmethod
            def can_spawn(_pos: Position) -> bool:
                return True

            @classmethod
            def spawn_builder(cls, pos: Position) -> None:
                cls.spawned.append(pos)

        bot = core_bot_module.CoreBot(11, 11)
        bot.run(Controller())

        self.assertEqual(Controller.spawned, [Position(5, 4)])
        self.assertEqual(len(bot.initial_spawned_directions), 1)


if __name__ == "__main__":
    unittest.main()
