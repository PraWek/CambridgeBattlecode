"""Enforce canonical Position ownership in the RC bot implementation."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

from cambc import Position


ROOT = Path(__file__).resolve().parents[2]
RC_BOT_DIRECTORY = ROOT / "bots" / "rc"
if str(RC_BOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(RC_BOT_DIRECTORY))

from tile_cache import TileCache


class _PositionOwnershipVisitor(ast.NodeVisitor):
    """Report allocations or coordinate ``.add`` calls in one source file."""

    def __init__(self, source: Path) -> None:
        self.source = source
        self.function_stack: list[str] = []
        self.violations: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "Position":
            if not (
                self.source.name == "tile_cache.py"
                and self.function_stack == ["__init__"]
            ):
                self.violations.append(
                    f"{self.source.relative_to(ROOT)}:{node.lineno}: Position(...)"
                )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add":
            self.violations.append(
                f"{self.source.relative_to(ROOT)}:{node.lineno}: .add(...)"
            )
        self.generic_visit(node)


class PositionOwnershipTests(unittest.TestCase):
    def test_tile_cache_public_accessors_return_canonical_positions(self) -> None:
        cache = TileCache(3, 2)
        external = Position(1, 1)

        self.assertIs(cache.canonicalize(external), cache.position_at(1, 1))
        self.assertIs(cache.offset(cache.position_at(1, 1), -1, 0), cache.position_at(0, 1))
        self.assertIsNone(cache.position_at(-1, 0))
        self.assertIsNone(cache.offset(cache.position_at(0, 0), -1, 0))

    def test_rc_source_cannot_allocate_or_add_positions(self) -> None:
        """Ensure TileCache is the only source of RC ``Position`` objects.

        The check parses every module in ``bots/rc`` rather than executing
        gameplay paths, so an accidental ``Position(...)`` allocation or
        ``position.add(...)`` call is caught even in rarely used code.
        ``Position(...)`` is permitted solely in ``TileCache.__init__``,
        where the fixed map-wide coordinate pool is created.  All other
        coordinate derivation must use TileCache accessors so callers share
        canonical objects and do not allocate during a turn.
        """
        violations: list[str] = []
        for source in RC_BOT_DIRECTORY.glob("*.py"):
            visitor = _PositionOwnershipVisitor(source)
            visitor.visit(ast.parse(source.read_text(encoding="utf-8")))
            violations.extend(visitor.violations)

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
