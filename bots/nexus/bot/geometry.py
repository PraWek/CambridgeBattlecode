from __future__ import annotations

from cambc import Controller, Position

from bot.constants import MARKER_KIND_BASE, MARKER_X_BASE, MARKER_Y_BASE


def chebyshev(a: Position, b: Position) -> int:
    return max(abs(a.x - b.x), abs(a.y - b.y))


def in_bounds(ct: Controller, pos: Position) -> bool:
    return 0 <= pos.x < ct.get_map_width() and 0 <= pos.y < ct.get_map_height()


def encode_marker(kind: int, pos: Position, payload: int = 0) -> int:
    return kind * MARKER_KIND_BASE + pos.x * MARKER_X_BASE + pos.y * MARKER_Y_BASE + payload


def decode_marker(value: int) -> tuple[int, Position, int]:
    kind = value // MARKER_KIND_BASE
    value %= MARKER_KIND_BASE
    x = value // MARKER_X_BASE
    value %= MARKER_X_BASE
    y = value // MARKER_Y_BASE
    payload = value % MARKER_Y_BASE
    return kind, Position(x, y), payload
