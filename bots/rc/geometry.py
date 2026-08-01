from cambc import Position

from constants import MARKER_KIND_BASE, MARKER_X_BASE, MARKER_Y_BASE


def chebyshev(first: Position, second: Position) -> int:
    """Return the Chebyshev distance between two map positions."""
    return max(abs(first.x - second.x), abs(first.y - second.y))


def encode_marker(kind: int, pos: Position, payload: int = 0) -> int:
    """Pack a marker kind, position, and small payload into one integer."""
    return kind * MARKER_KIND_BASE + pos.x * MARKER_X_BASE + pos.y * MARKER_Y_BASE + payload


def decode_marker(value: int) -> tuple[int, Position, int]:
    """Unpack a marker integer into its kind, position, and payload fields."""
    kind = value // MARKER_KIND_BASE
    value %= MARKER_KIND_BASE
    x = value // MARKER_X_BASE
    value %= MARKER_X_BASE
    y = value // MARKER_Y_BASE
    payload = value % MARKER_Y_BASE
    return kind, Position(x, y), payload
