from cambc import Direction

from constants import ASSIGNMENT_DIRECTIONS, MARKER_KIND_MASK


def encode_marker(kind: int, payload: int) -> int:
    return kind * MARKER_KIND_MASK + payload


def decode_marker(value: int) -> tuple[int, int]:
    return value // MARKER_KIND_MASK, value % MARKER_KIND_MASK


def direction_index(direction: Direction) -> int:
    return ASSIGNMENT_DIRECTIONS.index(direction)


def direction_to_vector(direction: Direction) -> tuple[int, int]:
    mapping = {
        Direction.NORTH: (0, -1),
        Direction.NORTHEAST: (1, -1),
        Direction.EAST: (1, 0),
        Direction.SOUTHEAST: (1, 1),
        Direction.SOUTH: (0, 1),
        Direction.SOUTHWEST: (-1, 1),
        Direction.WEST: (-1, 0),
        Direction.NORTHWEST: (-1, -1),
    }
    return mapping[direction]


def rotate_left(direction: Direction) -> Direction:
    dx, dy = direction_to_vector(direction)
    return vector_to_direction(-dy, dx)


def vector_to_direction(dx: int, dy: int) -> Direction:
    mapping = {
        (0, -1): Direction.NORTH,
        (1, -1): Direction.NORTHEAST,
        (1, 0): Direction.EAST,
        (1, 1): Direction.SOUTHEAST,
        (0, 1): Direction.SOUTH,
        (-1, 1): Direction.SOUTHWEST,
        (-1, 0): Direction.WEST,
        (-1, -1): Direction.NORTHWEST,
    }
    return mapping[(dx, dy)]
