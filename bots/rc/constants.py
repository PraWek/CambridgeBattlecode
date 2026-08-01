from cambc import Direction, EntityType


DIRECTIONS = [d for d in Direction if d != Direction.CENTRE]
CARDINALS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]

ASSIGNMENT_DIRECTIONS = [
    Direction.NORTH,
    Direction.SOUTH,
    Direction.WEST,
    Direction.EAST,
    Direction.NORTHEAST,
    Direction.SOUTHEAST,
    Direction.SOUTHWEST,
    Direction.NORTHWEST,
]

PASSABLE_BUILDINGS = {
    EntityType.CORE,
    EntityType.ROAD,
    EntityType.CONVEYOR,
    EntityType.ARMOURED_CONVEYOR,
    EntityType.BRIDGE,
    EntityType.SPLITTER,
}

MARKER_KIND_DIRECTION = 1
MARKER_KIND_BUILD_STATE = 2
MARKER_KIND_MASK = 1_000_000

BUILD_STATE_SCOUT = 0
BUILD_STATE_CONNECT = 1
