from cambc import Direction, EntityType, Environment


DIRECTIONS = [direction for direction in Direction if direction != Direction.CENTRE]
ORTHOGONAL_DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
BUILDER_WORK_DIRECTIONS = tuple(ORTHOGONAL_DIRECTIONS)

PASSABLE_BUILDINGS = {
    EntityType.CORE,
    EntityType.ROAD,
    EntityType.CONVEYOR,
    EntityType.ARMOURED_CONVEYOR,
    EntityType.BRIDGE,
    EntityType.SPLITTER,
}

ORE_TYPES = {Environment.ORE_TITANIUM, Environment.ORE_AXIONITE}

RESOURCE_TITANIUM = "titanium"
RESOURCE_AXIONITE = "axionite"

MARKER_KIND_ENEMY = 1
MARKER_KIND_PHASE = 2
MARKER_KIND_ORE_TI = 3
MARKER_KIND_ORE_AX = 4
# A sector order is a verified ore report from the core.  The payload contains
# the cardinal direction of the builder that owns the order.
MARKER_KIND_SECTOR_ORE_TI = 5
MARKER_KIND_SECTOR_ORE_AX = 6
# A one-turn handoff board lets a builder receive its sector even when the
# preferred spawn tile is unavailable (for example, a core at the map edge).
MARKER_KIND_SPAWN_DIRECTION = 7
MARKER_KIND_SPAWN_ORE_TI = 8
MARKER_KIND_SPAWN_ORE_AX = 9

PHASE_BOOTSTRAP = 1
PHASE_EXPAND_TITANIUM = 2
PHASE_EXPAND_AXIONITE = 3
PHASE_STABILIZE = 4

TITANIUM_LINE_READY_HARVESTERS = 2
TITANIUM_LINE_READY_SCALE = 118.0
MAX_BUILDERS_PHASE_ONE = 4
AXIONITE_TITANIUM_THRESHOLD = 1_000

BUILDER_DIRECTION_CODES = {
    Direction.NORTH: 1,
    Direction.EAST: 2,
    Direction.SOUTH: 3,
    Direction.WEST: 4,
}
BUILDER_CODE_DIRECTIONS = {
    code: direction for direction, code in BUILDER_DIRECTION_CODES.items()
}

MARKER_KIND_BASE = 1_000_000
MARKER_X_BASE = 10_000
MARKER_Y_BASE = 100

LARGE_NUMBER = 10**9
STUCK_KILL_ROUNDS = 25
MAX_IDLE_ROUNDS = 80

SCOUT_ROUTE_MEMORY_TILES = 24
SCOUT_REVISIT_STEP_PENALTY = 12
SCOUT_INWARD_STEP_PENALTY = 2
SCOUT_FORWARD_PROGRESS_WEIGHT = 6
SCOUT_NEW_VISION_WEIGHT = 8
SCOUT_RETURN_TO_BASE_WEIGHT = 10
SCOUT_DISTANCE_WEIGHT = 2
SCOUT_LATERAL_DEVIATION_WEIGHT = 2
SCOUT_ORE_HINT_PROGRESS_WEIGHT = 4
SCOUT_FRONTIER_CANDIDATE_LIMIT = 12

# A lower-ID builder has right of way at a head-on collision.  The yielding
# builder avoids the occupied tile long enough to choose a genuinely different
# route instead of immediately entering the same bottleneck again.
YIELD_ROUTE_AVOID_ROUNDS = 8
