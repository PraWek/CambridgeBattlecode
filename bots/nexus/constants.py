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
MAX_ECONOMY_BUILDERS = 8
ECONOMY_EXPANSION_START_ROUND = 100
ECONOMY_EXPANSION_INTERVAL_ROUNDS = 100
# Includes expansion builders and replacements after the initial four.  The
# separate cap prevents a fully explored map from entering an endless
# self-destruct/spawn loop.
MAX_ADDITIONAL_BUILDER_SPAWNS = 12
AXIONITE_TITANIUM_THRESHOLD = 1_000
# Raw axionite is destroyed by the core.  Keep deposits reserved until the
# foundry planner can commit two typed input lanes and a refined output lane as
# one transaction; mining it into an ordinary titanium trunk only wastes Ti
# and conveyor capacity.
AXIONITE_PIPELINE_ENABLED = False

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
SCOUT_PATH_GOAL_LIMIT = 12
# A 32-node search could not route back across bear_of_doom's already explored
# road tree, so scouts fell through to a local two-cell patrol while reachable
# frontiers still existed.  Ninety-six remains bounded and is only used when
# no adjacent move reveals new terrain.
SCOUT_PATH_MAX_EXPANSIONS = 96
NETWORK_PATROL_MAX_EXPANSIONS = 256
SCOUT_SECTOR_BONUS = 24
SCOUT_REPLAN_STUCK_ROUNDS = 3
SCOUT_ESCAPE_FAILURE_KILL_ROUNDS = 64
# This is based on confirmed end-of-turn positions, not accepted move
# commands.  It catches permanent builder collisions even when they still
# hold a mining or exploration target.
SCOUT_CONFIRMED_STALL_KILL_ROUNDS = 96
SCOUT_PERSISTENT_REVISIT_PENALTY = 4
SCOUT_DEAD_END_PENALTY = 48
SCOUT_DEAD_END_AVOID_ROUNDS = 40
SCOUT_CYCLE_ESCAPE_ROUNDS = 8
# Moving around an already known loop is not exploration progress.  Keep a
# builder alive long enough to backtrack across the largest map, but recycle
# it once it has failed to reveal a single new tile for a sustained period.
SCOUT_NO_DISCOVERY_KILL_ROUNDS = 240
SCOUT_PATROL_START_NO_DISCOVERY_ROUNDS = 64
# Once the local economy has had time to establish, idle scouts stop treating
# their original cardinal sector as a hard preference and push toward the
# inferred opposing Core.  On physically divided maps they still reject the
# separating wall/ore barrier as unreachable.
SCOUT_INVASION_START_ROUND = 600
SCOUT_KILL_STUCK_ROUNDS = 12
SCOUT_KILL_ROUTE_FAILURES = 24
SCOUT_KILL_CYCLES = 12
ORE_SURVEY_NEW_TILES_REQUIRED = 12
CONNECTION_DEFER_NEW_TILES = 8
CONNECTION_DEFER_MAX_ROUNDS = 32
CONNECTION_STALL_ROUNDS = 48
ORE_TARGET_STALL_ROUNDS = 32
IDLE_TARGET_RETRY_ROUNDS = 4

# A failed route must leave enough CPU time for the builder to fall back to
# exploration.  A nearby valid route normally finishes well below this bound;
# a longer or currently disconnected target is deferred until more terrain is
# known instead of consuming the whole turn.
ORE_PATH_A_STAR_MAX_EXPANSIONS = 64
CONNECTION_A_STAR_MAX_EXPANSIONS = 96

# One transport tile forwards one stack per round while a harvester produces
# one stack every four rounds.  A fifth source on the same downstream lane
# therefore adds queues but no throughput.
MAX_HARVESTERS_PER_LINE = 4
# A stale or directionally incompatible residual anchor must not make the
# whole ore unreachable.  Try a few next-cheapest sink lanes, while retaining
# a hard bound for the per-unit 2 ms budget.
MAX_FLOW_PLAN_ALTERNATIVES = 4
# Independently acting builders may choose from the same one-turn-old network
# snapshot.  Recovered mines therefore leave one slot of headroom.
RECOVERED_HARVESTER_LINE_LIMIT = 3
# Four harvesters produce exactly one stack per round, matching one conveyor's
# forwarding rate.  A new line excludes all existing transport tiles from its
# direct route, so it cannot silently merge a fifth source into an old trunk.
PLANNED_HARVESTERS_PER_LINE = 4
TRANSPORT_BUSY_OBSERVATION_TURNS = 4
# Resource-state controller calls are relatively expensive.  Structural load
# is authoritative for ordinary lanes; sample only one nearby transport per
# turn to detect a genuinely full merge without recreating the TLE cascade.
TRANSPORT_BUSY_SAMPLE_LIMIT = 1
STEINER_MAX_EXPANSIONS = 192

# A bridge is a costly fallback for a wall, hostile building, or saturated
# lane.  Open ground still uses conveyors because the search charges the live
# scaled construction costs returned by the controller.
BRIDGE_ROUTE_MAX_EXPANSIONS = 192
BRIDGE_MAX_JUMP_DISTANCE = 3

# Occupy the unused cardinal neighbours of a new titanium harvester with
# inward-facing conveyors.  They reject output from the harvester while
# preventing an opponent from placing a stealing conveyor or turret there.
HARVESTER_GUARD_LATEST_ROUND = 1500

# A lower-ID builder has right of way at a head-on collision.  The yielding
# builder avoids the occupied tile long enough to choose a genuinely different
# route instead of immediately entering the same bottleneck again.
YIELD_ROUTE_AVOID_ROUNDS = 8
