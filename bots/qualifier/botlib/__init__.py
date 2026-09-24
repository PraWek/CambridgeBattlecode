"""
Bot library that provides the following features:
- map vision
  - estimate of environment based on map symmetry
  - measure resource flow of conveyors, bridges etc...
  - map of sinks (tiles that conveyors, bridges etc... point towards)
  - map of hazards for various types (gunner, sentinel and launcher attack ranges)
- incremental pathing
  - preference over tiles already occupied with road/conveyor/bridge etc...
  - avoid hazard tiles

NOTE(randomuserhi): Code is optimized to generate efficient byte code for CPython, not for maintainability
"""

from __future__ import annotations

import sys
from typing import (
    Iterator,
    Callable,
    Optional,
    Any,
    Generic,
    TypeVar,
    Generator,
    Type,
)

from math import sqrt

from cambc import (
    Controller,
    Direction,
    EntityType,
    Environment,
    Position,
    ResourceType,
    Team,
)

from .constants import (
    POSITION_CACHE,
    DIRECTION_CACHE,
    ENVIRONMENT_CACHE,
    ENTITY_TYPE_CACHE,
    HASHMAP_DIRECTION,
    HASHMAP_ENTITY_TYPE,
    HASHMAP_RES,
    HASHMAP_ENV,
    DIRECTION_DELTAS,
    SPLITTER_DIRECTIONS,
    CARDINAL_DIRECTION_DELTAS,
    CONVEYOR_DIRECTIONS,
    BITBOARD_EDGES,
    ENTITY_COST_MAP,
    MAX_HPS,
)


class State:
    __slots__ = "state_class", "args", "kwargs"

    def __init__(self, state_class: Any, *args, **kwargs):
        self.state_class = state_class
        self.args = args
        self.kwargs = kwargs


def bind_bail(bail_func, *args, **kwargs):
    def bail(ct: Controller, other):
        return bail_func(ct, *args, **kwargs)

    return bail


class StateManager:
    __slots__ = (
        "tick",
        "current",
        "current_state_name",
        "_state_map",
        "_run",
        "_exit",
        "pathing_compute_limit",
        "_visited",
    )

    def __init__(self):
        # Current tick
        self.tick = 0

        # Current state id
        self.current = -1
        self.current_state_name = None

        # How much compute is left for pathing after map update
        self.pathing_compute_limit = 0

        self._state_map: list[Any] = []
        self._run: Callable[[Controller], None] = None
        self._exit: Callable[[Controller], None] = None

        self._visited = []

    def switch_to(
        self,
        ct: Controller,
        state_class: Any,
        *args,
        **kwargs,
    ):
        """Switch state"""
        if self.current != -1:
            print(f"{self.current_state_name}.exit()")  # TODO
            self._exit(ct)

        if hasattr(state_class, "STATE_ID"):
            state_id = state_class.STATE_ID
            args = (state_class,)
            kwargs = {}
        else:
            if isinstance(state_class, State):
                args = args + state_class.args
                kwargs = {**kwargs, **state_class.kwargs}
                state_class = state_class.state_class

            state_id = state_class.state.STATE_ID

        if state_id == -1:
            raise Exception(f"Cannot switch to the unregistered state: {state_class}")

        self._run = (state_map := self._state_map[state_id]).run
        self._exit = state_map.exit
        self.current_state_name = state_map.__name__
        self.current = state_id

        print(f"{self.current_state_name}.enter()")  # TODO
        state_map.enter(ct, *args, **kwargs)

        if (
            ct.get_cpu_time_elapsed() < 1000
            and self._visited[state_id] != self.tick
            and (ct.get_move_cooldown() == 0 and ct.get_action_cooldown() == 0)
        ):
            self._visited[state_id] = self.tick
            self._run(ct)

    def run(self, ct: Controller):
        """Immediately executes run of the given state"""
        print(f"{self.current_state_name}.run()")  # TODO
        self._visited[self.current] = self.tick
        self._run(ct)

    def register(self, state_class):
        id = len(self._state_map)
        self._state_map.append(state_class)
        self._visited.append(0)
        state_class.state.STATE_ID = id


def rotate_180(x: int, y: int) -> int:
    """Rotates a position by 180 degrees"""

    x = int(-(x - MAP_TRUE_CENTER_X) + MAP_TRUE_CENTER_X)
    y = int(-(y - MAP_TRUE_CENTER_Y) + MAP_TRUE_CENTER_Y)
    return y * MAP_INFO.width + x


def reflect_horizontal(x: int, y: int) -> int:
    """Reflects a position horizontally along the map center"""

    x = int(-(x - MAP_TRUE_CENTER_X) + MAP_TRUE_CENTER_X)
    return y * MAP_INFO.width + x


def reflect_vertical(x: int, y: int) -> int:
    """Reflects a position vertically along the map center"""

    y = int(-(y - MAP_TRUE_CENTER_Y) + MAP_TRUE_CENTER_Y)
    return y * MAP_INFO.width + x


# Constants - Internal, should not be used outside of this file
MAX_MAP_WIDTH = 50
MAX_MAP_HEIGHT = 50
MAX_MAP_SIZE = MAX_MAP_WIDTH * MAX_MAP_HEIGHT

# Controller methods - Internal, should not be used outside of this file
CT_GET_CPU_TIME_ELAPSED: Callable[[], int] = None
CT_GET_ENTITY_TYPE: Callable[[Optional[int]], EntityType] = None
CT_GET_POSITION: Callable[[Optional[int]], Position] = None
CT_IS_IN_VISION: Callable[[Position], bool] = None
CT_GET_TILE_ENV: Callable[[Position], Environment] = None
CT_GET_TEAM: Callable[[int | None], Team] = None
CT_GET_TILE_BUILDING_ID: Callable[[Position], None | int] = None
CT_GET_TILE_BUILDER_BOT_ID: Callable[[Position], None | int] = None
CT_GET_HP: Callable[[int | None], int] = None
CT_GET_DIRECTION: Callable[[int | None], Direction] = None
CT_GET_BRIDGE_TARGET: Callable[[int], Position] = None
CT_GET_CURRENT_ROUND: Callable[[], int] = None
CT_GET_STORED_RESOURCE: Callable[[int | None], ResourceType | None] = None
CT_GET_STORED_RESOURCE_ID: Callable[[int | None], int | None] = None
CT_GET_ATTACKABLE_TILES_FROM: Callable[[Position, Direction, EntityType], list[Position]] = None
CT_DRAW_INDICATOR_LINE: Callable[[Position, Position, int, int, int], None] = None
CT_DRAW_INDICATOR_DOT: Callable[[Position, int, int, int], None] = None
CT_GET_GLOBAL_RESOURCES: Callable[[], tuple[int, int]] = None
CT_GET_MAX_HP: Callable[[int | None], int] = None

# Cache of all positions possible based on max map width and height
POSITION_GLOBAL_CACHE: list[Position] = [
    Position(i % MAX_MAP_WIDTH, i // MAX_MAP_WIDTH) for i in range(MAX_MAP_SIZE)
]

VISION_RADIUS: int = 0
VISION_WIDTH: int = 0
VISION_ITER: list[bool] = None


# Map info
class MapInfo:
    __slots__ = "width", "height", "size"

    def __init__(self):
        self.width: int = 0
        self.height: int = 0
        self.size: int = 0


MAP_INFO: MapInfo = MapInfo()

MAP_TRUE_CENTER_X: int = 0
MAP_TRUE_CENTER_Y: int = 0


class IncomeInfo:
    __slots__ = (
        "ti_delta",
        "ax_delta",
        "_prev_ti",
        "_prev_ax",
    )

    def __init__(self):
        self.ti_delta = 0
        self.ax_delta = 0

        self._prev_ti = 0
        self._prev_ax = 0


INCOME_INFO = IncomeInfo()


# Generic information about self
class UnitInfo:
    __slots__ = (
        "team",
        "team_idx",
        "id",
        "position",
        "position_idx",
        "ally_core_pos",
        "ally_core_idx",
        "vision_mask",
    )

    def __init__(self):
        self.team: Team = None
        self.team_idx: int = 0
        self.id: int = 0
        self.position: Position = None
        self.position_idx: int = 0
        self.ally_core_pos: Position = None
        self.ally_core_idx: int = 0
        self.vision_mask: int = 0

    def refresh_position(self):
        UNIT_INFO.position = (pos := CT_GET_POSITION(None))
        pos_x, pos_y = pos.x, pos.y
        UNIT_INFO.position_idx = pos_y * MAP_INFO.width + pos_x


UNIT_INFO: UnitInfo = UnitInfo()

# Environment Ground Truth
ENV: list[Environment] = [None] * (MAX_MAP_SIZE)

# Environment Potential Mirrors
MIRROR_ENV_LENGTH = 1
MIRROR_ENV: list[tuple[list[Environment], Callable[[int, int], int]]] = [
    [([None] * MAX_MAP_SIZE), rotate_180],
    [([None] * MAX_MAP_SIZE), None],
    [([None] * MAX_MAP_SIZE), None],
]
# Potential core positions
ENEMY_CORE_POS: list[int] = [0, 0, 0]

# Buildings (48 bits)
# 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000
# ^                   ^                   ^              ^
# |                   |                   |              (4 bits) EntityType
# |                   |                   (12 bits) Direction / Bridge Target Idx
# |                   (16 bits) Building ID
# (16 bits) Building HP
ALLY_BUILDINGS: list[int] = [0] * (MAX_MAP_SIZE)
ENEMY_BUILDINGS: list[int] = [0] * (MAX_MAP_SIZE)
PREV_ALLY_BUILDINGS: list[int] = [0] * (MAX_MAP_SIZE)
PREV_ENEMY_BUILDINGS: list[int] = [0] * (MAX_MAP_SIZE)

# Builder Bots (32 bit)
# 0000 0000 0000 0000 0000 0000 0000 0000
# ^                   ^
# |                   (16 bits) Builder Bot ID
# (16 bits) Builder Bot HP
ALLY_BUILDER_BOT: list[int] = [0] * (MAX_MAP_SIZE)
ENEMY_BUILDER_BOT: list[int] = [0] * (MAX_MAP_SIZE)
PREV_ALLY_BUILDER_BOT: list[int] = [0] * (MAX_MAP_SIZE)
PREV_ENEMY_BUILDER_BOT: list[int] = [0] * (MAX_MAP_SIZE)

# Bit Boards for fast pattern matching

# 0000
# ^^^^
# |||Empty
# ||Wall
# |Titanium
# Axionite
BITBOARD_ENV: list[int] = [0] * MAX_MAP_SIZE
BITBOARD_MIRROR_ENV: list[int] = [
    ([0] * MAX_MAP_SIZE),
    ([0] * MAX_MAP_SIZE),
    ([0] * MAX_MAP_SIZE),
]

# 0000 0000 0000 0000 0000 0000 0000 0000
# Refer to `HASHMAP_ENTITY_TYPE` for bit positions of each entity type
# 1st bit is unused
#
# Can be used to quickly pattern match entities on a tile:
# `(BITBOARD_XXXX[tile_idx] & 0bXXXX) != 0``
BITBOARD_ALLY: list[int] = [0] * MAX_MAP_SIZE
BITBOARD_ENEMY: list[int] = [0] * MAX_MAP_SIZE


class SinkInfo:
    __slots__ = (
        "tile_idx",
        "bridge_sources",
        "num_bridge_sources",
        "conveyor_sources",
        "num_conveyor_sources",
        "harvester_sources",
        "num_harvester_sources",
        "foundry_sources",
        "num_foundry_sources",
    )

    def __init__(self, tile_idx: int):
        self.tile_idx = tile_idx

        self.bridge_sources: set[int] = set()
        self.num_bridge_sources: int = 0

        # NOTE includes splitters
        self.conveyor_sources: set[int] = set()
        self.num_conveyor_sources: int = 0

        self.harvester_sources: set[int] = set()
        self.num_harvester_sources: int = 0

        self.foundry_sources: set[int] = set()
        self.num_foundry_sources: int = 0


SINKS: list[SinkInfo] = [SinkInfo(tile_idx) for tile_idx in range(MAX_MAP_SIZE)]


class ResourceFlowNode:
    __slots__ = "_flow_value", "_stall_value", "_seen"

    def __init__(self):
        self._flow_value: float = 0
        self._stall_value: int = 0
        self._seen: int = 0

    def get_flow(self):
        return self._flow_value / 4

    def get_stall(self):
        return self._stall_value / 4


# Internal buffer used to count occurences of resources on a tile
# Is reset once sampled
RESOURCE_FLOW_SAMPLE: list[list[ResourceFlowNode]] = [
    ([ResourceFlowNode() for _ in range(MAX_MAP_SIZE)]),  # TITANIUM
    ([ResourceFlowNode() for _ in range(MAX_MAP_SIZE)]),  # AXIONITE
    ([ResourceFlowNode() for _ in range(MAX_MAP_SIZE)]),  # REFINED
]
# Tracks the resource flow rate of each tile
# a non-zero value means theres flow
# Inaccurate for harvester / foundary outputs
RESOURCE_FLOW: list[list[ResourceFlowNode]] = [
    ([ResourceFlowNode() for _ in range(MAX_MAP_SIZE)]),  # TITANIUM
    ([ResourceFlowNode() for _ in range(MAX_MAP_SIZE)]),  # AXIONITE
    ([ResourceFlowNode() for _ in range(MAX_MAP_SIZE)]),  # REFINED
]
RESOURCE_ID: list[int | None] = [None] * MAX_MAP_SIZE
# 0000
#  ^^^
#  ||titanium
#  |axionite
#  refined
RESOURCE_TYPE: list[int] = [0] * MAX_MAP_SIZE

# Tracks hazards presented by each entity type
# Hazards[0] is unused (represents "None" entity)
HAZARDS: list[list[int]] = [([0] * MAX_MAP_SIZE) for _ in range(16)]

BITBOARD_HAZARDS: list[int] = [0] * 16


BITBOARD_TARGETS = [1 << tile_idx for tile_idx in range(MAX_MAP_SIZE)]
NEG_BITBOARD_TARGETS = [~target for target in BITBOARD_TARGETS]


class BitboardGrid:
    __slots__ = (
        "env",
        "env_mirrors",
        "ally",
        "enemy",
        "flow",
        "stall",
        "prev_ally_bots",
        "prev_enemy_bots",
    )

    def __init__(self):
        self.env: list[int] = [0] * len(ENVIRONMENT_CACHE)
        self.env_mirrors: list[list[int]] = [
            [0] * len(ENVIRONMENT_CACHE),
            [0] * len(ENVIRONMENT_CACHE),
            [0] * len(ENVIRONMENT_CACHE),
        ]
        self.ally: list[int] = [0] * len(ENTITY_TYPE_CACHE)
        self.enemy: list[int] = [0] * len(ENTITY_TYPE_CACHE)
        self.prev_ally_bots: int = 0
        self.prev_enemy_bots: int = 0
        self.flow: list[int] = [
            [0] * 5,
            [0] * 5,
            [0] * 5,
        ]
        self.stall: list[int] = [
            [0] * 5,
            [0] * 5,
            [0] * 5,
        ]


BITBOARD_GRID = BitboardGrid()

# ------------- MAIN LOGIC -------------


def botlib_init(ct: Controller, get_core_pos=True):
    """Initialize map state, must be called once upon unit spawning."""

    # Capture frequently used controller methods
    global CT_GET_MAX_HP, CT_GET_GLOBAL_RESOURCES, CT_DRAW_INDICATOR_DOT, CT_DRAW_INDICATOR_LINE, CT_GET_ATTACKABLE_TILES_FROM, CT_GET_STORED_RESOURCE_ID, CT_GET_STORED_RESOURCE, CT_GET_CURRENT_ROUND, CT_GET_BRIDGE_TARGET, CT_GET_DIRECTION, CT_GET_HP, CT_GET_TEAM, CT_GET_POSITION, CT_GET_ENTITY_TYPE, CT_IS_IN_VISION, CT_GET_TILE_ENV, CT_GET_CPU_TIME_ELAPSED, CT_GET_TILE_BUILDING_ID, CT_GET_TILE_BUILDER_BOT_ID
    CT_GET_CPU_TIME_ELAPSED = ct.get_cpu_time_elapsed
    CT_GET_HP = ct.get_hp
    CT_GET_POSITION = ct.get_position
    CT_GET_ENTITY_TYPE = ct.get_entity_type
    CT_IS_IN_VISION = ct.is_in_vision
    CT_GET_TILE_ENV = ct.get_tile_env
    CT_GET_TEAM = ct.get_team
    CT_GET_TILE_BUILDING_ID = ct.get_tile_building_id
    CT_GET_TILE_BUILDER_BOT_ID = ct.get_tile_builder_bot_id
    CT_GET_DIRECTION = ct.get_direction
    CT_GET_BRIDGE_TARGET = ct.get_bridge_target
    CT_GET_CURRENT_ROUND = ct.get_current_round
    CT_GET_STORED_RESOURCE = ct.get_stored_resource
    CT_GET_STORED_RESOURCE_ID = ct.get_stored_resource_id
    CT_GET_ATTACKABLE_TILES_FROM = ct.get_attackable_tiles_from
    CT_DRAW_INDICATOR_DOT = ct.draw_indicator_dot
    CT_DRAW_INDICATOR_LINE = ct.draw_indicator_line
    CT_GET_GLOBAL_RESOURCES = ct.get_global_resources
    CT_GET_MAX_HP = ct.get_max_hp

    # Team and ID
    UNIT_INFO.id = ct.get_id()
    UNIT_INFO.team = CT_GET_TEAM()
    UNIT_INFO.team_idx = (team_idx := (0 if UNIT_INFO.team == Team.A else 1))

    # Map properties
    MAP_INFO.width = (map_width := ct.get_map_width())
    MAP_INFO.height = (map_height := ct.get_map_height())
    MAP_INFO.size = (map_size := (map_width * map_height))

    # Map center used for figuring out symmetries
    global MAP_TRUE_CENTER_X, MAP_TRUE_CENTER_Y
    MAP_TRUE_CENTER_X = int(map_width / 2)
    if map_width % 2 == 0:
        MAP_TRUE_CENTER_X -= 0.5
    MAP_TRUE_CENTER_Y = int(map_height / 2)
    if map_height % 2 == 0:
        MAP_TRUE_CENTER_Y -= 0.5

    # Generate a cached list of Position objects for this map
    POSITION_CACHE.extend(
        [POSITION_GLOBAL_CACHE[(i // map_width) * 50 + (i % map_width)] for i in range(map_size)]
    )

    # Setup candidate core positions and mirror environments
    global MIRROR_ENV_LENGTH
    if get_core_pos:
        # import sys

        # print(f"ally bot is at: {CT_GET_POSITION()} and is team: {CT_GET_TEAM()}", file=sys.stderr)
        ally_core_x, ally_core_y = CT_GET_POSITION(team_idx + 1)
    else:
        ally_core_x, ally_core_y = 0, 0
    UNIT_INFO.ally_core_idx = (ally_core_idx := (ally_core_y * map_width + ally_core_x))
    UNIT_INFO.ally_core_pos = POSITION_CACHE[ally_core_idx]

    ENEMY_CORE_POS[0] = rotate_180(ally_core_x, ally_core_y)

    horizontal_candidate = reflect_horizontal(ally_core_x, ally_core_y)
    if horizontal_candidate != ally_core_idx:
        MIRROR_ENV[MIRROR_ENV_LENGTH][1] = reflect_horizontal
        ENEMY_CORE_POS[MIRROR_ENV_LENGTH] = horizontal_candidate
        MIRROR_ENV_LENGTH += 1

    vertical_candidate = reflect_vertical(ally_core_x, ally_core_y)
    if vertical_candidate != ally_core_idx:
        MIRROR_ENV[MIRROR_ENV_LENGTH][1] = reflect_vertical
        ENEMY_CORE_POS[MIRROR_ENV_LENGTH] = vertical_candidate
        MIRROR_ENV_LENGTH += 1

    # Setup vision properties
    global VISION_RADIUS, VISION_WIDTH, VISION_WIDTH_RANGE, VISION_ITER
    VISION_RADIUS = int(sqrt(ct.get_vision_radius_sq()))
    VISION_WIDTH = VISION_RADIUS * 2 + 1

    # bind costmap
    ENTITY_COST_MAP[None] = None
    ENTITY_COST_MAP[EntityType.BUILDER_BOT] = ct.get_builder_bot_cost
    ENTITY_COST_MAP[EntityType.CORE] = None
    ENTITY_COST_MAP[EntityType.GUNNER] = ct.get_gunner_cost
    ENTITY_COST_MAP[EntityType.SENTINEL] = ct.get_sentinel_cost
    ENTITY_COST_MAP[EntityType.BREACH] = ct.get_breach_cost
    ENTITY_COST_MAP[EntityType.LAUNCHER] = ct.get_launcher_cost
    ENTITY_COST_MAP[EntityType.CONVEYOR] = ct.get_conveyor_cost
    ENTITY_COST_MAP[EntityType.SPLITTER] = ct.get_splitter_cost
    ENTITY_COST_MAP[EntityType.ARMOURED_CONVEYOR] = ct.get_armoured_conveyor_cost
    ENTITY_COST_MAP[EntityType.BRIDGE] = ct.get_bridge_cost
    ENTITY_COST_MAP[EntityType.HARVESTER] = ct.get_harvester_cost
    ENTITY_COST_MAP[EntityType.FOUNDRY] = ct.get_foundry_cost
    ENTITY_COST_MAP[EntityType.ROAD] = ct.get_road_cost
    ENTITY_COST_MAP[EntityType.BARRIER] = ct.get_barrier_cost
    ENTITY_COST_MAP[EntityType.MARKER] = None


def botlib_update(state: StateManager | None) -> None | list[int]:
    """Updates state, run once at the start of each turn"""

    # ------------- Bindings -------------
    ct_get_team = CT_GET_TEAM
    ct_is_in_vision = CT_IS_IN_VISION
    ct_get_tile_env = CT_GET_TILE_ENV
    ct_get_tile_builder_bot_id = CT_GET_TILE_BUILDER_BOT_ID
    ct_get_tile_building_id = CT_GET_TILE_BUILDING_ID
    ct_get_entity_type = CT_GET_ENTITY_TYPE
    ct_get_hp = CT_GET_HP
    ct_get_direction = CT_GET_DIRECTION
    ct_get_bridge_target = CT_GET_BRIDGE_TARGET
    ct_get_stored_resource = CT_GET_STORED_RESOURCE
    ct_get_stored_resource_id = CT_GET_STORED_RESOURCE_ID
    ct_get_max_hp = CT_GET_MAX_HP

    team = UNIT_INFO.team
    self_id = UNIT_INFO.id

    position_cache = POSITION_CACHE
    direction_cache = DIRECTION_CACHE

    vision_width = VISION_WIDTH
    map_width = MAP_INFO.width
    map_height = MAP_INFO.height
    map_size = MAP_INFO.size

    env = ENV

    global MIRROR_ENV_LENGTH
    mirror_env = MIRROR_ENV
    enemy_core_pos = ENEMY_CORE_POS

    hashmap_env = HASHMAP_ENV
    hashmap_entity_type = HASHMAP_ENTITY_TYPE
    hashmap_direction = HASHMAP_DIRECTION
    hashmap_res = HASHMAP_RES

    bitboard_env = BITBOARD_ENV
    bitboard_mirror_env = BITBOARD_MIRROR_ENV

    bitboard_ally = BITBOARD_ALLY
    bitboard_enemy = BITBOARD_ENEMY

    bitboard_grid = BITBOARD_GRID
    bitboard_grid_env = bitboard_grid.env
    bitboard_grid_env_mirrors = bitboard_grid.env_mirrors
    bitboard_grid_ally = bitboard_grid.ally
    bitboard_grid_enemy = bitboard_grid.enemy
    bitboard_grid_flow = bitboard_grid.flow
    bitboard_grid_stall = bitboard_grid.stall

    ally_buildings = ALLY_BUILDINGS
    prev_ally_buildings = PREV_ALLY_BUILDINGS
    ally_builder_bot = ALLY_BUILDER_BOT
    prev_ally_builder_bot = PREV_ALLY_BUILDER_BOT
    enemy_buildings = ENEMY_BUILDINGS
    prev_enemy_buildings = PREV_ENEMY_BUILDINGS
    enemy_builder_bot = ENEMY_BUILDER_BOT
    prev_enemy_builder_bot = PREV_ENEMY_BUILDER_BOT

    resource_flow_sample = RESOURCE_FLOW_SAMPLE
    resource_flow = RESOURCE_FLOW
    resource_type = RESOURCE_TYPE
    resource_id = RESOURCE_ID

    max_hps = MAX_HPS

    # ------------- Implementation -------------

    income_info = INCOME_INFO
    ti, ax = CT_GET_GLOBAL_RESOURCES()
    delta_ti = ti - income_info._prev_ti
    delta_ax = ax - income_info._prev_ax
    income_info._prev_ti = ti
    income_info._prev_ax = ax
    income_info.ti_delta += 0.1 * (delta_ti - income_info.ti_delta)
    income_info.ax_delta += 0.1 * (delta_ax - income_info.ax_delta)

    UNIT_INFO.vision_mask = 0

    UNIT_INFO.position = (pos := CT_GET_POSITION(None))
    pos_x, pos_y = pos.x, pos.y

    UNIT_INFO.position_idx = (pos_idx := pos_y * map_width + pos_x)

    origin_x, origin_y = pos_x - VISION_RADIUS, pos_y - VISION_RADIUS

    bitboard_grid.prev_ally_bots = bitboard_grid_ally[1] & NEG_BITBOARD_TARGETS[pos_idx]
    bitboard_grid.prev_enemy_bots = bitboard_grid_enemy[1]

    for y in range(vision_width):
        query_idx = (query_x := origin_x) + (query_y := origin_y + y) * map_width

        for _ in range(vision_width):
            if (
                query_idx < 0
                or query_idx >= map_size
                or not ct_is_in_vision(query := position_cache[query_idx])
            ):
                query_x += 1
                query_idx += 1
                continue

            bitboard_target = BITBOARD_TARGETS[query_idx]
            neg_bitboard_target = NEG_BITBOARD_TARGETS[query_idx]

            UNIT_INFO.vision_mask |= bitboard_target

            # reset resource flow count in vision
            for i in range(3):
                (flow_node := resource_flow[i][query_idx])._seen += 1
                if flow_node._seen > 3:
                    flow_sample_node = resource_flow_sample[i][query_idx]

                    flow_node._flow_value += 0.5 * (
                        flow_sample_node._flow_value - flow_node._flow_value
                    )
                    flow_sample_node._flow_value = 0

                    flow_node._stall_value += 0.5 * (
                        flow_sample_node._stall_value - flow_node._stall_value
                    )
                    flow_sample_node._stall_value = 0

                    flow_node._seen = 0

                    flow_bitboard = bitboard_grid_flow[i]
                    stall_bitboard = bitboard_grid_stall[i]

                    if flow_value := (flow_node._flow_value / 4) < 0.05:
                        flow_bitboard[0] &= neg_bitboard_target
                        flow_bitboard[1] &= neg_bitboard_target
                        flow_bitboard[2] &= neg_bitboard_target
                        flow_bitboard[3] &= neg_bitboard_target
                        flow_bitboard[4] &= neg_bitboard_target
                    elif (flow_value) < 0.2:
                        flow_bitboard[0] |= bitboard_target
                        flow_bitboard[1] &= neg_bitboard_target
                        flow_bitboard[2] &= neg_bitboard_target
                        flow_bitboard[3] &= neg_bitboard_target
                        flow_bitboard[4] &= neg_bitboard_target
                    elif flow_value < 0.4:
                        flow_bitboard[0] &= neg_bitboard_target
                        flow_bitboard[1] |= bitboard_target
                        flow_bitboard[2] &= neg_bitboard_target
                        flow_bitboard[3] &= neg_bitboard_target
                        flow_bitboard[4] &= neg_bitboard_target
                    elif flow_value < 0.6:
                        flow_bitboard[0] &= neg_bitboard_target
                        flow_bitboard[1] &= neg_bitboard_target
                        flow_bitboard[2] |= bitboard_target
                        flow_bitboard[3] &= neg_bitboard_target
                        flow_bitboard[4] &= neg_bitboard_target
                    elif flow_value < 0.8:
                        flow_bitboard[0] &= neg_bitboard_target
                        flow_bitboard[1] &= neg_bitboard_target
                        flow_bitboard[2] &= neg_bitboard_target
                        flow_bitboard[3] |= bitboard_target
                        flow_bitboard[4] &= neg_bitboard_target
                    else:
                        flow_bitboard[0] &= neg_bitboard_target
                        flow_bitboard[1] &= neg_bitboard_target
                        flow_bitboard[2] &= neg_bitboard_target
                        flow_bitboard[3] &= neg_bitboard_target
                        flow_bitboard[4] |= bitboard_target

                    if (stall_value := (flow_node._stall_value / 4)) < 0.05:
                        stall_bitboard[0] &= neg_bitboard_target
                        stall_bitboard[1] &= neg_bitboard_target
                        stall_bitboard[2] &= neg_bitboard_target
                        stall_bitboard[3] &= neg_bitboard_target
                        stall_bitboard[4] &= neg_bitboard_target
                    elif stall_value < 0.2:
                        stall_bitboard[0] |= bitboard_target
                        stall_bitboard[1] &= neg_bitboard_target
                        stall_bitboard[2] &= neg_bitboard_target
                        stall_bitboard[3] &= neg_bitboard_target
                        stall_bitboard[4] &= neg_bitboard_target
                    elif stall_value < 0.4:
                        stall_bitboard[0] &= neg_bitboard_target
                        stall_bitboard[1] |= bitboard_target
                        stall_bitboard[2] &= neg_bitboard_target
                        stall_bitboard[3] &= neg_bitboard_target
                        stall_bitboard[4] &= neg_bitboard_target
                    elif stall_value < 0.6:
                        stall_bitboard[0] &= neg_bitboard_target
                        stall_bitboard[1] &= neg_bitboard_target
                        stall_bitboard[2] |= bitboard_target
                        stall_bitboard[3] &= neg_bitboard_target
                        stall_bitboard[4] &= neg_bitboard_target
                    elif stall_value < 0.8:
                        stall_bitboard[0] &= neg_bitboard_target
                        stall_bitboard[1] &= neg_bitboard_target
                        stall_bitboard[2] &= neg_bitboard_target
                        stall_bitboard[3] |= bitboard_target
                        stall_bitboard[4] &= neg_bitboard_target
                    else:
                        stall_bitboard[0] &= neg_bitboard_target
                        stall_bitboard[1] &= neg_bitboard_target
                        stall_bitboard[2] &= neg_bitboard_target
                        stall_bitboard[3] &= neg_bitboard_target
                        stall_bitboard[4] |= bitboard_target

            # Update environment map
            env[query_idx] = (env_tile := ct_get_tile_env(query))
            bitboard_env[query_idx] = (env_bit_cell := (1 << (env_type := hashmap_env[env_tile])))
            bitboard_grid_env[env_type] |= bitboard_target

            # Update potential mirrors
            for i in range(MIRROR_ENV_LENGTH - 1, -1, -1):
                menv, func = mirror_env[i]
                menv[(mquery_idx := func(query_x, query_y))] = env_tile
                bitboard_mirror_env[i][mquery_idx] = env_bit_cell
                bitboard_grid_env_mirrors[i][env_type] |= 1 << mquery_idx

                # Check ground truth with mirror env
                if menv[query_idx] is not None and menv[query_idx] != env_tile:
                    # If it does not match, remove it
                    # (we use a fast swap + backwards iteration to do this op quickly)
                    MIRROR_ENV_LENGTH -= 1
                    if i != MIRROR_ENV_LENGTH:
                        mirror_env[i], mirror_env[MIRROR_ENV_LENGTH] = (
                            mirror_env[MIRROR_ENV_LENGTH],
                            mirror_env[i],
                        )
                        (
                            bitboard_mirror_env[i],
                            bitboard_mirror_env[MIRROR_ENV_LENGTH],
                        ) = (
                            bitboard_mirror_env[MIRROR_ENV_LENGTH],
                            bitboard_mirror_env[i],
                        )
                        enemy_core_pos[i], enemy_core_pos[MIRROR_ENV_LENGTH] = (
                            enemy_core_pos[MIRROR_ENV_LENGTH],
                            enemy_core_pos[i],
                        )
                        (
                            bitboard_grid_env_mirrors[i],
                            bitboard_grid_env_mirrors[MIRROR_ENV_LENGTH],
                        ) = (
                            bitboard_grid_env_mirrors[MIRROR_ENV_LENGTH],
                            bitboard_grid_env_mirrors[i],
                        )

            # Update building info
            if (building_id := ct_get_tile_building_id(query)) is not None:
                flag = 1 << (entity_type := hashmap_entity_type[ct_get_entity_type(building_id)])
                direction = 0

                if entity_type == 10:
                    # Bridge
                    bx, by = ct_get_bridge_target(building_id)
                    direction = by * map_width + bx

                    # Get resource
                    resource_type[query_idx] = (
                        0
                        if (res := ct_get_stored_resource(building_id)) is None
                        else (1 << hashmap_res[res])
                    )

                    if res is not None:
                        res_idx = hashmap_res[res]
                        flow = resource_flow[res_idx]
                        flow_node = flow[query_idx]
                        flow_sample = resource_flow_sample[res_idx]
                        flow_sample_node = flow_sample[query_idx]

                        if (res_id := ct_get_stored_resource_id(building_id)) != resource_id[
                            query_idx
                        ]:
                            flow_node._flow_value = max(flow_node._flow_value, 1)
                            flow_sample_node._flow_value += 1
                            resource_id[query_idx] = res_id
                        else:
                            flow_node._stall_value = max(flow_node._stall_value, 1)
                            flow_sample_node._stall_value += 1
                    else:
                        resource_id[query_idx] = None

                elif entity_type == 8:
                    # Splitter
                    direction = hashmap_direction[ct_get_direction(building_id)]

                    # Get resource
                    resource_type[query_idx] = (
                        0
                        if (res := ct_get_stored_resource(building_id)) is None
                        else (1 << hashmap_res[res])
                    )

                    if res is not None:
                        res_idx = hashmap_res[res]
                        flow = resource_flow[res_idx]
                        flow_node = flow[query_idx]
                        flow_sample = resource_flow_sample[res_idx]
                        flow_sample_node = flow_sample[query_idx]

                        if (res_id := ct_get_stored_resource_id(building_id)) != resource_id[
                            query_idx
                        ]:
                            flow_node._flow_value = max(flow_node._flow_value, 1)
                            flow_sample_node._flow_value += 1
                            resource_id[query_idx] = res_id
                        else:
                            flow_node._stall_value = max(flow_node._stall_value, 1)
                            flow_sample_node._stall_value += 1
                    else:
                        resource_id[query_idx] = None

                elif (flag & 0b0011_1000) != 0:
                    # gunner (3), sentinel (4), breach (5)
                    direction = hashmap_direction[ct_get_direction(building_id)]
                    resource_id[query_idx] = None
                    resource_type[query_idx] = 0
                elif (flag & 0b0010_1000_0000) != 0:
                    # conveyor (7), armoured_conveyor (9)
                    direction = hashmap_direction[ct_get_direction(building_id)]

                    # Get resource
                    resource_type[query_idx] = (
                        0
                        if (res := ct_get_stored_resource(building_id)) is None
                        else (1 << hashmap_res[res])
                    )

                    if res is not None:
                        res_idx = hashmap_res[res]
                        flow = resource_flow[res_idx]
                        flow_node = flow[query_idx]
                        flow_sample = resource_flow_sample[res_idx]
                        flow_sample_node = flow_sample[query_idx]

                        if (res_id := ct_get_stored_resource_id(building_id)) != resource_id[
                            query_idx
                        ]:
                            flow_node._flow_value = max(flow_node._flow_value, 1)
                            flow_sample_node._flow_value += 1
                            resource_id[query_idx] = res_id
                        else:
                            flow_node._stall_value = max(flow_node._stall_value, 1)
                            flow_sample_node._stall_value += 1
                    else:
                        resource_id[query_idx] = None
                else:
                    resource_id[query_idx] = None
                    resource_type[query_idx] = 0

                max_hps[entity_type] = ct_get_max_hp(building_id)

                token = (
                    (entity_type & 0xF)
                    | ((direction & 0xFFF) << 4)
                    | ((building_id & 0xFFFF) << 16)
                    | ((ct_get_hp(building_id) & 0xFFFF) << 32)
                )
                if ct_get_team(building_id) == team:
                    bitboard_ally[query_idx] = flag
                    bitboard_enemy[query_idx] = 0

                    ally_buildings[query_idx] = token
                    enemy_buildings[query_idx] = 0
                else:
                    bitboard_enemy[query_idx] = flag
                    bitboard_ally[query_idx] = 0

                    enemy_buildings[query_idx] = token
                    ally_buildings[query_idx] = 0
            else:
                bitboard_ally[query_idx] = 0
                bitboard_enemy[query_idx] = 0

                ally_buildings[query_idx] = 0
                enemy_buildings[query_idx] = 0

                resource_id[query_idx] = None
                resource_type[query_idx] = 0

            # Update bot info
            if (bot_id := ct_get_tile_builder_bot_id(query)) is not None:
                max_hps[1] = ct_get_max_hp(bot_id)
                token = (bot_id & 0xFFFF) | ((ct_get_hp(bot_id) & 0xFFFF) << 16)
                if ct_get_team(bot_id) == team:
                    if bot_id != self_id:
                        bitboard_ally[query_idx] |= 2

                        bitboard_grid_ally[1] |= bitboard_target
                        bitboard_grid_enemy[1] &= neg_bitboard_target
                    else:
                        bitboard_grid_ally[1] &= neg_bitboard_target
                        bitboard_grid_enemy[1] &= neg_bitboard_target

                    ally_builder_bot[query_idx] = token
                    enemy_builder_bot[query_idx] = 0
                else:
                    bitboard_enemy[query_idx] |= 2

                    bitboard_grid_enemy[1] |= bitboard_target
                    bitboard_grid_ally[1] &= neg_bitboard_target

                    enemy_builder_bot[query_idx] = token
                    ally_builder_bot[query_idx] = 0
            else:
                bitboard_grid_ally[1] &= neg_bitboard_target
                bitboard_grid_enemy[1] &= neg_bitboard_target

                ally_builder_bot[query_idx] = 0
                enemy_builder_bot[query_idx] = 0

            ally_building = ally_buildings[query_idx] & 0xFFFF
            prev_ally_building = prev_ally_buildings[query_idx] & 0xFFFF
            enemy_building = enemy_buildings[query_idx] & 0xFFFF
            prev_enemy_building = prev_enemy_buildings[query_idx] & 0xFFFF
            ally_building_type = ally_building & 0xF
            prev_ally_building_type = prev_ally_building & 0xF
            enemy_building_type = enemy_building & 0xF
            prev_enemy_building_type = prev_enemy_building & 0xF

            # Update changed building bitboards
            if (
                ally_building_type != prev_ally_building_type
                or enemy_building_type != prev_enemy_building_type
            ):
                for i in range(2, 16):
                    if i == ally_building_type:
                        bitboard_grid_ally[i] |= bitboard_target
                    else:
                        bitboard_grid_ally[i] &= neg_bitboard_target

                    if i == enemy_building_type:
                        bitboard_grid_enemy[i] |= bitboard_target
                    else:
                        bitboard_grid_enemy[i] &= neg_bitboard_target

            # Update sinks and hazards
            # remove old sinks and hazards first
            if diff_ally_building := ally_building != prev_ally_building:
                update_sink(query_idx, query_x, query_y, prev_ally_building, -1)  # remove old sink
            if diff_enemy_building := enemy_building != prev_enemy_building:
                update_sink(query_idx, query_x, query_y, prev_enemy_building, -1)  # remove old sink
                update_hazards(
                    query_x, query_y, query_idx, prev_enemy_building, -1
                )  # Remove old hazard
            # add new sinks and hazards
            if diff_ally_building:
                update_sink(query_idx, query_x, query_y, ally_building, 1)  # add new sink
            if diff_enemy_building:
                update_sink(query_idx, query_x, query_y, enemy_building, 1)  # add new sink
                update_hazards(query_x, query_y, query_idx, enemy_building, 1)  # Add new hazard

            # Update previous values
            prev_ally_builder_bot[query_idx] = ally_builder_bot[query_idx]
            prev_ally_buildings[query_idx] = ally_buildings[query_idx]
            prev_enemy_builder_bot[query_idx] = enemy_builder_bot[query_idx]
            prev_enemy_buildings[query_idx] = enemy_buildings[query_idx]

            query_x += 1
            query_idx += 1

    if state != None:
        state.tick += 1


def update_hazards(query_x: int, query_y: int, query_idx: int, building: int, delta: int):
    """
    Internal method, do not use

    Updates hazard structures for detecting dangerous tilies covered by specific buildings such
    as launchers, sentinels, gunners
    """

    ct_get_attackable_tiles_from = CT_GET_ATTACKABLE_TILES_FROM
    map_width = MAP_INFO.width
    map_height = MAP_INFO.height
    map_size = MAP_INFO.size
    hazards = HAZARDS
    bitboard_hazards = BITBOARD_HAZARDS

    building_type = building & 0xF
    if building_type == 3 or building_type == 4 or building_type == 5:  # Gunner, Sentinel, Breach
        direction = (building & 0xFFF0) >> 4
        for x, y in ct_get_attackable_tiles_from(
            POSITION_CACHE[query_idx],
            DIRECTION_CACHE[direction],
            ENTITY_TYPE_CACHE[building_type],
        ):
            hazard_idx = y * map_width + x
            if hazard_idx < 0 or hazard_idx >= map_size:
                continue
            hazards[building_type][hazard_idx] += delta

            if hazards[building_type][hazard_idx] > 0:
                bitboard_hazards[building_type] |= BITBOARD_TARGETS[hazard_idx]
            else:
                bitboard_hazards[building_type] &= ~BITBOARD_TARGETS[hazard_idx]
    elif building_type == 6:  # Launcher
        for dx, dy in DIRECTION_DELTAS:
            hazard_x = query_x + dx
            hazard_y = query_y + dy
            if hazard_x < 0 or hazard_x >= map_width or hazard_y < 0 or hazard_y >= map_height:
                continue
            hazard_idx = hazard_y * map_width + hazard_x
            hazards[6][hazard_idx] += delta

            if hazards[6][hazard_idx] > 0:
                bitboard_hazards[6] |= BITBOARD_TARGETS[hazard_idx]
            else:
                bitboard_hazards[6] &= ~BITBOARD_TARGETS[hazard_idx]


def update_sink(query_idx: int, query_x: int, query_y: int, building: int, delta: int):
    """
    Internal method, do not use

    Updates resource sinks created by resource lines such as conveyors, bridges
    and harvesters
    """

    map_width = MAP_INFO.width
    map_height = MAP_INFO.height
    sinks = SINKS

    building_type = building & 0xF
    if building_type == 7 or building_type == 9:  # conveyor or armoured conveyor
        conveyor_direction = (building & 0xFFF0) >> 4
        dx, dy = CONVEYOR_DIRECTIONS[conveyor_direction]
        sink_x = query_x + dx
        sink_y = query_y + dy

        if sink_x < 0 or sink_x >= map_width or sink_y < 0 or sink_y >= map_height:
            return

        sink_idx = sink_y * map_width + sink_x
        (sink := sinks[sink_idx]).num_conveyor_sources += delta
        if delta > 0:
            sink.conveyor_sources.add(query_idx)
        else:
            sink.conveyor_sources.remove(query_idx)

    elif building_type == 8:  # splitter
        splitter_direction = (building & 0xFFF0) >> 4
        splitter_outputs = SPLITTER_DIRECTIONS[splitter_direction]
        for dx, dy in splitter_outputs:
            sink_x = query_x + dx
            sink_y = query_y + dy

            if sink_x < 0 or sink_x >= map_width or sink_y < 0 or sink_y >= map_height:
                continue

            sink_idx = sink_y * map_width + sink_x
            (sink := sinks[sink_idx]).num_conveyor_sources += delta
            if delta > 0:
                sink.conveyor_sources.add(query_idx)
            else:
                sink.conveyor_sources.remove(query_idx)

    elif building_type == 10:  # bridge
        sink_idx = (building & 0xFFF0) >> 4
        (sink := sinks[sink_idx]).num_bridge_sources += delta
        if delta > 0:
            sink.bridge_sources.add(query_idx)
        else:
            sink.bridge_sources.remove(query_idx)

    elif building_type == 11:  # harvester
        for dx, dy in CARDINAL_DIRECTION_DELTAS:
            sink_x = query_x + dx
            sink_y = query_y + dy

            if sink_x < 0 or sink_x >= map_width or sink_y < 0 or sink_y >= map_height:
                continue

            sink_idx = sink_y * map_width + sink_x
            (sink := sinks[sink_idx]).num_harvester_sources += delta
            if delta > 0:
                sink.harvester_sources.add(query_idx)
            else:
                sink.harvester_sources.remove(query_idx)

    elif building_type == 12:  # Foundary
        for dx, dy in CARDINAL_DIRECTION_DELTAS:
            sink_x = query_x + dx
            sink_y = query_y + dy

            if sink_x < 0 or sink_x >= map_width or sink_y < 0 or sink_y >= map_height:
                continue

            sink_idx = sink_y * map_width + sink_x
            (sink := sinks[sink_idx]).num_foundry_sources += delta
            if delta > 0:
                sink.foundry_sources.add(query_idx)
            else:
                sink.foundry_sources.remove(query_idx)


class BitBoardNode:
    __slots__ = "dist", "version"

    def __init__(self):
        self.dist: int | None = 0
        self.version = 0


# TODO Optimize bitboards and make them more user friendly, comment how it works and uses

T = TypeVar("T")


class BitBoardBFS(Generic[T]):
    __slots__ = (
        "root_idx",
        "visited",
        "field",
        "field_version",
        "frontier_archives",
        "archive",
        "archive_size",
        "reachable",
        "unreachable",
        "walls",
        "solve_version",
        "solver",
        "full_mask",
        "max_cost",
        "schedulers",
        "_root_idx",
        "_frontier_archives",
        "_archive",
        "_archive_size",
        "gen",
        "default_inflate_value",
    )

    def __init__(self, solver: Type[T]):
        self.solver = solver()

        # -- SOLVED --

        self.root_idx = -1
        self.visited = 0
        self.field = [BitBoardNode() for _ in range(MAX_MAP_SIZE)]
        self.field_version = 1

        self.frontier_archives: list[list[int]] = [
            ([0] * MAX_MAP_SIZE) for _ in range(solver.NUM_FRONTIERS)
        ]
        self.archive = [0] * MAX_MAP_SIZE
        self.archive_size = 0

        self.reachable = 0
        self.unreachable = 0
        self.walls = 0

        self.solve_version = 0

        self.default_inflate_value = 0

        # -- PARTIAL SOLVE --

        self.full_mask: int = None

        self.max_cost = solver.MAX_COST
        self.schedulers: list[list[int]] = [
            ([0] * (solver.MAX_COST - 1)) for _ in range(solver.NUM_FRONTIERS)
        ]

        self._root_idx = -1

        self._frontier_archives: list[list[int]] = [
            ([0] * MAX_MAP_SIZE) for _ in range(solver.NUM_FRONTIERS)
        ]
        self._archive = [0] * MAX_MAP_SIZE
        self._archive_size = 0

        self.gen: Generator[None | int, Any, None] | None = None

    def invalidate_last_solve(self):
        self.root_idx = -1

    def is_tile_pathable(self, tile_idx: int):
        """Checks if the solver can path onto the given tile"""
        return (BITBOARD_TARGETS[tile_idx] & self.walls) == 0

    def ready(self):
        return self.root_idx != -1

    def init(self, default_inflate_value=0):
        self.full_mask = (1 << MAP_INFO.size) - 1
        self.default_inflate_value = default_inflate_value

    def mark_unreachable(self, tile_idx: int, mark: bool = True):
        tile_bit = 1 << tile_idx
        if mark:
            self.unreachable |= tile_bit
        else:
            self.unreachable &= ~tile_bit
        return self.unreachable

    def mark_reachable(self, tile_idx: int, mark: bool = True):
        tile_bit = 1 << tile_idx
        if mark:
            self.reachable |= tile_bit
        else:
            self.reachable &= ~tile_bit
        return self.reachable

    def get_cpu_time_elapsed(self):
        return CT_GET_CPU_TIME_ELAPSED()

    def pad_1(self, pattern):
        """
        Pad a bit pattern by 1
        """

        map_width = MAP_INFO.width
        map_height = MAP_INFO.height

        not_left_edge, not_right_edge = BITBOARD_EDGES[map_width - 20][map_height - 20]

        horiz = pattern | ((pattern << 1) & not_left_edge) | ((pattern >> 1) & not_right_edge)
        return horiz | (horiz >> map_width) | ((horiz << map_width) & self.full_mask)

    def solve(
        self,
        root_idx: int,
        stop_mask: int = 0,
        stop_mask_any: int = 0,
        stop_cost: int | None = None,
        stop_idx: int | None = None,
        inflate: int | None = None,
        time_limit: int = 500,
        max_depth: int = 0,
    ):
        """
        If a stop position is provided, then solver stops early upon reaching the start
        this creates partial solutions.

        As this incrementally solves over multiple rounds, on root change the solver will
        cancel the previous solve to start the solve on the new root.

        `root_idx`
        - The root of the search

        `stop_idx`
        - Will stop once this position is reached
        - If used with `stop_mask`, will wait for both conditions to be fullfilled.

        `stop_mask`
        - Will stop once all reachable positions in the stop_mask have been visited,
        - If used with `stop_idx`, will wait for both conditions to be fullfilled.

        `stop_mask_any`
        - Will stop if any of the positions in this mask is reached

        `stop_cost`
        - Will stop once cost to reach a given tile exceeds this value

        `inflate`
        - If the search would early stop, keeps searching this many extra iterations.
        - NOTE doesn't necessarily search this many extra tiles, depends on `self.max_cost`
        """

        if root_idx != self._root_idx:
            # root changed, performing a different solve
            # clear current solve and start new solve at new root
            #
            # NOTE if the root changes too often, can cause solver to never finish
            #      and thus you never get a result. It is the responsibility of the user
            #      to wait for `.ready()` before switching solver root if this is a problem.
            self._root_idx = root_idx
            self.gen = None

        if root_idx != self.root_idx:
            # root changed from currently available root, invalidate information
            self.root_idx = -1

        if self.gen is None:
            # restart solver

            # reset schedulers
            for scheduler in self.schedulers:
                for i in range(self.max_cost - 1):
                    scheduler[i] = 0

            # reset archive
            self._archive_size = 0

            if inflate is None:
                inflate = self.default_inflate_value

            self.gen = self.solver.solve(
                self,
                root_idx,
                stop_mask,
                stop_mask_any,
                stop_cost,
                stop_idx,
                inflate,
                time_limit,
                max_depth,
            )

        self.walls = self.unreachable
        self.solver.update(self)
        self.walls &= ~self.reachable

        result = next(self.gen)
        if result is not None:
            # Mark as solved
            self.root_idx = root_idx
            self.solve_version += 1
            self.field_version += 1
            self.frontier_archives, self._frontier_archives = (
                self._frontier_archives,
                self.frontier_archives,
            )
            self.archive, self._archive = self._archive, self.archive
            self.archive_size, self._archive_size = (
                self._archive_size,
                self.archive_size,
            )
            self.visited = result
            self.gen = None
            return True
        return False

    def get_next_tile(self, tile_idx: int, *args, **kwargs):
        if self.root_idx == -1:
            raise Exception("Cannot access an unsolved bitboard")

        t = self.dist(tile_idx)

        if t is None or t == 0:
            return None

        return self.solver.get_next_tile(self, tile_idx, t, *args, **kwargs)

    def get_prev_tile(self, tile_idx: int, *args, **kwargs):
        if self.root_idx == -1:
            raise Exception("Cannot access an unsolved bitboard")

        t = self.dist(tile_idx)

        if t is None:
            return None

        return self.solver.get_prev_tile(self, tile_idx, t, *args, **kwargs)

    def has_visited(self, tile_idx: int):
        if self.root_idx == -1:
            raise Exception("Cannot access an unsolved bitboard")

        return (self.visited & BITBOARD_TARGETS[tile_idx]) != 0

    def dist(self, tile_idx: int):
        if self.root_idx == -1:
            raise Exception("Cannot access an unsolved bitboard")

        if tile_idx == self.root_idx:
            return 0

        node = self.field[tile_idx]
        if node.version == self.field_version:
            return node.dist

        node.version = self.field_version
        target = BITBOARD_TARGETS[tile_idx]

        if self.visited & target:
            # Binary search is fine, but tighten bounds using archive_size directly
            lo = 0
            hi = self.archive_size - 1
            archive = self.archive
            while lo < hi:
                mid = (lo + hi) >> 1
                if archive[mid] & target:
                    hi = mid
                else:
                    lo = mid + 1
            node.dist = lo
        else:
            node.dist = None

        return node.dist

    def debug_archive(self, archive_idx: int):
        if self.root_idx == -1:
            raise Exception("Cannot access an unsolved bitboard")

        self.debug_bitmap(self.archive[archive_idx])

    def debug_frontier_archive(self, frontier_idx: int, archive_idx: int):
        if self.root_idx == -1:
            raise Exception("Cannot access an unsolved bitboard")

        self.debug_bitmap(self.frontier_archives[frontier_idx][archive_idx])

    def debug_bitmap(self, bitmap: int):
        for y in range(MAP_INFO.height):
            row = ""
            for x in range(MAP_INFO.width):
                idx = y * MAP_INFO.width + x
                if bitmap & (1 << idx):
                    row += f" []"
                else:
                    row += f" .."
            print(row)

    def debug_field(self):
        for y in range(MAP_INFO.height):
            row = ""
            for x in range(MAP_INFO.width):
                idx = y * MAP_INFO.width + x
                if self.walls & (1 << idx):
                    row += " ..."
                else:
                    dist = self.dist(idx)
                    if dist is None:
                        row += " ???"
                    else:
                        row += f" {dist:3}"
            print(row)

    def debug_visited(self):
        if self.root_idx == -1:
            raise Exception("Cannot access an unsolved bitboard")

        for y in range(MAP_INFO.height):
            row = ""
            for x in range(MAP_INFO.width):
                idx = y * MAP_INFO.width + x
                if self.visited & (1 << idx):
                    row += f"1"
                else:
                    row += f"0"
            print(row)

    def debug_path(self, start_idx: int):
        if self.root_idx == -1:
            raise Exception("Cannot access an unsolved bitboard")

        current_idx = start_idx
        while True:
            next_idx = self.get_next_tile(current_idx)
            if next_idx is None or next_idx == current_idx:
                break
            CT_DRAW_INDICATOR_LINE(POSITION_CACHE[current_idx], POSITION_CACHE[next_idx], 0, 255, 0)
            CT_DRAW_INDICATOR_DOT(POSITION_CACHE[current_idx], 0, 255, 0)
            current_idx = next_idx


# ------------- HELPER METHODS -------------


def pos_to_idx(pos: Position):
    """Converts a position to index"""
    return pos.y * MAP_INFO.width + pos.x


def xy_to_idx(x: int, y: int):
    """Converts a position to index"""
    return y * MAP_INFO.width + x


def idx_to_pos(idx: int):
    """Converts an index to position"""
    return POSITION_CACHE[idx]


def in_bounds(x: int, y: int):
    return not (x < 0 or y < 0 or x >= MAP_INFO.width or y >= MAP_INFO.height)


def not_in_bounds(x: int, y: int):
    return x < 0 or y < 0 or x >= MAP_INFO.width or y >= MAP_INFO.height


def best_enemy_core_idx():
    """Gets the best candidate enemy core position based on map symmetry"""
    return ENEMY_CORE_POS[0]


def best_mirror_env():
    """Gets the best candidate environment layout based on map symmetry"""
    return MIRROR_ENV[0][0]


def get_flow_rate(type: int, tile_idx: int):
    """
    `type`:
    - 0 = titanium
    - 1 = axionite
    - 2 = refined axionite
    """
    return RESOURCE_FLOW[type][tile_idx]._flow_value / 4


def get_stall_rate(type: int, tile_idx: int):
    """
    `type`:
    - 0 = titanium
    - 1 = axionite
    - 2 = refined axionite
    """
    return RESOURCE_FLOW[type][tile_idx]._stall_value / 4


def get_bot_hp(bot: int):
    """
    Gets the bot hp from bot tile information

    ```
    get_bot_hp(ALLY_BUILDER_BOT[tile_idx])
    ```
    """
    if bot == 0:
        return None
    return (bot >> 16) & 0xFFFF


def get_bot_id(bot: int):
    """
    Gets the bot id from bot tile information

    ```
    get_bot_id(ALLY_BUILDER_BOT[tile_idx])
    ```
    """
    if bot == 0:
        return None
    return bot & 0xFFFF


def get_building_type(building: int):
    """
    Gets the building entity type from building tile information

    ```
    get_building_type(ALLY_BUILDINGS[tile_idx])
    ```
    """
    return ENTITY_TYPE_CACHE[building & 0xF]


def get_building_maxhp(building: int):
    """
    Gets the building max hp from building tile information

    ```
    get_building_maxhp(ALLY_BUILDINGS[tile_idx])
    ```
    """
    return MAX_HPS[building & 0xF]


def get_building_info(tile_idx: int):
    """
    Gets the building info regardless of team
    """
    if ALLY_BUILDINGS[tile_idx] != 0:
        return ALLY_BUILDINGS[tile_idx]
    if ENEMY_BUILDINGS[tile_idx] != 0:
        return ENEMY_BUILDINGS[tile_idx]
    return None


def get_building_id(building: int):
    """
    Gets the building id from building tile information

    ```
    get_building_id(ALLY_BUILDINGS[tile_idx])
    ```
    """
    if building is None or building == 0:
        return None
    return (building >> 16) & 0xFFFF


def get_building_hp(building: int):
    """
    Gets building hp from building tile information

    ```
    get_building_hp(ALLY_BUILDINGS[tile_idx])
    ```
    """
    if building is None or building == 0:
        return None
    return building >> 32


def get_building_direction(building: int):
    """
    Gets building direction from building tile information

    ```
    get_building_direction(ALLY_BUILDINGS[tile_idx])
    ```
    """
    if building is None or building == 0:
        return None
    return DIRECTION_CACHE[(building >> 4) & 0xFFF]


def get_bridge_target_idx(building: int):
    """
    Gets bridge target tile index from building tile information

    ```
    get_bridge_target_idx(ALLY_BUILDINGS[tile_idx])
    ```
    """
    if building is None or building == 0:
        return None
    return (building >> 4) & 0xFFF


def vision_iter() -> Iterator[int]:
    """
    Provides an iterator over tile indices within bot vision

    ```
    for tile_idx in vision_iter():
        ...
    ```
    """

    ct_is_in_vision = CT_IS_IN_VISION

    vision_width = VISION_WIDTH
    map_width = MAP_INFO.width
    map_size = MAP_INFO.size
    position_cache = POSITION_CACHE

    pos = CT_GET_POSITION()
    pos_x, pos_y = pos.x, pos.y

    origin_x, origin_y = pos_x - VISION_RADIUS, pos_y - VISION_RADIUS

    for y in range(vision_width):
        query_idx = origin_x + (origin_y + y) * map_width

        for _ in range(vision_width):
            if query_idx < 0 or query_idx >= map_size:
                query_idx += 1
                continue

            if not ct_is_in_vision(position_cache[query_idx]):
                query_idx += 1
                continue

            yield query_idx
            query_idx += 1


# ------------- DEBUGGING -------------


def print_env_grid():
    for i in range(MAP_INFO.height):
        row = ""
        for j in range(MAP_INFO.width):
            idx = i * MAP_INFO.width + j
            if ENV[idx] == Environment.WALL:
                row += f"W"
            elif ENV[idx] == Environment.ORE_AXIONITE:
                row += f"A"
            elif ENV[idx] == Environment.ORE_TITANIUM:
                row += f"T"
            elif ENV[idx] is None:
                row += f"-"
            else:
                row += f"*"
        print(row)


def print_mirror_grid(index: int):
    env = MIRROR_ENV[index][0]
    for i in range(MAP_INFO.height):
        row = ""
        for j in range(MAP_INFO.width):
            idx = i * MAP_INFO.width + j
            if env[idx] == Environment.WALL:
                row += f"W"
            elif env[idx] == Environment.ORE_AXIONITE:
                row += f"A"
            elif env[idx] == Environment.ORE_TITANIUM:
                row += f"T"
            elif env[idx] is None:
                row += f"-"
            else:
                row += f"*"
        print(row)


def print_env_bitboard(bitboard: list[int]):
    for i in range(MAP_INFO.height):
        row = ""
        for j in range(MAP_INFO.width):
            idx = i * MAP_INFO.width + j
            if (bit_cell := (bitboard[idx] & 0b1111)) != 0:
                if (bit_cell & 0b1) != 0:
                    row += f"*"
                elif (bit_cell & 0b10) != 0:
                    row += f"W"
                elif (bit_cell & 0b100) != 0:
                    row += f"T"
                elif (bit_cell & 0b1000) != 0:
                    row += f"A"
                else:
                    row += f"?"
            else:
                row += f"."
        print(row)


def print_flow(type: int):
    flows = RESOURCE_FLOW[type]
    for i in range(MAP_INFO.height):
        row = ""
        for j in range(MAP_INFO.width):
            idx = i * MAP_INFO.width + j
            if flows[idx]._flow_value > 0:
                row += f"X"
            else:
                row += f"."
        print(row)


def print_flow_values(type: int):
    flows = RESOURCE_FLOW[type]
    for i in range(MAP_INFO.height):
        row = ""
        for j in range(MAP_INFO.width):
            idx = i * MAP_INFO.width + j
            if flows[idx]._flow_value > 0:
                row += f"{(flows[idx]._flow_value / 4):.2f} "
            else:
                row += f".... "
        print(row)


def print_flow_values_combined():
    for i in range(MAP_INFO.height):
        row = ""
        for j in range(MAP_INFO.width):
            idx = i * MAP_INFO.width + j
            flow_value = max(
                RESOURCE_FLOW[0][idx]._flow_value,
                RESOURCE_FLOW[1][idx]._flow_value,
                RESOURCE_FLOW[2][idx]._flow_value,
            )
            if flow_value > 0:
                row += f"{(flow_value / 4):.2f} "
            else:
                row += f".... "
        print(row)


def print_stall_values(type: int):
    flows = RESOURCE_FLOW[type]
    for i in range(MAP_INFO.height):
        row = ""
        for j in range(MAP_INFO.width):
            idx = i * MAP_INFO.width + j
            if flows[idx]._stall_value > 0:
                row += f"{(flows[idx]._stall_value / 16):.2f} "
            else:
                row += f".... "
        print(row)


def print_stall_values_combined():
    for i in range(MAP_INFO.height):
        row = ""
        for j in range(MAP_INFO.width):
            idx = i * MAP_INFO.width + j
            stall_value = max(
                RESOURCE_FLOW[0][idx]._stall_value,
                RESOURCE_FLOW[1][idx]._stall_value,
                RESOURCE_FLOW[2][idx]._stall_value,
            )
            if stall_value > 0:
                row += f"{(stall_value / 4):.2f} "
            else:
                row += f".... "
        print(row)


def print_sinks():
    sinks = SINKS
    for i in range(MAP_INFO.height):
        row = ""
        for j in range(MAP_INFO.width):
            idx = i * MAP_INFO.width + j
            if sinks[idx].num_conveyor_sources > 0:
                row += f"c{(sinks[idx].num_conveyor_sources):2} "
            elif sinks[idx].num_bridge_sources > 0:
                row += f"b{(sinks[idx].num_bridge_sources):2} "
            elif sinks[idx].num_harvester_sources > 0:
                row += f"h{(sinks[idx].num_harvester_sources):2} "
            else:
                row += f"... "
        print(row)


def print_hazards(type: EntityType):
    hazards = HAZARDS[HASHMAP_ENTITY_TYPE[type]]
    for i in range(MAP_INFO.height):
        row = ""
        for j in range(MAP_INFO.width):
            idx = i * MAP_INFO.width + j
            if hazards[idx] > 0:
                row += f"X"
            else:
                row += f"."
        print(row)


def print_ent_bitboard(bitboard: list[int]):
    for i in range(MAP_INFO.height):
        row = ""
        for j in range(MAP_INFO.width):
            idx = i * MAP_INFO.width + j
            if (bit_cell := (bitboard[idx] & 0xFFFF)) != 0:
                if (bit_cell & 0b10) != 0:
                    row += f"*"
                elif (bit_cell & 0b100) != 0:
                    row += f"C"
                elif (bit_cell & 0b1000) != 0:
                    row += f"G"
                elif (bit_cell & 0b1_0000) != 0:
                    row += f"S"
                elif (bit_cell & 0b10_0000) != 0:
                    row += f"B"
                elif (bit_cell & 0b100_0000) != 0:
                    row += f"L"
                elif (bit_cell & 0b1000_0000) != 0:
                    row += f"c"
                elif (bit_cell & 0b1_0000_0000) != 0:
                    row += f"s"
                elif (bit_cell & 0b10_0000_0000) != 0:
                    row += f"a"
                elif (bit_cell & 0b100_0000_0000) != 0:
                    row += f"b"
                elif (bit_cell & 0b1000_0000_0000) != 0:
                    row += f"H"
                elif (bit_cell & 0b1_0000_0000_0000) != 0:
                    row += f"F"
                elif (bit_cell & 0b10_0000_0000_0000) != 0:
                    row += f"r"
                elif (bit_cell & 0b100_0000_0000_0000) != 0:
                    row += f"#"
                elif (bit_cell & 0b1000_0000_0000_0000) != 0:
                    row += f"m"
                else:
                    row += f"?"
            else:
                row += f"."
        print(row)


def print_buildings(buildings: list[int]):
    for i in range(MAP_INFO.height):
        row = ""
        for j in range(MAP_INFO.width):
            idx = i * MAP_INFO.width + j
            if (building_type := (buildings[idx] & 0xF)) != 0:
                if building_type == 1:
                    row += f"*"
                elif building_type == 2:
                    row += f"C"
                elif building_type == 3:
                    row += f"G"
                elif building_type == 4:
                    row += f"S"
                elif building_type == 5:
                    row += f"B"
                elif building_type == 6:
                    row += f"L"
                elif building_type == 7:
                    row += f"c"
                elif building_type == 8:
                    row += f"s"
                elif building_type == 9:
                    row += f"a"
                elif building_type == 10:
                    row += f"b"
                elif building_type == 11:
                    row += f"H"
                elif building_type == 12:
                    row += f"F"
                elif building_type == 13:
                    row += f"r"
                elif building_type == 14:
                    row += f"#"
                elif building_type == 15:
                    row += f"m"
                else:
                    row += f"?"
            else:
                row += f"."
        import sys

        print(row, file=sys.stderr)


def format_bits(n: int, width: int = None, group: int = 4) -> str:
    # Determine width automatically if not provided
    if width is None:
        width = max(1, n.bit_length())

    # Round width up to a multiple of group size
    if width % group != 0:
        width += group - (width % group)

    # Format as zero-padded binary
    bits = format(n, f"0{width}b")

    # Insert underscores every `group` bits
    grouped = "_".join(bits[i : i + group] for i in range(0, len(bits), group))

    return f"0b{grouped}"
