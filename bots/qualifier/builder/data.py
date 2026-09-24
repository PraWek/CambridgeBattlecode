from botlib import MAX_MAP_SIZE, BitBoardBFS

from .pathing import BotBitboardSolver, ResourceBitboardSolver, BotUniformBitboardSolver

# ------------- GLOBALS -------------

DISTANCE_FIELD = BitBoardBFS[BotUniformBitboardSolver](BotUniformBitboardSolver)
BOT_PATHING = BitBoardBFS[BotBitboardSolver](BotBitboardSolver)
RESOURCE_PATHING = BitBoardBFS[ResourceBitboardSolver](ResourceBitboardSolver)

ORES_COMPLETED = [0] * MAX_MAP_SIZE
BRIDGE_COOP_COOLDOWN = [0] * MAX_MAP_SIZE

# TODO botlib should track enemy and ally harvesters placed
#      for easy consideration
TITANIUM_HARVESTERS_PLACED: set = set()


SW_VISION_DELTAS = [
    (-5, -1),
    (-5, 0),
    (-5, 1),
    (-5, 2),
    (-5, 3),
    (-4, 3),
    (-4, 4),
    (-3, 4),
    (-3, 5),
    (-2, 5),
    (-1, 5),
    (0, 5),
    (1, 5),
]

S_VISION_DELTAS = [
    (-4, 3),
    (-3, 4),
    (-2, 5),
    (-1, 5),
    (0, 5),
    (1, 5),
    (2, 5),
    (3, 4),
    (4, 3),
]

SE_VISION_DELTAS = [
    (5, -1),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 3),
    (4, 3),
    (4, 4),
    (3, 4),
    (3, 5),
    (2, 5),
    (1, 5),
    (0, 5),
    (-1, 5),
]

W_VISION_DELTAS = [
    (-3, -4),
    (-4, -3),
    (-5, -2),
    (-5, -1),
    (-5, 0),
    (-5, 1),
    (-5, 2),
    (-4, 3),
    (-3, 4),
]

E_VISION_DELTAS = [
    (3, -4),
    (4, -3),
    (5, -2),
    (5, -1),
    (5, 0),
    (5, 1),
    (5, 2),
    (4, 3),
    (3, 4),
]

NW_VISION_DELTAS = [
    (-5, 1),
    (-5, 0),
    (-5, -1),
    (-5, -2),
    (-5, -3),
    (-4, -3),
    (-4, -4),
    (-3, -4),
    (-3, -5),
    (-2, -5),
    (-1, -5),
    (0, -5),
    (1, -5),
]

N_VISION_DELTAS = [
    (-4, -3),
    (-3, -4),
    (-2, -5),
    (-1, -5),
    (0, -5),
    (1, -5),
    (2, -5),
    (3, -4),
    (4, -3),
]

NE_VISION_DELTAS = [
    (5, 1),
    (5, 0),
    (5, -1),
    (5, -2),
    (5, -3),
    (4, -3),
    (4, -4),
    (3, -4),
    (3, -5),
    (2, -5),
    (1, -5),
    (0, -5),
    (-1, -5),
]

VISION_DELTAS = [
    N_VISION_DELTAS,
    NE_VISION_DELTAS,
    E_VISION_DELTAS,
    SE_VISION_DELTAS,
    S_VISION_DELTAS,
    SW_VISION_DELTAS,
    W_VISION_DELTAS,
    NW_VISION_DELTAS,
]

CORE_NEIGHBOUR_DELTAS = [
    (0, -2),
    (1, -2),
    (2, -1),
    (2, 0),
    (2, 1),
    (1, 2),
    (0, 2),
    (-1, 2),
    (-2, 1),
    (-2, 0),
    (-2, -1),
    (-1, -2),
]

CORE_DELTAS = [
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
]

CORE_CANDIDATE_LOCATIONS = [
    (-2, -2),
    (-2, -1),
    (-2, 0),
    (-2, 1),
    (-2, 2),
    (-1, -2),
    (-1, 2),
    (0, -2),
    (0, 2),
    (1, -2),
    (1, 2),
    (2, -2),
    (2, -1),
    (2, 0),
    (2, 1),
    (2, 2),
    (-3, -3),
    (-3, -2),
    (-3, -1),
    (-3, 0),
    (-3, 1),
    (-3, 2),
    (-3, 3),
    (-2, -3),
    (-2, 3),
    (-1, -3),
    (-1, 3),
    (0, -3),
    (0, 3),
    (1, -3),
    (1, 3),
    (2, -3),
    (2, 3),
    (3, -3),
    (3, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (-4, -4),
    (-4, -3),
    (-4, -2),
    (-4, -1),
    (-4, 0),
    (-4, 1),
    (-4, 2),
    (-4, 3),
    (-4, 4),
    (-3, -4),
    (-3, 4),
    (-2, -4),
    (-2, 4),
    (-1, -4),
    (-1, 4),
    (0, -4),
    (0, 4),
    (1, -4),
    (1, 4),
    (2, -4),
    (2, 4),
    (3, -4),
    (3, 4),
    (4, -4),
    (4, -3),
    (4, -2),
    (4, -1),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
]
