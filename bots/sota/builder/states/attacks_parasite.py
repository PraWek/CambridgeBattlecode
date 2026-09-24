"""
Parasite attacks are attacks that utilize enemy lines and their gathered resources
rather than our own.
"""

from typing import Any

from cambc import Controller, Direction, EntityType, Position

from botlib import State
from .. import STATE

from botlib import MAP_INFO, UNIT_INFO
from botlib import best_enemy_core_idx

from botlib.constants import POSITION_CACHE, DIRECTION_CACHE

from ..utility.movement import move_to

from ..data import BOT_PATHING

# ------------- STATES -------------
