"""Newborn role handoff shared by combat and economic builders."""

from cambc import EntityType

from constants import (
    BUILDER_CODE_DIRECTIONS, MARKER_KIND_SPAWN_DIRECTION,
    MARKER_KIND_SPAWN_INTRUDER,
)
from geometry import decode_marker_coordinates


def read_spawn_assignment(controller):
    """Capture a friendly handoff before the first multi-turn terrain scan."""
    current = controller.get_position()
    team = controller.get_team()
    direction = None
    for entity_id in controller.get_nearby_buildings():
        if controller.get_entity_type(entity_id) != EntityType.MARKER:
            continue
        if controller.get_team(entity_id) != team:
            continue
        kind, x, y, payload = decode_marker_coordinates(controller.get_marker_value(entity_id))
        if (x, y) != (current.x, current.y):
            continue
        if kind == MARKER_KIND_SPAWN_INTRUDER:
            return True, None
        if kind == MARKER_KIND_SPAWN_DIRECTION:
            direction = BUILDER_CODE_DIRECTIONS.get(payload)
    return False, direction

