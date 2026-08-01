from cambc import Controller, Direction, GameConstants, Position

from base import BaseBot
from constants import ASSIGNMENT_DIRECTIONS, MARKER_KIND_DIRECTION
from geometry import direction_index, encode_marker


CORE_SCOUT_DIRECTIONS = ASSIGNMENT_DIRECTIONS[:4]


class CoreBot(BaseBot):
    def __init__(self, map_width: int, map_height: int) -> None:
        super().__init__(map_width, map_height)
        self.core_pos: Position | None = None
        self.next_direction_index = 0
        self.assignment_pads: dict[int, Position] = {}

    def run(self, c: Controller) -> None:
        super().run(c)
        self.core_pos = c.get_position()
        self.ensure_assignment_pads()
        self.skip_out_of_bounds_directions()

        if self.next_direction_index >= len(CORE_SCOUT_DIRECTIONS):
            return

        assigned_direction = CORE_SCOUT_DIRECTIONS[self.next_direction_index]
        pad = self.assignment_pads[self.next_direction_index]
        if self.in_bounds(pad) and c.can_place_marker(pad):
            c.place_marker(pad, encode_marker(MARKER_KIND_DIRECTION, direction_index(assigned_direction)))

        if c.get_unit_count() >= GameConstants.MAX_TEAM_UNITS:
            return

        titanium, axionite = c.get_global_resources()
        builder_cost, axionite_cost = c.get_builder_bot_cost()
        if titanium < builder_cost:
            return

        spawn_pos = self.core_pos.add(assigned_direction)
        if c.can_spawn(spawn_pos):
            c.spawn_builder(spawn_pos)
            self.next_direction_index += 1

    def skip_out_of_bounds_directions(self) -> None:
        if self.core_pos is None:
            return

        while self.next_direction_index < len(CORE_SCOUT_DIRECTIONS):
            spawn_pos = self.core_pos.add(CORE_SCOUT_DIRECTIONS[self.next_direction_index])
            if self.in_bounds(spawn_pos):
                return
            self.next_direction_index += 1

    def ensure_assignment_pads(self) -> None:
        if self.core_pos is None or self.assignment_pads:
            return
        offsets = {
            Direction.NORTH: (-2, -1),
            Direction.EAST: (1, -2),
            Direction.SOUTH: (2, 1),
            Direction.WEST: (-1, 2),
            Direction.NORTHEAST: (2, -2),
            Direction.SOUTHEAST: (2, 0),
            Direction.SOUTHWEST: (-2, 2),
            Direction.NORTHWEST: (-2, 0),
        }
        for idx, direction in enumerate(ASSIGNMENT_DIRECTIONS):
            dx, dy = offsets[direction]
            self.assignment_pads[idx] = Position(self.core_pos.x + dx, self.core_pos.y + dy)
