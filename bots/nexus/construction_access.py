"""Cache the known terrain component a builder can reach to install transport."""
from collections import deque

from cambc import Environment


class ConstructionAccess:
    def __init__(self):
        self.signature = None
        self.tiles = set()

    def reachable(self, start, environments, blocked, neighbor, directions):
        # Terrain only grows; walls and ore never change.  Builder occupancy
        # is transient and must not split this permanent construction graph.
        signature = (len(environments), frozenset(blocked))
        if signature == self.signature and start in self.tiles:
            return self.tiles
        tiles = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for direction in directions:
                target = neighbor(current, direction)
                if target is None or target in tiles or target in blocked:
                    continue
                if environments.get(target) != Environment.EMPTY:
                    continue
                tiles.add(target)
                queue.append(target)
        self.signature = signature
        self.tiles = tiles
        return tiles
