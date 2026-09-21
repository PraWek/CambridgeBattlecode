"""Cache the known terrain component a builder can reach to install transport."""
from collections import deque

from cambc import Environment


class ConstructionAccess:
    def __init__(self):
        self.signature = None
        self.tiles = set()
        self.queue = deque()
        self.pending = False

    def reachable(self, start, environments, blocked, neighbor, directions, budget=None):
        # Terrain only grows; walls and ore never change.  Builder occupancy
        # is transient and must not split this permanent construction graph.
        signature = (len(environments), frozenset(blocked))
        if not self.pending and signature == self.signature and start in self.tiles:
            return self.tiles
        if not self.pending or start not in self.tiles or frozenset(blocked) != self.signature[1]:
            self.tiles = {start}
            self.queue = deque([start])
            self.signature = signature
            self.pending = True
        tiles, queue = self.tiles, self.queue
        expansions = 0
        while queue:
            if budget is not None and expansions % 16 == 0:
                budget.checkpoint()
            current = queue.popleft()
            expansions += 1
            for direction in directions:
                target = neighbor(current, direction)
                if target is None or target in tiles or target in blocked:
                    continue
                if environments.get(target) != Environment.EMPTY:
                    continue
                tiles.add(target)
                queue.append(target)
        # Keep the starting terrain version. New observations during a slice
        # may border already-expanded cells; the next call must revisit those.
        self.tiles = tiles
        self.pending = False
        return tiles
