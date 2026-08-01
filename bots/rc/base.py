from cambc import Position, Controller


class BaseBot:

    def __init__(self, map_width: int, map_height: int) -> None:
        """Store immutable map dimensions shared by every bot role."""
        self.map_width = map_width
        self.map_height = map_height

        self.max_cpu_cost = 0
        self.rolling_avg_cpu_cost = 0

    def run(self, c: Controller) -> None:
        """Execute common per-turn accounting; subclasses extend this behavior."""
        # Log CPU costs
        cpu_cost = c.get_cpu_time_elapsed()
        if cpu_cost > self.max_cpu_cost:
            self.max_cpu_cost = cpu_cost
        self.rolling_avg_cpu_cost = (self.rolling_avg_cpu_cost * 39 + cpu_cost) / 40
        print(f"[{self}] avg cpu: {self.rolling_avg_cpu_cost}")
        print(f"[{self}] max cpu: {self.max_cpu_cost}")

    def in_bounds(self, pos: Position) -> bool:
        """Return whether ``pos`` lies inside the current map."""
        return 0 <= pos.x < self.map_width and 0 <= pos.y < self.map_height
