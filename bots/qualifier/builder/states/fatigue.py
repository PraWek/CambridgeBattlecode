from typing import Callable


class task_fatigue:
    """Generic fatigue tracker"""

    def __init__(self, fatigue: int):
        self.fatigue = fatigue

    def is_fatigued(self):
        if self.fatigue > 0:
            self.fatigue -= 1
            return True
        return False
