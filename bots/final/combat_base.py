"""RC's resumable combat searches over Nexus's shared observation cache."""

from base import BaseBot
from combat_navigation import AStarSearchState


class CombatBot(BaseBot):
    def __init__(self, map_width: int, map_height: int) -> None:
        super().__init__(map_width, map_height)
        self._a_star_states: dict[str, AStarSearchState] = {}

    def a_star_state(self, name: str) -> AStarSearchState:
        state = self._a_star_states.get(name)
        if state is None:
            state = AStarSearchState()
            self._a_star_states[name] = state
        return state

    def clear_a_star_state(self, name: str) -> None:
        state = self._a_star_states.get(name)
        if state is not None:
            state.finish()

    def a_star_pending(self, name: str) -> bool:
        state = self._a_star_states.get(name)
        return state is not None and state.pending
