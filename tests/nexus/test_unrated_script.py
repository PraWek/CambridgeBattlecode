from __future__ import annotations

import unittest

from run_nexus_unrated_matches import (
    Game,
    nexus_side,
    nexus_won,
    parse_match_info,
    safe_name,
)


class NexusUnratedScriptTests(unittest.TestCase):
    def test_match_info_parser_finds_team_and_games(self) -> None:
        output = """
Status: complete
Team A: Nexus (our-team)
Team B: Rival (enemy-team)
│ 1 │ thread_of_connection │ A Nexus │ Titanium collected │ ready │
│ 2 │ strangecurves        │ B Rival │ Core destroyed      │ ready │
"""
        info = parse_match_info(output)

        self.assertEqual(info.status, "complete")
        self.assertEqual(nexus_side(info, "enemy-team"), "A")
        self.assertEqual([game.map_name for game in info.games], [
            "thread_of_connection",
            "strangecurves",
        ])
        self.assertTrue(nexus_won(info.games[0], "A"))
        self.assertFalse(nexus_won(info.games[1], "A"))

    def test_replay_filename_is_portable(self) -> None:
        self.assertEqual(safe_name('bad<map>:name'), "bad_map__name")
        self.assertFalse(nexus_won(Game(1, "map", "--", "pending"), "A"))


if __name__ == "__main__":
    unittest.main()
