#!/usr/bin/env python3
"""Submit RC, start unrated matches, and collect their results with cambc."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterable


OPPONENT_ID = "2c1ad21b-c935-4074-84d9-17e03d0c2fe3"
TERMINAL_STATUSES = {"complete", "error", "cancelled", "canceled", "failed"}

MATCH_ID_RE = re.compile(r"\bMatch ID:\s*([A-Za-z0-9_-]+)", re.IGNORECASE)
ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
STATUS_RE = re.compile(r"^\s*Status:\s*(\w+)", re.IGNORECASE)
TEAM_RE = re.compile(r"^\s*Team ([AB]):\s*(.*?)\s*\(([^)]+)\)", re.IGNORECASE)
UNSAFE_FILE_NAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class QueuedMatch:
    number: int
    match_id: str


@dataclass(frozen=True)
class Game:
    number: int
    map_name: str
    winner: str
    condition: str


@dataclass(frozen=True)
class MatchInfo:
    status: str
    teams: dict[str, str]
    games: list[Game]
    raw_output: str


def find_cambc(root: Path) -> str:
    """Use cambc from PATH, or the virtual environments already in the project."""
    if installed := shutil.which("cambc"):
        return installed

    executable = "cambc.exe" if os.name == "nt" else "cambc"
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    for environment in (".venv", ".venv-3.13", ".venv-3.12"):
        candidate = root / environment / scripts_dir / executable
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError("cambc was not found. Install it or activate its virtual environment.")


def display_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return " ".join(command)


def run_cambc(root: Path, cambc: str, args: list[str]) -> str:
    """Run one cambc command and return its plain-text output."""
    command = [cambc, *args]
    environment = dict(os.environ)
    environment["NO_COLOR"] = "1"
    completed = subprocess.run(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"cambc exited with code {completed.returncode}: {display_command(command)}\n"
            f"{completed.stdout.rstrip()}"
        )
    return completed.stdout


def queue_match(root: Path, cambc: str, opponent_id: str, number: int) -> QueuedMatch:
    output = run_cambc(root, cambc, ["match", "unrated", opponent_id])
    id_match = MATCH_ID_RE.search(output)
    if not id_match:
        raise RuntimeError(f"Match {number}: no Match ID in cambc output:\n{output.rstrip()}")

    return QueuedMatch(number, id_match.group(1))


def plain_lines(output: str) -> Iterable[str]:
    for line in ANSI_RE.sub("", output).splitlines():
        yield line.rstrip()


def parse_game_row(line: str) -> Game | None:
    """Parse a row of the Rich table printed by `cambc match info`."""
    separator = "│" if "│" in line else "|" if "|" in line else None
    if separator is None:
        return None

    cells = [cell.strip() for cell in line.split(separator)]
    # Rich tables have an empty cell at each outer border.
    cells = [cell for cell in cells if cell]
    if len(cells) != 5 or not cells[0].isdigit():
        return None

    return Game(
        number=int(cells[0]),
        map_name=cells[1],
        winner=cells[2],
        condition=cells[3],
    )


def parse_match_info(output: str) -> MatchInfo:
    """Extract the small amount of data needed from `cambc match info` output."""
    status = "unknown"
    teams: dict[str, str] = {}
    games: list[Game] = []

    for line in plain_lines(output):
        if status_match := STATUS_RE.match(line):
            status = status_match.group(1).lower()
        if team_match := TEAM_RE.match(line):
            teams[team_match.group(1).upper()] = team_match.group(3)
        if game := parse_game_row(line):
            games.append(game)

    return MatchInfo(status=status, teams=teams, games=games, raw_output=output)


def rc_side(info: MatchInfo, opponent_id: str) -> str | None:
    """Identify RC by excluding the opponent ID supplied to `match unrated`."""
    if info.teams.get("A") == opponent_id:
        return "B"
    if info.teams.get("B") == opponent_id:
        return "A"
    return None


def safe_map_name(map_name: str) -> str:
    """Turn a map name into a portable filename component."""
    cleaned = UNSAFE_FILE_NAME_RE.sub("_", map_name).strip(" .")
    return cleaned or "unknown_map"


def download_replay(
    root: Path, cambc: str, match: QueuedMatch, game: Game, replay_dir: Path,
) -> Path | None:
    """Download one losing game replay and include its map name in the filename."""
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_path = replay_dir / (
        f"{match.match_id}_game_{game.number}_{safe_map_name(game.map_name)}.replay26"
    )
    command = [
        "match",
        "replay",
        match.match_id,
        "--game",
        str(game.number),
        "--output",
        str(replay_path),
    ]
    print(f"    $ {display_command([cambc, *command])}")
    output = run_cambc(root, cambc, command)
    if output:
        print(output.rstrip())
    if replay_path.is_file():
        print(f"    Saved: {replay_path.resolve()}")
        return replay_path

    print("    cambc did not create a replay file.", file=sys.stderr)
    return None


def monitor_matches(
    root: Path, cambc: str, matches: list[QueuedMatch], poll_seconds: int, timeout_minutes: float,
) -> dict[str, MatchInfo]:
    deadline = time.monotonic() + timeout_minutes * 60 if timeout_minutes else None
    results: dict[str, MatchInfo] = {}
    previous: dict[str, tuple[str, int]] = {}

    while len(results) < len(matches):
        for match in matches:
            if match.match_id in results:
                continue

            info = parse_match_info(run_cambc(root, cambc, ["match", "info", match.match_id]))
            progress = (info.status, len(info.games))
            if progress != previous.get(match.match_id):
                print(f"[{match.number}] {info.status}; games reported: {len(info.games)}/5", flush=True)
                previous[match.match_id] = progress

            if info.status in TERMINAL_STATUSES:
                results[match.match_id] = info

        if len(results) == len(matches):
            return results
        if deadline is not None and time.monotonic() >= deadline:
            pending = [match.match_id for match in matches if match.match_id not in results]
            raise TimeoutError("Timed out waiting for: " + ", ".join(pending))

        print(f"Waiting {poll_seconds} seconds before the next check...", flush=True)
        time.sleep(poll_seconds)

    return results


def show_results(
    root: Path,
    cambc: str,
    matches: list[QueuedMatch],
    results: dict[str, MatchInfo],
    opponent_id: str,
    replay_dir: Path,
) -> bool:
    print("\n=== Results ===")
    all_complete = True

    for match in matches:
        info = results[match.match_id]
        print(f"\nMatch {match.number}: {match.match_id}")
        if info.status != "complete":
            all_complete = False
            print(f"  Match did not complete: {info.status}")
            continue

        side = rc_side(info, opponent_id)
        if side is None:
            all_complete = False
            print("  Could not distinguish the RC team from the opponent in cambc output.")
            print(info.raw_output.rstrip())
            continue

        if not info.games:
            all_complete = False
            print("  cambc did not report any games.")
            continue

        for game in sorted(info.games, key=lambda item: item.number):
            if game.winner.startswith(f"{side} ") or game.winner == side:
                print(f"  Game {game.number}, {game.map_name}: RC won ({game.condition}).")
            elif game.winner in {"--", ""}:
                all_complete = False
                print(f"  Game {game.number}, {game.map_name}: winner is not reported yet.")
            else:
                print(f"  Game {game.number}, {game.map_name}: enemy won ({game.condition}).")
                if download_replay(root, cambc, match, game, replay_dir) is None:
                    all_complete = False

    return all_complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit RC, run parallel unrated matches, and monitor them with cambc match info."
    )
    parser.add_argument("--opponent", default=OPPONENT_ID, help="Opponent team ID")
    parser.add_argument("--matches", type=int, default=3, help="Parallel match count (default: 3)")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Polling interval (default: 30)")
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=Path("replays/rc-unrated-losses"),
        help="Where to save replays for RC losses (default: replays/rc-unrated-losses)",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=float,
        default=0,
        help="Stop waiting after this many minutes; 0 waits indefinitely (default: 0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.matches < 1 or args.poll_seconds < 1 or args.timeout_minutes < 0:
        print("--matches and --poll-seconds must be positive; --timeout-minutes cannot be negative.", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parent
    if not (root / "bots" / "rc" / "main.py").is_file():
        print("RC bot was not found: bots/rc/main.py", file=sys.stderr)
        return 2

    try:
        cambc = find_cambc(root)
        print(f"$ {display_command([cambc, 'submit', './bots/rc'])}")
        submit_output = run_cambc(root, cambc, ["submit", "./bots/rc"])
        if submit_output:
            print(submit_output.rstrip())

        print(f"\nQueueing {args.matches} unrated matches in parallel...", flush=True)
        queued: list[QueuedMatch] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.matches) as executor:
            futures = [
                executor.submit(queue_match, root, cambc, args.opponent, number)
                for number in range(1, args.matches + 1)
            ]
            for future in concurrent.futures.as_completed(futures):
                match = future.result()
                queued.append(match)
                print(f"[{match.number}] queued: {match.match_id}", flush=True)

        queued.sort(key=lambda match: match.number)
        print("\nMonitoring with `cambc match info <match-id>`...", flush=True)
        results = monitor_matches(
            root, cambc, queued, args.poll_seconds, args.timeout_minutes
        )
        replay_dir = args.replay_dir if args.replay_dir.is_absolute() else root / args.replay_dir
        return 0 if show_results(
            root, cambc, queued, results, args.opponent, replay_dir
        ) else 1
    except (FileNotFoundError, RuntimeError, TimeoutError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. The match IDs printed above remain valid.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
