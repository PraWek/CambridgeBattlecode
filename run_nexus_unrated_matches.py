#!/usr/bin/env python3
"""Submit Nexus, run unrated matches, and save useful replay files."""

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


TERMINAL_STATUSES = {"complete", "error", "cancelled", "canceled", "failed"}
MATCH_ID_RE = re.compile(r"\bMatch ID:\s*([A-Za-z0-9_-]+)", re.IGNORECASE)
STATUS_RE = re.compile(r"^\s*Status:\s*(\w+)", re.IGNORECASE)
TEAM_RE = re.compile(r"^\s*Team ([AB]):\s*(.*?)\s*\(([^)]+)\)", re.IGNORECASE)
ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
UNSAFE_FILE_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


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


def find_cambc(root: Path) -> str:
    if installed := shutil.which("cambc"):
        return installed
    executable = "cambc.exe" if os.name == "nt" else "cambc"
    scripts = "Scripts" if os.name == "nt" else "bin"
    for environment in (".venv", ".venv-3.13", ".venv-3.12"):
        candidate = root / environment / scripts / executable
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("cambc was not found")


def run_cambc(root: Path, executable: str, *arguments: str) -> str:
    environment = dict(os.environ)
    environment["NO_COLOR"] = "1"
    completed = subprocess.run(
        [executable, *arguments],
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
        command = subprocess.list2cmdline([executable, *arguments])
        raise RuntimeError(f"{command} exited with {completed.returncode}\n{completed.stdout}")
    return completed.stdout


def parse_game_row(line: str) -> Game | None:
    separator = "│" if "│" in line else "|" if "|" in line else None
    if separator is None:
        return None
    cells = [cell.strip() for cell in line.split(separator) if cell.strip()]
    if len(cells) != 5 or not cells[0].isdigit():
        return None
    return Game(int(cells[0]), cells[1], cells[2], cells[3])


def parse_match_info(output: str) -> MatchInfo:
    status = "unknown"
    teams: dict[str, str] = {}
    games: list[Game] = []
    for line in ANSI_RE.sub("", output).splitlines():
        if match := STATUS_RE.match(line):
            status = match.group(1).lower()
        if match := TEAM_RE.match(line):
            teams[match.group(1).upper()] = match.group(3)
        if game := parse_game_row(line):
            games.append(game)
    return MatchInfo(status, teams, games)


def nexus_side(info: MatchInfo, opponent_id: str) -> str | None:
    if info.teams.get("A") == opponent_id:
        return "B"
    if info.teams.get("B") == opponent_id:
        return "A"
    return None


def nexus_won(game: Game, side: str) -> bool:
    return game.winner == side or game.winner.startswith(f"{side} ")


def queue_match(root: Path, executable: str, opponent: str, number: int) -> tuple[int, str]:
    output = run_cambc(root, executable, "match", "unrated", opponent)
    match = MATCH_ID_RE.search(output)
    if match is None:
        raise RuntimeError(f"match {number}: cambc did not report a Match ID\n{output}")
    return number, match.group(1)


def safe_name(value: str) -> str:
    return UNSAFE_FILE_RE.sub("_", value).strip(" .") or "unknown_map"


def download_replay(
        root: Path,
        executable: str,
        match_id: str,
        game: Game,
        replay_dir: Path,
) -> Path:
    replay_dir.mkdir(parents=True, exist_ok=True)
    destination = replay_dir / (
        f"{match_id}_game_{game.number}_{safe_name(game.map_name)}.replay26"
    )
    run_cambc(
        root,
        executable,
        "match",
        "replay",
        match_id,
        "--game",
        str(game.number),
        "--output",
        str(destination),
    )
    if not destination.is_file():
        raise RuntimeError(f"cambc did not create {destination}")
    return destination.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", required=True, help="Opponent team ID")
    parser.add_argument("--matches", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-minutes", type=float, default=30)
    parser.add_argument("--no-submit", action="store_true", help="Use the active Nexus submission")
    parser.add_argument(
        "--download",
        choices=("losses", "all", "none"),
        default="losses",
        help="Which completed games to download",
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=Path("replays/nexus-unrated"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.matches < 1 or args.poll_seconds < 1 or args.timeout_minutes <= 0:
        print("match count, poll interval, and timeout must be positive", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parent
    if not (root / "bots" / "nexus" / "main.py").is_file():
        print("bots/nexus/main.py was not found", file=sys.stderr)
        return 2

    try:
        executable = find_cambc(root)
        if not args.no_submit:
            print("Submitting bots/nexus...", flush=True)
            print(run_cambc(root, executable, "submit", "./bots/nexus").rstrip())

        queued: list[tuple[int, str]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.matches) as pool:
            futures = [
                pool.submit(queue_match, root, executable, args.opponent, number)
                for number in range(1, args.matches + 1)
            ]
            for future in concurrent.futures.as_completed(futures):
                item = future.result()
                queued.append(item)
                print(f"[{item[0]}] queued: {item[1]}", flush=True)
        queued.sort()

        deadline = time.monotonic() + args.timeout_minutes * 60
        pending = {match_id for _, match_id in queued}
        results: dict[str, MatchInfo] = {}
        progress: dict[str, tuple[str, int]] = {}
        while pending:
            for match_id in tuple(pending):
                info = parse_match_info(
                    run_cambc(root, executable, "match", "info", match_id)
                )
                state = (info.status, len(info.games))
                if progress.get(match_id) != state:
                    print(f"{match_id}: {info.status}, {len(info.games)}/5 games", flush=True)
                    progress[match_id] = state
                if info.status in TERMINAL_STATUSES:
                    results[match_id] = info
                    pending.remove(match_id)
            if pending:
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for " + ", ".join(sorted(pending)))
                time.sleep(args.poll_seconds)

        replay_dir = args.replay_dir if args.replay_dir.is_absolute() else root / args.replay_dir
        failures = 0
        print("\n=== Nexus results ===")
        for number, match_id in queued:
            info = results[match_id]
            side = nexus_side(info, args.opponent)
            if info.status != "complete" or side is None or not info.games:
                failures += 1
                print(f"[{number}] {match_id}: incomplete or team side is unknown")
                continue
            wins = 0
            for game in sorted(info.games, key=lambda item: item.number):
                won = nexus_won(game, side)
                wins += int(won)
                result = "won" if won else "lost"
                print(f"[{number}] game {game.number}, {game.map_name}: {result} ({game.condition})")
                if args.download == "all" or (args.download == "losses" and not won):
                    saved = download_replay(root, executable, match_id, game, replay_dir)
                    print(f"    replay: {saved}")
            print(f"[{number}] score: {wins}-{len(info.games) - wins}")
        return 1 if failures else 0
    except (FileNotFoundError, RuntimeError, TimeoutError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped; queued match IDs remain valid.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
