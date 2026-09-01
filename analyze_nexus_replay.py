#!/usr/bin/env python3
"""Extract exploration and economy metrics from a Cambridge Battlecode replay."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterator

from cambc import GameConstants


ENTITY_KINDS = {
    10: "builder",
    11: "conveyor",
    12: "splitter",
    13: "armoured_conveyor",
    14: "bridge",
    15: "harvester",
    16: "foundry",
    17: "road",
    18: "barrier",
    19: "marker",
    20: "core",
    21: "gunner",
    22: "sentinel",
    23: "breach",
    24: "launcher",
}
DIRECTION_DELTA = {
    1: (0, -1),
    2: (1, -1),
    3: (1, 0),
    4: (1, 1),
    5: (0, 1),
    6: (-1, 1),
    7: (-1, 0),
    8: (-1, -1),
}
TRANSPORT_KINDS = {"conveyor", "splitter", "armoured_conveyor", "bridge"}


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift > 70:
            raise ValueError("invalid protobuf varint")


def fields(data: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    offset = 0
    while offset < len(data):
        tag, offset = read_varint(data, offset)
        number = tag >> 3
        wire = tag & 7
        if wire == 0:
            value, offset = read_varint(data, offset)
        elif wire == 1:
            value = data[offset:offset + 8]
            offset += 8
        elif wire == 2:
            size, offset = read_varint(data, offset)
            value = data[offset:offset + size]
            offset += size
        elif wire == 5:
            value = data[offset:offset + 4]
            offset += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def grouped(data: bytes) -> dict[int, list[tuple[int, int | bytes]]]:
    result: dict[int, list[tuple[int, int | bytes]]] = defaultdict(list)
    for number, wire, value in fields(data):
        result[number].append((wire, value))
    return result


def integer(message: dict, number: int, default: int = 0) -> int:
    values = message.get(number)
    if not values:
        return default
    value = values[-1][1]
    return int(value) if isinstance(value, int) else default


def messages(message: dict, number: int) -> list[dict]:
    return [
        grouped(value)
        for wire, value in message.get(number, ())
        if wire == 2 and isinstance(value, bytes)
    ]


def message(message_data: dict, number: int) -> dict:
    nested = messages(message_data, number)
    return nested[-1] if nested else defaultdict(list)


def position(message_data: dict, number: int = 3) -> tuple[int, int]:
    value = message(message_data, number)
    return integer(value, 1), integer(value, 2)


def packed_integers(message_data: dict, number: int) -> list[int]:
    result: list[int] = []
    for wire, value in message_data.get(number, ()):
        if wire == 0 and isinstance(value, int):
            result.append(value)
        elif wire == 2 and isinstance(value, bytes):
            offset = 0
            while offset < len(value):
                item, offset = read_varint(value, offset)
                result.append(item)
    return result


@dataclass
class Entity:
    entity_id: int
    team: int
    position: tuple[int, int]
    kind: str
    direction: int = 0
    bridge_target: tuple[int, int] | None = None


@dataclass
class BuilderStats:
    entity_id: int
    team: int
    born: int
    last_position: tuple[int, int]
    last_move_turn: int
    moves: int = 0
    current_stand: int = 0
    longest_stand: int = 0
    tle: int = 0
    unique_positions: set[tuple[int, int]] = field(default_factory=set)
    position_two_turns_ago: tuple[int, int] | None = None
    alternating_streak: int = 0
    longest_alternating: int = 0
    moves_after_1000: int = 0
    unique_after_1000: set[tuple[int, int]] = field(default_factory=set)

    def observe_turn(self, turn: int, current: tuple[int, int]) -> None:
        previous = self.last_position
        if (
            self.position_two_turns_ago is not None
            and current == self.position_two_turns_ago
            and current != previous
        ):
            self.alternating_streak += 1
            self.longest_alternating = max(
                self.longest_alternating,
                self.alternating_streak,
            )
        else:
            self.alternating_streak = 0
        self.position_two_turns_ago = previous
        if current == self.last_position:
            self.current_stand += 1
            self.longest_stand = max(self.longest_stand, self.current_stand)
            return
        self.moves += 1
        if turn >= 1000:
            self.moves_after_1000 += 1
            self.unique_after_1000.add(current)
        self.current_stand = 0
        self.last_position = current
        self.last_move_turn = turn
        self.unique_positions.add(current)


def decode_entity(data: bytes) -> Entity:
    entity = grouped(data)
    kind_field = next((number for number in ENTITY_KINDS if number in entity), 17)
    kind = ENTITY_KINDS[kind_field]
    details = message(entity, kind_field)
    return Entity(
        entity_id=integer(entity, 1),
        team=integer(entity, 2),
        position=position(entity),
        kind=kind,
        direction=integer(details, 1),
        bridge_target=position(details, 1) if kind == "bridge" else None,
    )


def vision_tiles(
        centre: tuple[int, int], radius_sq: int, width: int, height: int,
) -> Iterator[tuple[int, int]]:
    radius = int(radius_sq ** 0.5)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius_sq:
                continue
            x, y = centre[0] + dx, centre[1] + dy
            if 0 <= x < width and 0 <= y < height:
                yield x, y


def reachable_builder_vision(
        start: tuple[int, int],
        environments: list[list[int]],
        radius_sq: int,
) -> set[tuple[int, int]]:
    """Return the theoretical vision ceiling of the core's walkable component."""
    height = len(environments)
    width = len(environments[0]) if environments else 0
    reachable = {start}
    queue = [start]
    index = 0
    while index < len(queue):
        x, y = queue[index]
        index += 1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidate = x + dx, y + dy
                if (
                    dx == 0 and dy == 0
                    or candidate in reachable
                    or not (0 <= candidate[0] < width and 0 <= candidate[1] < height)
                    or environments[candidate[1]][candidate[0]] in {1, 2, 3}
                ):
                    continue
                reachable.add(candidate)
                queue.append(candidate)
    visible: set[tuple[int, int]] = set()
    for tile in reachable:
        visible.update(vision_tiles(tile, radius_sq, width, height))
    return visible


def transport_receiver(entity: Entity) -> tuple[int, int] | None:
    if entity.kind == "bridge":
        return entity.bridge_target
    delta = DIRECTION_DELTA.get(entity.direction)
    if delta is None:
        return None
    return entity.position[0] + delta[0], entity.position[1] + delta[1]


def final_network_metrics(
        team: int,
        entities: dict[int, Entity],
        core_position: tuple[int, int],
) -> dict:
    transports = {
        entity.position: entity
        for entity in entities.values()
        if entity.team == team and entity.kind in TRANSPORT_KINDS
    }
    harvesters = [
        entity
        for entity in entities.values()
        if entity.team == team and entity.kind == "harvester"
    ]
    core_tiles = {
        (core_position[0] + dx, core_position[1] + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    }
    loads: Counter[tuple[int, int]] = Counter()
    entries: Counter[tuple[int, int]] = Counter()
    flow_options: dict[tuple[int, int], set[tuple[int, int]]] = {}
    connected = 0
    disconnected: list[tuple[int, int]] = []
    harvester_positions = {harvester.position for harvester in harvesters}
    for harvester in harvesters:
        paths: list[list[tuple[int, int]]] = []
        x, y = harvester.position
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            first = transports.get((x + dx, y + dy))
            if first is None or transport_receiver(first) == harvester.position:
                continue
            path: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            current = first
            while current.position not in seen:
                seen.add(current.position)
                path.append(current.position)
                target = transport_receiver(current)
                if target in core_tiles:
                    paths.append(path)
                    break
                current = transports.get(target)
                if current is None:
                    break
        if not paths:
            disconnected.append(harvester.position)
            continue
        flow_options[harvester.position] = {path[-1] for path in paths}
        path = min(paths, key=len)
        connected += 1
        loads.update(path)
        entries[path[-1]] += 1

    # One harvester emits one stack every four rounds; a physical root lane
    # can move one stack per round.  Match sources to root lanes with capacity
    # four so the report distinguishes a visually dense tree from actual
    # sustainable throughput.
    root_members: dict[tuple[int, int], list[tuple[int, int]]] = {}
    flow_assignment: dict[tuple[int, int], tuple[int, int]] = {}

    def augment(
            harvester: tuple[int, int],
            seen_roots: set[tuple[int, int]],
            seen_harvesters: set[tuple[int, int]],
    ) -> bool:
        for root in sorted(
            flow_options.get(harvester, ()),
            key=lambda pos: (len(root_members.get(pos, ())), pos),
        ):
            if root in seen_roots:
                continue
            seen_roots.add(root)
            members = root_members.setdefault(root, [])
            if len(members) < 4:
                members.append(harvester)
                flow_assignment[harvester] = root
                return True
            for displaced in tuple(members):
                if displaced in seen_harvesters:
                    continue
                seen_harvesters.add(displaced)
                if not augment(displaced, seen_roots, seen_harvesters):
                    continue
                members.remove(displaced)
                members.append(harvester)
                flow_assignment[harvester] = root
                return True
        return False

    for harvester in sorted(flow_options):
        augment(harvester, set(), {harvester})

    reaches_core: set[tuple[int, int]] = set()
    for start in transports:
        current = transports[start]
        seen: set[tuple[int, int]] = set()
        while current.position not in seen:
            seen.add(current.position)
            target = transport_receiver(current)
            if target in core_tiles:
                reaches_core.update(seen)
                break
            current = transports.get(target)
            if current is None:
                break
    rejecting_guards = {
        position
        for position, transport in transports.items()
        if transport_receiver(transport) in harvester_positions
    }
    orphan_transports = sorted(
        set(transports) - reaches_core - rejecting_guards
    )
    dangling_bridges = sorted(
        position
        for position, transport in transports.items()
        if transport.kind == "bridge"
        and transport_receiver(transport) not in transports
        and transport_receiver(transport) not in core_tiles
    )
    return {
        "harvesters": len(harvesters),
        "connected_harvesters": connected,
        "max_flow_harvesters": len(flow_assignment),
        "capacity_blocked_harvesters": sorted(
            set(flow_options) - set(flow_assignment)
        ),
        "disconnected_harvesters": disconnected,
        "max_structural_load": max(loads.values(), default=0),
        "core_entry_loads": {
            f"{x},{y}": load for (x, y), load in sorted(entries.items())
        },
        "transport_tiles": len(transports),
        "core_reaching_transport_tiles": len(reaches_core),
        "rejecting_guard_tiles": len(rejecting_guards),
        "orphan_transport_tiles": orphan_transports,
        "dangling_bridge_tiles": dangling_bridges,
    }


def analyze(path: Path) -> dict:
    replay = grouped(path.read_bytes())
    map_data = message(replay, 1)
    width, height = integer(map_data, 1), integer(map_data, 2)
    rows = messages(map_data, 3)
    environments = [packed_integers(row, 1) for row in rows]
    ore_positions = {
        (x, y)
        for y, row in enumerate(environments)
        for x, environment in enumerate(row)
        if environment in {2, 3}
    }
    cores = [
        {
            "id": integer(core, 1),
            "team": integer(core, 2),
            "position": position(core),
        }
        for core in messages(map_data, 4)
    ]
    entities: dict[int, Entity] = {
        core["id"]: Entity(core["id"], core["team"], core["position"], "core")
        for core in cores
    }
    builders: dict[int, BuilderStats] = {}
    placed: list[Counter[str]] = [Counter(), Counter()]
    placement_turns: list[dict[str, list[int]]] = [defaultdict(list), defaultdict(list)]
    placement_tiles: list[Counter[tuple[str, tuple[int, int]]]] = [Counter(), Counter()]
    coverage: list[set[tuple[int, int]]] = [set(), set()]
    players = [
        {"titanium": 500, "titanium_collected": 0},
        {"titanium": 500, "titanium_collected": 0},
    ]

    for core in cores:
        coverage[core["team"]].update(vision_tiles(
            core["position"], GameConstants.CORE_VISION_RADIUS_SQ, width, height,
        ))

    turns = messages(replay, 3)
    for turn_number, turn in enumerate(turns, start=1):
        for update in messages(turn, 1):
            if 1 in update:
                placed_entity = decode_entity(
                    message(update, 1)[1][-1][1]  # PlaceEntity.entity
                )
                entities[placed_entity.entity_id] = placed_entity
                placed[placed_entity.team][placed_entity.kind] += 1
                placement_turns[placed_entity.team][placed_entity.kind].append(turn_number)
                placement_tiles[placed_entity.team][(
                    placed_entity.kind,
                    placed_entity.position,
                )] += 1
                if placed_entity.kind == "builder":
                    builders[placed_entity.entity_id] = BuilderStats(
                        placed_entity.entity_id,
                        placed_entity.team,
                        turn_number,
                        placed_entity.position,
                        turn_number,
                        unique_positions={placed_entity.position},
                    )
            elif 2 in update:
                movement = message(update, 2)
                entity = entities.get(integer(movement, 1))
                if entity is not None:
                    entity.position = position(movement, 2)
            elif 3 in update:
                removed = integer(message(update, 3), 1)
                entities.pop(removed, None)
            elif 6 in update:
                values = message(message(update, 6), 1)
                for team, player_field in enumerate((1, 2)):
                    player = message(values, player_field)
                    if player:
                        players[team] = {
                            "titanium": integer(player, 1),
                            "titanium_collected": integer(player, 4),
                        }
            elif 9 in update:
                output = message(update, 9)
                stats = builders.get(integer(output, 1))
                if stats is not None and integer(output, 4):
                    stats.tle += 1

        for entity in entities.values():
            if entity.kind != "builder":
                continue
            stats = builders.get(entity.entity_id)
            if stats is not None:
                stats.observe_turn(turn_number, entity.position)
            coverage[entity.team].update(vision_tiles(
                entity.position,
                GameConstants.BUILDER_BOT_VISION_RADIUS_SQ,
                width,
                height,
            ))

    result = {
        "file": str(path.resolve()),
        "map": {
            "width": width,
            "height": height,
            "turns": len(turns),
            "ore_tiles": len(ore_positions),
            "wall_tiles": sum(value == 1 for row in environments for value in row),
        },
        "teams": [],
    }
    for team in (0, 1):
        core = next(item for item in cores if item["team"] == team)
        reachable_vision = reachable_builder_vision(
            core["position"],
            environments,
            GameConstants.BUILDER_BOT_VISION_RADIUS_SQ,
        )
        result["teams"].append({
            "team": team,
            "titanium": players[team]["titanium"],
            "titanium_collected": players[team]["titanium_collected"],
            "coverage": len(coverage[team]),
            "coverage_percent": round(100 * len(coverage[team]) / (width * height), 1),
            "reachable_coverage": len(reachable_vision),
            "reachable_coverage_percent": round(
                100 * len(coverage[team] & reachable_vision) / len(reachable_vision),
                1,
            ),
            "ores_seen": len(coverage[team] & ore_positions),
            "reachable_ores": len(reachable_vision & ore_positions),
            "placed": dict(placed[team]),
            "harvester_turns": placement_turns[team].get("harvester", []),
            "builders": [
                {
                    "id": stats.entity_id,
                    "born": stats.born,
                    "moves": stats.moves,
                    "unique_tiles": len(stats.unique_positions),
                    "last_move_turn": stats.last_move_turn,
                    "longest_stand": stats.longest_stand,
                    "longest_two_tile_cycle": stats.longest_alternating,
                    "moves_after_1000": stats.moves_after_1000,
                    "unique_tiles_after_1000": len(stats.unique_after_1000),
                    "tiles_after_1000": sorted(stats.unique_after_1000),
                    "last_position": stats.last_position,
                    "tle": stats.tle,
                }
                for stats in builders.values()
                if stats.team == team
            ],
            "network": final_network_metrics(team, entities, core["position"]),
            "repeated_placements": [
                {
                    "kind": kind,
                    "position": pos,
                    "count": count,
                }
                for (kind, pos), count in placement_tiles[team].most_common(20)
                if count > 1
            ],
        })
    return result


def print_summary(result: dict) -> None:
    map_data = result["map"]
    print(
        f"{Path(result['file']).name}: {map_data['width']}x{map_data['height']}, "
        f"{map_data['turns']} turns, {map_data['ore_tiles']} ore"
    )
    for team in result["teams"]:
        builders = team["builders"]
        total_tle = sum(builder["tle"] for builder in builders)
        longest_stand = max(
            (builder["longest_stand"] for builder in builders),
            default=0,
        )
        longest_cycle = max(
            (builder["longest_two_tile_cycle"] for builder in builders),
            default=0,
        )
        network = team["network"]
        placed = team["placed"]
        disconnected = ";".join(
            f"{x},{y}" for x, y in network["disconnected_harvesters"]
        ) or "-"
        print(
            f"team {team['team']}: Ti={team['titanium_collected']}, "
            f"vision={team['coverage_percent']}% "
            f"({team['reachable_coverage_percent']}% reachable), "
            f"ore={team['ores_seen']}/{team['reachable_ores']} reachable "
            f"({map_data['ore_tiles']} total), "
            f"builders={placed.get('builder', 0)}, "
            f"harvesters={network['connected_harvesters']}/{network['harvesters']}, "
            f"flow={network['max_flow_harvesters']}/{network['harvesters']}, "
            f"conv={placed.get('conveyor', 0)}, bridges={placed.get('bridge', 0)}, "
            f"roads={placed.get('road', 0)}, load={network['max_structural_load']}, "
            f"TLE={total_tle}, max_stand={longest_stand}, disconnected={disconnected}"
        )
        print(
            f"        two_tile_cycle={longest_cycle}, "
            f"orphan_transport={len(network['orphan_transport_tiles'])}, "
            f"dangling_bridges={len(network['dangling_bridge_tiles'])}, "
            f"repeated_build_tiles={len(team['repeated_placements'])}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--json", action="store_true", help="Print the complete JSON report")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if not args.replay.is_file():
        parser.error(f"replay does not exist: {args.replay}")
    result = analyze(args.replay)
    if args.json:
        print(json.dumps(
            result,
            indent=None if args.compact else 2,
            ensure_ascii=False,
        ))
    else:
        print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
