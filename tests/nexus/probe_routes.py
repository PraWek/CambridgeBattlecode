"""Offline route diagnostics against a replay snapshot; never runs game actions."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bots' / 'nexus'))
import analyze_nexus_replay as replay
import builder_bot
from cambc import Direction, EntityType, Environment, Team


def snapshot(path, limit=2000):
    data = replay.grouped(Path(path).read_bytes())
    map_data = replay.message(data, 1)
    rows = [replay.packed_integers(row, 1) for row in replay.messages(map_data, 3)]
    entities = {}
    for turn in replay.messages(data, 3)[:limit]:
        for update in replay.messages(turn, 1):
            if 1 in update:
                entity = replay.decode_entity(replay.message(update, 1)[1][-1][1])
                entities[entity.entity_id] = entity
            elif 2 in update:
                movement = replay.message(update, 2)
                entity = entities.get(replay.integer(movement, 1))
                if entity is not None:
                    entity.position = replay.position(movement, 2)
            elif 3 in update:
                entities.pop(replay.integer(replay.message(update, 3), 1), None)
    return map_data, rows, entities


def make_bot(path, limit=2000):
    map_data, rows, entities = snapshot(path, limit)
    bot = builder_bot.BuilderBot(len(rows[0]), len(rows))
    bot.team = Team.A
    bot.work_direction = Direction.EAST
    bot.current_round = limit
    bot.core_pos = bot.tile_cache.position_at(*replay.position(replay.messages(map_data, 4)[0]))
    envs = [Environment.EMPTY, Environment.WALL, Environment.ORE_TITANIUM, Environment.ORE_AXIONITE]
    for y, row in enumerate(rows):
        for x, env in enumerate(row):
            pos = bot.tile_cache.position_at(x, y)
            bot.known_env[pos] = envs[env]
            bot.observed_tiles.add(pos)
    for entity in entities.values():
        if entity.kind == 'builder':
            continue
        pos = bot.tile_cache.position_at(*entity.position)
        kind = EntityType(entity.kind)
        direction = None
        if entity.direction:
            dx, dy = replay.DIRECTION_DELTA[entity.direction]
            direction = next(d for d in Direction if d.delta() == (dx, dy))
        bot.tile_cache.remember_building(pos, entity.entity_id, kind, Team.A if entity.team == 0 else Team.B, direction=direction)
        if entity.bridge_target:
            bot.known_bridge_targets[pos] = bot.tile_cache.position_at(*entity.bridge_target)
    return bot


class Costs:
    def get_conveyor_cost(self):
        return (3, 0)

    def get_bridge_cost(self):
        return (20, 0)


if __name__ == '__main__':
    bot = make_bot(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'replay.replay26')
    for xy in [(6,18),(6,19),(13,0),(13,1),(13,4),(13,5),(13,9),(18,0),(19,0),(19,4)]:
        ore = bot.tile_cache.position_at(*xy)
        result = bot.connection_plan(Costs(), ore)
        print(xy, bot.known_env[ore], 'harvester', bot.is_harvester_on_tile(ore), 'plan', None if result is None else (len(result[1]), len(result[3])))
        if result:
            for source, target in result[3].items():
                path = builder_bot.a_star_to_any(None, source, {target}, bot.traversable_for_planning, bot.tile_cache.neighbor, movement_directions=builder_bot.DIRECTIONS, max_expansions=96)
                print('  bridge', (source.x,source.y), (target.x,target.y), 'walk', len(path))
