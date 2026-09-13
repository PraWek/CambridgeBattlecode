"""Extract replay maps and compare current Nexus with its committed baseline.

Usage: python tests/nexus/benchmark_replays.py --games 3 4
Artifacts stay under tests/nexus/artifacts; RC and original replays are read-only.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import shutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import analyze_nexus_replay as replay


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=True)
    data = replay.grouped(source.read_bytes())
    map_path = output / (source.stem + '.map26')
    map_path.write_bytes(data[1][0][1])
    return map_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', nargs='+', type=int, default=[3])
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'test_replays')
    parser.add_argument('--candidate', default='nexus')
    parser.add_argument('--baseline-dir', type=Path, help='Frozen opponent; defaults to committed Nexus')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'tests' / 'nexus' / 'artifacts')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--swap', action='store_true', help='Put the candidate on team B')
    parser.add_argument('--tle', type=int, default=2)
    parser.add_argument('--match', default='4d5708f7-96d4-495b-8b97-a3b7af223fc8')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline_dir.resolve() if args.baseline_dir else output / 'baseline'
    if args.baseline_dir is None:
        baseline.mkdir(parents=True, exist_ok=True)
        git = ['git', '-c', f'safe.directory={ROOT.as_posix()}']
        names = subprocess.check_output(git + ['ls-tree', '-r', '--name-only', 'HEAD', 'bots/nexus'], cwd=ROOT, text=True)
        for name in names.splitlines():
            if name.endswith('.py'):
                (baseline / Path(name).name).write_bytes(subprocess.check_output(git + ['show', f'HEAD:{name}'], cwd=ROOT))
    cli = ROOT / '.venv' / 'Scripts' / 'cambc.exe'
    candidate = args.candidate
    if args.trace:
        trace = output / 'trace'
        trace.mkdir(exist_ok=True)
        for source in (ROOT / 'bots' / 'nexus').glob('*.py'):
            shutil.copy2(source, trace / source.name)
        main_file = trace / 'main.py'
        code = main_file.read_text()
        code = code.replace('self.bot.run(c)', 'self.bot.run(c)\n            if c.get_entity_type() == EntityType.BUILDER_BOT and self.bot.current_position is not None:\n                c.draw_indicator_dot(self.bot.current_position, min(255, self.bot.path_index), min(255, len(self.bot.path)), int(self.bot.target_is_connection))\n                if self.bot.target_ore is not None:\n                    c.draw_indicator_line(self.bot.current_position, self.bot.target_ore, int(self.bot.harvester_is_connected(self.bot.target_ore)), int(c.can_build_harvester(self.bot.target_ore)), int(self.bot.replan_after_yield))')
        main_file.write_text(code)
        # Trace mode must never rewrite the supplied baseline snapshot.
        traced_baseline = output / 'trace_baseline'
        shutil.copytree(baseline, traced_baseline, dirs_exist_ok=True)
        baseline = traced_baseline
        baseline_main = baseline / 'main.py'
        code = baseline_main.read_text().replace('self.init_once(c)', 'if c.get_entity_type() == EntityType.CORE and c.get_current_round() > 250:\n            c.resign()\n            return\n        self.init_once(c)')
        baseline_main.write_text(code)
        candidate = str(trace)
    for game in args.games:
        source = args.source_dir / f'{args.match}_game_{game}.replay26'
        map_path = prepare(source, output)
        if args.prepare_only:
            print(map_path, flush=True)
            continue
        destination = output / f'game_{game}_fixed.replay26'
        sides = [candidate, str(baseline)]
        if args.swap:
            sides.reverse()
        command = [str(cli), 'run', *sides, str(map_path), '--seed', str(args.seed), '--tle', str(args.tle), '--replay', str(destination)]
        print(f'game {game}: {sides[0]} vs {sides[1]}', flush=True)
        with (output / f'game_{game}.log').open('w', encoding='utf8') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = replay.analyze(destination)
        report['benchmark'] = {'candidate_team': int(args.swap), 'seed': args.seed,
                               'tle_ms': args.tle, 'command': command}
        for team in report['teams']:
            print(json.dumps({key: team[key] for key in ['team', 'titanium_collected', 'coverage_percent', 'ores_seen', 'placed', 'network']}, ensure_ascii=False), flush=True)
        (output / f'game_{game}.json').write_text(json.dumps(report), encoding='utf8')


if __name__ == '__main__':
    main()
