"""Freeze both bots and run the five diagnostic maps on both starting sides."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', default='bots/final')
    parser.add_argument('--maps', nargs='+', default=[
        'arena', 'default_small1', 'default_medium1', 'minimaze', 'bear_of_doom',
    ])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1])
    parser.add_argument('--output', default='tests/final/artifacts/sota')
    parser.add_argument('--tle', type=int, default=2)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--opponent', default='bots/sota')
    parser.add_argument('--economy-only', action='store_true',
                        help='Disable candidate intruders in the frozen snapshot for passive diagnostics')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--side', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        import cambc
        from cambc.cambc_engine import run_game
        sides = [str((ROOT / args.candidate / 'main.py').resolve()),
                 str((ROOT / args.opponent / 'main.py').resolve())]
        if args.side:
            sides.reverse()
        name = f'{args.maps[0]}_s{args.seeds[0]}_{args.side}'
        result = run_game(*sides, str(Path(cambc.__file__).resolve().parent.parent),
                          str(ROOT / 'maps' / (args.maps[0] + '.map26')),
                          str(output / (name + '.replay26')), args.seeds[0], args.tle)
        result.update(map=args.maps[0], seed=args.seeds[0], side=args.side,
                      candidate=args.candidate, opponent=args.opponent, tle_ms=args.tle)
        result['won'] = result['winner'] == ('B' if args.side else 'A')
        from replay_health import health
        result['runtime'] = health(output / (name + '.replay26'))
        (output / (name + '.json')).write_text(json.dumps(result, indent=2), encoding='utf8')
        # The engine leaves subinterpreters alive (same exit as cambc run).
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    # Every match in the batch sees exactly this source, even while the next
    # candidate is being edited. Only Python submission files enter the copy.
    snapshots = []
    manifest = {'tle_ms': args.tle, 'bots': []}
    for label, source in (('candidate', args.candidate), ('opponent', args.opponent)):
        source = (ROOT / source).resolve()
        if not (source / 'main.py').is_file():
            parser.error(f'bot entry point not found: {source / "main.py"}')
        snapshot = Path(tempfile.mkdtemp(prefix=label + '_', dir=output))
        hashes = {}
        for path in sorted(source.rglob('*.py')):
            relative = path.relative_to(source)
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
        snapshots.append(snapshot)
        if label == 'candidate' and args.economy_only:
            constants = snapshot / 'constants.py'
            code = constants.read_text(encoding='utf8')
            import re
            code, count = re.subn(r'^MAX_INTRUDER_SPAWNS = \d+', 'MAX_INTRUDER_SPAWNS = 0', code, flags=re.M)
            if count != 1:
                parser.error('economy diagnostic requires the original final role constants')
            constants.write_text(code, encoding='utf8')
            hashes['constants.py'] = hashlib.sha256(constants.read_bytes()).hexdigest()
            manifest['economy_only'] = True
        manifest['bots'].append({'role': label, 'source': str(source),
                                 'snapshot': str(snapshot), 'sha256': hashes})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    def match(map_name, seed, side):
        name = f'{map_name}_s{seed}_{side}'
        command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                   '--candidate', str(snapshots[0]), '--opponent', str(snapshots[1]),
                   '--maps', map_name, '--seeds', str(seed), '--side', str(side),
                   '--tle', str(args.tle), '--output', str(output)]
        with (output / (name + '.log')).open('w', encoding='utf8') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        return name, json.loads((output / (name + '.json')).read_text(encoding='utf8'))

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        matches = [pool.submit(match, m, seed, side)
                   for m in args.maps for seed in args.seeds for side in (0, 1)]
        for future in as_completed(matches):
            name, result = future.result()
            results.append(result)
            print(f"{name}: {'WIN' if result['won'] else 'LOSS'} "
                  f"{result['win_condition']} r{result['turns']}", flush=True)
            results.sort(key=lambda r: (r['map'], r['seed'], r['side']))
            (output / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    print(f"Wins: {sum(r['won'] for r in results)}/{len(results)}", flush=True)


if __name__ == '__main__':
    main()
