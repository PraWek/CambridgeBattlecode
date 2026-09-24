"""Run reproducible local matches against SOTA, on both sides of each map."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', default='bots/final')
    parser.add_argument('--maps', nargs='+', default=[
        'arena', 'default_small1', 'default_small2', 'default_medium1',
        'default_medium2', 'bear_of_doom', 'the_great_divide', 'rush_bait',
    ])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1])
    parser.add_argument('--output', default='tests/final/artifacts/sota')
    parser.add_argument('--tle', type=int, default=2)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--side', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        import cambc
        from cambc.cambc_engine import run_game
        sides = [str((ROOT / args.candidate / 'main.py').resolve()),
                 str(ROOT / 'bots/sota/main.py')]
        if args.side:
            sides.reverse()
        name = f'{args.maps[0]}_s{args.seeds[0]}_{args.side}'
        result = run_game(*sides, str(Path(cambc.__file__).resolve().parent.parent),
                          str(ROOT / 'maps' / (args.maps[0] + '.map26')),
                          str(output / (name + '.replay26')), args.seeds[0], args.tle)
        result.update(map=args.maps[0], seed=args.seeds[0], side=args.side,
                      candidate=args.candidate, tle_ms=args.tle)
        result['won'] = result['winner'] == ('B' if args.side else 'A')
        (output / (name + '.json')).write_text(json.dumps(result, indent=2), encoding='utf8')
        # The engine leaves subinterpreters alive (same exit as cambc run).
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    results = []
    for map_name in args.maps:
        for seed in args.seeds:
            for side in (0, 1):
                name = f'{map_name}_s{seed}_{side}'
                command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                           '--candidate', args.candidate, '--maps', map_name,
                           '--seeds', str(seed), '--side', str(side), '--tle', str(args.tle),
                           '--output', str(output)]
                with (output / (name + '.log')).open('w', encoding='utf8') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = json.loads((output / (name + '.json')).read_text(encoding='utf8'))
                results.append(result)
                print(f"{name}: {'WIN' if result['won'] else 'LOSS'} "
                      f"{result['win_condition']} r{result['turns']}", flush=True)
                (output / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    print(f"Wins: {sum(r['won'] for r in results)}/{len(results)}", flush=True)


if __name__ == '__main__':
    main()
