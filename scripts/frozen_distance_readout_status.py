"""Read-only, local command-checked distance-study progress."""
import argparse
import json
from pathlib import Path
import time

from supervised_source_status import command_for_pid


def snapshot(root):
    def read(path):
        return json.loads(path.read_text()) if path.exists() else {}
    progress, launcher = read(root / 'progress.json'), read(root / 'review/launcher.json')
    command = command_for_pid(progress.get('pid'))
    worker = bool(command and '-m mi_localization.frozen_distance_readout' in command)
    runner = command_for_pid(launcher.get('pid'))
    runner_alive = bool(runner and 'scripts/run_frozen_distance_readout.py' in runner)
    phase = ('verification' if '--verify' in command else 'fitting') if worker else (
        launcher.get('state', 'unknown') if runner_alive else 'no matching local worker')
    summary = read(root / 'summary.json')
    complete = (summary.get('complete') is True and summary.get('smoke') is False
                and summary.get('distance_heads') == 420 and summary.get('feature_contexts') == 120
                and summary.get('source_ridge_heads') == 210 and summary.get('identity_controls', 0) >= 90)
    lines = [f'Frozen64 distance readouts: {root}', f'Current local phase: {phase}',
             f'Worker command matches: {worker}; launcher command matches: {runner_alive}',
             f"Last worker update UTC: {progress.get('updated_utc', 'unavailable')}",
             f"Saved feature contexts: {len(list((root / 'review').glob('features_*.json')))}/120",
             f"Saved new distance heads: {len(list((root / 'review').glob('head_*.json')))}/420",
             f"Latest fold: {progress.get('fold', '—')}/10; seed: {progress.get('seed', '—')}; inner: {progress.get('inner', '—')}",
             f'Full verified summary/report present: {complete and (root / "REPORT.md").exists()}',
             'Saved counts measure fitting artifacts, not verified completion or final macro F1.',
             'Fixed 64-feature encoders; three seed ensembles; primary 145 earliest-record patients.']
    if launcher.get('state') == 'stopped_with_error':
        lines.append(f"Launcher stopped: {launcher.get('error_type')}: {launcher.get('error')}")
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='results/frozen_distance_readout_v1/complete')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    try:
        while True:
            if args.watch:
                print('\033[2J\033[H', end='')
            print(snapshot(Path(args.output_dir)), flush=True)
            if not args.watch:
                break
            time.sleep(10)
    except KeyboardInterrupt:
        pass
