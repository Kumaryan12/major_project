"""Read-only local progress for the frozen patient readout comparison."""
import argparse
import json
from pathlib import Path
import time

from supervised_source_status import command_for_pid


def snapshot(root):
    progress = []
    for path in (root / 'progress.json', root / 'review/progress.json'):
        if path.exists():
            progress.append(json.loads(path.read_text()))
    latest = max(progress, key=lambda p: p.get('updated_utc', ''), default={})
    command = command_for_pid(latest.get('pid'))
    alive = bool(command and '-m mi_localization.frozen_patient_readout' in command)
    phase = 'verification' if alive and '--verify' in command else 'fitting' if alive else 'no matching local worker'
    review = root / 'review'
    fits = len(list(review.glob('inner_*.json'))) + len(list(review.glob('outer_*.json')))
    contexts = len(list(review.glob('features_*.json')))
    summary_path = root / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    complete = (summary.get('complete') is True and summary.get('smoke') is False
                and summary.get('ridge_fits') == 420 and summary.get('encoder_contexts') == 120)
    return '\n'.join([
        f'Frozen patient readouts: {root}',
        f'Current local phase: {phase}',
        f"Saved state: {latest.get('state', 'not started')} | updated UTC: {latest.get('updated_utc', 'unavailable')}",
        f'Saved encoder feature contexts: {contexts}/120',
        f'Saved ridge classifier audits: {fits}/420',
        f'Full verified summary and report saved: {complete and (root / "REPORT.md").exists()}',
        'Saved fitting counts do not measure verification progress or final macro F1.',
        '120 frozen encoders; no neural weight updates. Primary: 145 dated earliest-record patients.',
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='results/frozen_patient_readout_v1/complete')
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
