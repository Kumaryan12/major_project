"""Fit, independently replay and optionally publish one distance study, then stop."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from run_supervised_source_pilot import exclusive_lock


def record(root, state, **fields):
    path = root / 'review/launcher.json'
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(dict(pid=os.getpid(), state=state,
                                         updated_utc=datetime.now(timezone.utc).isoformat(), **fields), indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/frozen_distance_readout_v1.yaml')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config['output_dir']) / 'complete'
    with exclusive_lock(root / 'review/launcher.lock'):
        try:
            base = [sys.executable, '-u', '-m', 'mi_localization.frozen_distance_readout', '--config', args.config]
            record(root, 'fitting')
            subprocess.run(base, check=True)
            record(root, 'verifying')
            subprocess.run(base + ['--verify'], check=True)
            result = json.loads((root / 'summary.json').read_text())
            if (not result['complete'] or result['smoke'] or result['distance_heads'] != 420
                    or result['source_ridge_heads'] != 210 or result['feature_contexts'] != 120
                    or result['identity_controls'] < 90 or result['encoder_updates'] != 0):
                raise ValueError('Declared verification inventory is incomplete')
            if args.publish:
                record(root, 'publishing')
                branch = subprocess.run(['git', 'branch', '--show-current'], check=True, capture_output=True, text=True).stdout.strip()
                if branch != 'main':
                    raise RuntimeError('Publication expects main; completed results retained locally')
                paths = [str(root / name) for name in ('study_manifest.json', 'summary.json', 'REPORT.md')]
                subprocess.run(['git', 'add', '--', *paths], check=True)
                diff = subprocess.run(['git', 'diff', '--cached', '--quiet', '--', *paths])
                if diff.returncode == 1:
                    subprocess.run(['git', 'commit', '--only', '-m', 'Publish verified frozen distance readout results', '--', *paths], check=True)
                elif diff.returncode != 0:
                    raise RuntimeError('Cannot inspect publication changes')
                subprocess.run(['git', 'push', 'origin', 'HEAD:main'], check=True)
            record(root, 'complete_validated', next_experiment_queued=False)
            print('Distance comparison verified and stopped. No next experiment launched.', flush=True)
        except Exception as error:
            record(root, 'stopped_with_error', error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    main()
