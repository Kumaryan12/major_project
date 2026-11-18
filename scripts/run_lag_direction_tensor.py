"""Fit the within-beat directional tensor and replay it independently."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml

from run_residual_context import acquire_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/lag_direction_tensor_v1.yaml')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    config = yaml.safe_load(Path(args.config).read_text())
    command = [sys.executable, '-m', 'mi_localization.lag_direction_study', '--config', args.config]
    if args.smoke:
        command.append('--smoke')
    with acquire_lock(Path(config['output_dir']) / 'runner.lock'):
        if not args.smoke:
            root = Path(config['output_dir'])
            report = root / 'complete' / 'summary.json'
            if report.exists():
                print('A completed report is already present. Read it before starting another run.')
                return
        subprocess.run(command, check=True)
        subprocess.run([*command, '--verify'], check=True)


if __name__ == '__main__':
    main()
