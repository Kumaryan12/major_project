"""Run the bounded score-correction study and independently replay its outputs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml

from run_residual_context import acquire_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/distribution_score_correction_v1.yaml')
    parser.add_argument('--smoke', action='store_true', help='Use one inner partition only')
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    config = yaml.safe_load(Path(args.config).read_text())
    with acquire_lock(Path(config['output_dir']) / 'runner.lock'):
        command = [sys.executable, '-m', 'mi_localization.distribution_score_correction']
        if args.smoke:
            command.append('--smoke')
        subprocess.run([*command, '--config', args.config], check=True)
        if not args.smoke:
            subprocess.run([sys.executable, '-m', 'mi_localization.distribution_score_correction_report',
                            '--config', args.config], check=True)


if __name__ == '__main__':
    main()
