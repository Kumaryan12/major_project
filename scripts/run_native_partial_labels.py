"""Run a matched native partial-label comparison and independently restore its fits."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml

from run_residual_context import acquire_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/native_partial_labels_v1.yaml')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    config = yaml.safe_load(Path(args.config).read_text())
    command = [sys.executable, '-m', 'mi_localization.native_partial_labels', '--config', args.config]
    if args.audit:
        command.append('--audit')
    elif args.smoke:
        command.append('--smoke')
    with acquire_lock(Path(config['output_dir']) / 'runner.lock'):
        subprocess.run(command, check=True)
        if not args.audit:
            subprocess.run([*command, '--verify'], check=True)


if __name__ == '__main__':
    main()
