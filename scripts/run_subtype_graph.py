"""Fit and independently replay a bounded subtype graph study."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml

from run_residual_context import acquire_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/subtype_graph_v1.yaml')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    config = yaml.safe_load(Path(args.config).read_text())
    command = [sys.executable, '-m', 'mi_localization.subtype_graph', '--config', args.config]
    if args.smoke:
        command.append('--smoke')
    with acquire_lock(Path(config['output_dir']) / 'runner.lock'):
        subprocess.run(command, check=True)
        subprocess.run([*command, '--verify'], check=True)


if __name__ == '__main__':
    main()
