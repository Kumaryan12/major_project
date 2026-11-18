"""Run the local timing feasibility audit and independently replay the results."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml

from run_residual_context import acquire_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/local_timing_audit_v1.yaml')
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    config = yaml.safe_load(Path(args.config).read_text())
    command = [sys.executable, '-m', 'mi_localization.local_timing_audit', '--config', args.config]
    with acquire_lock(Path(config['output_dir']) / 'runner.lock'):
        subprocess.run(command, check=True)
        subprocess.run([*command, '--verify'], check=True)


if __name__ == '__main__':
    main()
