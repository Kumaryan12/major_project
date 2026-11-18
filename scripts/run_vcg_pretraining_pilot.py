"""Run one authorized VCG pilot, its smoke, or the native source audit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from run_residual_context import acquire_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit', 'native', 'bridge'])
    parser.add_argument('--config', default='configs/vcg_pretraining_pilots_v1.yaml')
    parser.add_argument('--download', action='store_true', help='Fetch missing official native XYZ files during audit')
    parser.add_argument('--smoke', action='store_true', help='Fit and replay three arms on one inner training/query pair')
    parser.add_argument('--status', action='store_true', help='Print saved progress and local POSIX process command')
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    config = yaml.safe_load(Path(args.config).read_text())
    if args.stage == 'audit':
        from mi_localization.vcg_expansion_bank import audit

        with acquire_lock(Path(config['source_audit_dir']) / 'runner.lock'):
            print(json.dumps(audit(config, args.download), indent=2))
        return
    root = Path(config[f'{args.stage}_output_dir'])
    target = root / ('smoke' if args.smoke else 'complete')
    if args.status:
        progress = target / 'progress.json'
        if progress.is_file():
            saved = json.loads(progress.read_text())
            print(json.dumps(saved, indent=2))
            if os.name == 'posix':
                subprocess.run(['ps', '-p', str(saved['pid']), '-o', 'pid=,etime=,command='], check=False)
        else:
            print('No progress receipt for this scope.')
        print('Validated summary:', (target / 'summary.json').exists())
        return
    command = [sys.executable, '-m', 'mi_localization.vcg_pretraining_study',
               args.stage, '--config', args.config]
    if args.smoke:
        command.append('--smoke')
    with acquire_lock(root / 'runner.lock'):
        subprocess.run(command, check=True)
        subprocess.run([*command, '--verify'], check=True)


if __name__ == '__main__':
    main()
