"""Run the ACS-ECG inner smoke, verified source preparation, training and transfer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from run_residual_context import acquire_lock

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('acs_ecg_bank', 'acs_artery_pretraining', 'acs_artery_transfer')
PUBLIC = ('REPORT.md', 'summary.json', 'source_summary.json', 'source_manifest.json',
          'study_manifest.json', 'verification_manifest.json')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pipeline(config_path, smoke):
    mode = ['--smoke'] if smoke else []
    for name in MODULES:
        command = [sys.executable, '-m', f'mi_localization.{name}', '--config', config_path, *mode]
        phase = 'inner smoke' if smoke else 'full'
        print(f'{phase}: {name}', flush=True)
        subprocess.run(command, check=True)
        print(f'{phase}: verify {name}', flush=True)
        subprocess.run([*command, '--verify'], check=True)


def verified_report(root):
    summary = read(root / 'summary.json')
    scope = read(root / 'study_manifest.json')
    source = read(root / 'source_summary.json')
    evidence = read(root / 'verification_manifest.json')
    if (summary['complete'] is not True or summary['outer_test_patients_used'] is not True
            or summary['restored_target_models'] != 90 or summary['matched_fold_seed_groups'] != 30
            or len(evidence['fit_audit_hashes']) != 90
            or summary['study_fingerprint'] != scope['fingerprint']
            or evidence['fingerprint'] != scope['fingerprint']
            or summary['verification_manifest_sha256'] != sha(root / 'verification_manifest.json')
            or summary['report_sha256'] != sha(root / 'REPORT.md')
            or scope['source_summary_sha256'] != sha(root / 'source_summary.json')
            or source['source_manifest_sha256'] != sha(root / 'source_manifest.json')
            or source['source_bank']['recording_copy_screen']['passed'] is not True
            or source['complete'] is not True or source['models'] != 2 or source['matched_sampling'] is not True):
        raise ValueError('Require independently verified source and target coverage for publication')
    return [root / name for name in PUBLIC], summary


def publish(files):
    if subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() != 'main':
        raise RuntimeError('Results verified locally. Publication requires the main branch.')
    if subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip():
        raise RuntimeError('Results verified locally. Git index contains other work.')
    subprocess.run(['git', 'add', '--', *map(str, files)], check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet']).returncode == 1:
        subprocess.run(['git', 'commit', '-m', 'Publish verified ACS-ECG artery transfer comparison'], check=True)
    subprocess.run(['git', 'push', 'origin', 'HEAD:main'], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/acs_artery_transfer_v1.yaml')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    config = yaml.safe_load(Path(args.config).read_text())
    base = Path(config['output_dir'])
    with acquire_lock(base / 'runner.lock'):
        pipeline(args.config, smoke=True)
        smoke = read(base / 'smoke' / 'summary.json')
        if smoke['restored_target_models'] != 3 or smoke['outer_test_patients_used'] is not False:
            raise ValueError('Require all three inner target smoke arms')
        if args.smoke:
            return
        pipeline(args.config, smoke=False)
        files, result = verified_report(base / 'complete')
        print({a: v['macro_f1'] for a, v in result['ensemble_metrics'].items()}, flush=True)
        if args.publish:
            publish(files)


if __name__ == '__main__':
    main()
