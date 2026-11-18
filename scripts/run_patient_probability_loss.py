"""Train and independently restore the matched ResNet loss comparison."""
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


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_results(base):
    root = base / 'complete'
    summary = read(root / 'summary.json')
    scope = read(root / 'study_manifest.json')
    evidence = read(root / 'verification_manifest.json')
    required = (summary['complete'] is True and summary['outer_test_patients_used'] is True
                and summary['restored_models'] == len(evidence['fit_audit_hashes']) == 90
                and summary['matched_fold_seed_groups'] == 30
                and summary['study_fingerprint'] == scope['fingerprint'] == evidence['fingerprint']
                and summary['verification_manifest_sha256'] == sha(root / 'verification_manifest.json')
                and summary['report_sha256'] == sha(root / 'REPORT.md'))
    if not required:
        raise ValueError('Full independent checkpoint replay is required before publication')
    return [root / n for n in ('summary.json', 'REPORT.md', 'study_manifest.json', 'verification_manifest.json')]


def publish(files):
    if subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() != 'main':
        raise RuntimeError('Verified reports saved locally; selected branch is not main')
    if subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip():
        raise RuntimeError('Verified reports saved locally; Git index contains other work')
    subprocess.run(['git', 'add', '--', *map(str, files)], check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet']).returncode == 1:
        subprocess.run(['git', 'commit', '-m', 'Publish verified patient probability loss comparison'], check=True)
    subprocess.run(['git', 'push', 'origin', 'HEAD:main'], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/patient_probability_loss_v1.yaml')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    config = yaml.safe_load(Path(args.config).read_text())
    base = Path(config['output_dir'])
    command = [sys.executable, '-m', 'mi_localization.patient_probability_loss', '--config', args.config]
    with acquire_lock(base / 'runner.lock'):
        if args.smoke:
            subprocess.run([*command, '--smoke'], check=True)
            subprocess.run([*command, '--smoke', '--verify'], check=True)
            return
        smoke = read(base / 'smoke' / 'summary.json')
        if (smoke['complete'] is not True or smoke['restored_models'] != 3
                or smoke['outer_test_patients_used'] is not False):
            raise ValueError('Require the verified three-arm inner smoke')
        subprocess.run([*command, '--smoke', '--verify'], check=True)
        subprocess.run(command, check=True)
        subprocess.run([*command, '--verify'], check=True)
        files = verified_results(base)
        if args.publish:
            publish(files)


if __name__ == '__main__':
    main()
