"""Run and independently replay the R-relative convolution pooling study."""
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
PUBLIC = ('REPORT.md', 'summary.json', 'study_manifest.json', 'verification_manifest.json')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_verified(root):
    summary = read(root / 'summary.json')
    manifest = read(root / 'study_manifest.json')
    evidence = read(root / 'verification_manifest.json')
    if (summary['complete'] is not True or summary['study_fingerprint'] != manifest['study_fingerprint']
            or evidence['study_fingerprint'] != manifest['study_fingerprint']
            or summary['reconstructed_contexts'] != len(evidence['context_audits']) or summary['reconstructed_contexts'] != 120
            or summary['restored_baseline_models'] != 210 or summary['replayed_new_models'] != 630
            or summary['report_sha256'] != sha(root / 'REPORT.md')
            or summary['verification_manifest_sha256'] != sha(root / 'verification_manifest.json')):
        raise ValueError('Full verified evidence is required before publication')
    return [root / filename for filename in PUBLIC]


def publish(files):
    if subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() != 'main':
        raise RuntimeError('Verified reports saved locally. Publication requires main.')
    if subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip():
        raise RuntimeError('Verified reports saved locally. Git index contains other work.')
    subprocess.run(['git', 'add', '--', *map(str, files)], check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet']).returncode == 1:
        subprocess.run(['git', 'commit', '-m', 'Publish verified R-relative convolution comparison'], check=True)
    subprocess.run(['git', 'push', 'origin', 'HEAD:main'], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/r_relative_convolution_v1.yaml')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    config = yaml.safe_load(Path(args.config).read_text())
    base = Path(config['output_dir'])
    command = [sys.executable, '-m', 'mi_localization.r_relative_convolution_study', '--config', args.config]
    with acquire_lock(base / 'runner.lock'):
        if args.smoke:
            subprocess.run([*command, '--smoke'], check=True)
            subprocess.run([*command, '--smoke', '--verify'], check=True)
            return
        smoke = read(base / 'smoke' / 'summary.json')
        if smoke['complete'] is not True or smoke['outer_test_patients_used'] is not False or smoke['replayed_new_models'] != 18:
            raise ValueError('Require the verified inner-only smoke')
        subprocess.run([*command, '--smoke', '--verify'], check=True)
        subprocess.run(command, check=True)
        subprocess.run([*command, '--verify'], check=True)
        files = require_verified(base / 'complete')
        if args.publish:
            publish(files)


if __name__ == '__main__':
    main()
