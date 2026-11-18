"""Run, replay, and optionally publish the matched probabilistic subtype study."""
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


def verify_report(root):
    final = read(root / 'summary.json')
    scope = read(root / 'study_manifest.json')
    evidence = read(root / 'verification_manifest.json')
    if (final['complete'] is not True or final['restored_procedures'] != 600
            or final['matched_contexts'] != len(evidence['contexts']) or final['matched_contexts'] != 120
            or final['study_fingerprint'] != scope['fingerprint'] or evidence['study_fingerprint'] != scope['fingerprint']
            or final['report_sha256'] != sha(root / 'REPORT.md')
            or final['verification_manifest_sha256'] != sha(root / 'verification_manifest.json')):
        raise ValueError('Require all restored procedures before publication')
    return [root / filename for filename in PUBLIC]


def publish(files):
    if subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() != 'main':
        raise RuntimeError('Verified report is saved locally. Publication requires main.')
    if subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip():
        raise RuntimeError('Verified report is saved locally. Git index contains staged work.')
    subprocess.run(['git', 'add', '--', *map(str, files)], check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet']).returncode == 1:
        subprocess.run(['git', 'commit', '-m', 'Publish verified probabilistic MI subtype comparison'], check=True)
    subprocess.run(['git', 'push', 'origin', 'HEAD:main'], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/probabilistic_subtype_v1.yaml')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    config = yaml.safe_load(Path(args.config).read_text())
    base = Path(config['output_dir'])
    cmd = [sys.executable, '-m', 'mi_localization.probabilistic_subtype', '--config', args.config]
    with acquire_lock(base / 'runner.lock'):
        if args.smoke:
            subprocess.run([*cmd, '--smoke'], check=True)
            subprocess.run([*cmd, '--smoke', '--verify'], check=True)
            return
        smoke = read(base / 'smoke' / 'summary.json')
        if smoke['complete'] is not True or smoke['outer_test_patients_used'] is not False or smoke['restored_procedures'] != 18:
            raise ValueError('Require the verified inner-only smoke')
        subprocess.run([*cmd, '--smoke', '--verify'], check=True)
        subprocess.run(cmd, check=True)
        subprocess.run([*cmd, '--verify'], check=True)
        files = verify_report(base / 'complete')
        if args.publish:
            publish(files)


if __name__ == '__main__':
    main()
