"""Run paired STAFF III preparation, source training and frozen target replay."""
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
PUBLIC = ('REPORT.md', 'summary.json', 'study_manifest.json', 'source_summary.json',
          'source_manifest.json', 'verification_manifest.json')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(files, message):
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip() != 'main':
        raise RuntimeError('Verified aggregate outputs are local. Publication requires main.')
    if subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip():
        raise RuntimeError('Verified aggregate outputs are local. Git index contains work.')
    subprocess.run(['git','add','--',*map(str,files)],check=True)
    if subprocess.run(['git','diff','--cached','--quiet']).returncode == 1:
        subprocess.run(['git','commit','-m',message],check=True)
    subprocess.run(['git','push','origin','HEAD:main'],check=True)


def run_stage(module, config, *flags):
    print(module, ' '.join(flags), flush=True)
    subprocess.run([sys.executable,'-m','mi_localization.'+module,*flags,'--config',config],check=True)


def verified_bank(config, destination):
    root = Path(config['bank_dir'])
    receipt = root / 'receipt.json'
    if read(root / 'verified.json')['receipt_sha256'] != sha(receipt):
        raise ValueError('Require the independently verified source bank')
    value = read(receipt)
    result = {key:value[key] for key in ('timing_status','qc','patient_support','waveform_methods',
                                        'recording_copy_screen','source_support_passed')}
    result['bank_receipt_sha256'] = sha(receipt)
    result['verified'] = True
    destination.parent.mkdir(parents=True,exist_ok=True)
    content = json.dumps(result,indent=2)+'\n'
    if destination.exists() and destination.read_text() != content:
        raise ValueError('Verified QC summary changed')
    temp = destination.with_suffix('.partial')
    temp.write_text(content)
    temp.replace(destination)
    return result


def verified_target(root):
    result = read(root/'summary.json')
    scope = read(root/'study_manifest.json')
    schedules = read(root/'verification_manifest.json')
    source = read(root/'source_summary.json')
    if (result['complete'] is not True or result['outer_test_patients_used'] is not True
            or result['restored_models'] != len(schedules['audits']) or result['restored_models'] != 90
            or result['matched_groups'] != 30 or result['study_fingerprint'] != scope['fingerprint']
            or schedules['study_fingerprint'] != scope['fingerprint']
            or result['report_sha256'] != sha(root/'REPORT.md')
            or result['verification_manifest_sha256'] != sha(root/'verification_manifest.json')
            or scope['source_summary_sha256'] != sha(root/'source_summary.json')
            or source['source_scope_sha256'] != sha(root/'source_manifest.json')
            or source['models'] != 2 or not source['recording_copy_screen']['passed']):
        raise ValueError('Require all independently restored source and target results')
    return result, [root/name for name in PUBLIC]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/staffiii_pair_transfer_v1.yaml')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--bank-only',action='store_true')
    mode.add_argument('--publish',action='store_true')
    args = p.parse_args()
    os.chdir(ROOT)
    c = yaml.safe_load(Path(args.config).read_text())
    out = Path(c['output_dir'])
    bank_output = out/'bank_summary.json'
    with acquire_lock(out/'runner.lock'):
        if (out/'complete'/'summary.json').is_file():
            scores, files = verified_target(out/'complete')
            verified_bank(c, bank_output)
            if args.publish:
                publish([bank_output,*files],'Publish verified paired STAFF III transfer')
            return
        for phase in ('prepare','verify'):
            run_stage('staffiii_pair_bank',args.config,phase)
        bank = verified_bank(c,bank_output)
        if args.bank_only or not (bank['source_support_passed'] and bank['recording_copy_screen']['passed']):
            print('Source bank complete. Training gates:',bank['source_support_passed'],bank['recording_copy_screen']['passed'],flush=True)
            if args.publish:
                publish([bank_output],'Publish verified STAFF III source feasibility')
            return
        for smoke in (True, False):
            flags = ['--smoke'] if smoke else []
            for module in ('staffiii_pair_pretraining','staffiii_pair_transfer'):
                for verify in (False,True):
                    run_stage(module,args.config,*flags,*(('--verify',) if verify else ()))
            if smoke:
                check = read(out/'smoke'/'summary.json')
                if check['restored_models'] != 3 or check['outer_test_patients_used'] is not False:
                    raise ValueError('Require all three verified inner-only smoke fits')
        result, files = verified_target(out/'complete')
        print({a:m['macro_f1'] for a,m in result['ensemble_metrics'].items()},flush=True)
        if args.publish:
            publish([bank_output,*files],'Publish verified paired STAFF III transfer')


if __name__ == '__main__':
    main()
