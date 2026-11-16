"""Run and independently verify source-only learning, publish aggregates, then stop."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from supervised_source_status import command_for_pid


@contextmanager
def exclusive_lock(path):
    """OS-held lock releases on crash, avoiding stale-PID lock deletion."""
    path.parent.mkdir(parents=True,exist_ok=True)
    handle=path.open('a+b')
    try:
        if os.name=='posix':
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        else:
            import msvcrt
            if path.stat().st_size==0: handle.write(b'0');handle.flush()
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        yield
    finally:
        handle.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/supervised_source_pilot_v1.yaml')
    p.add_argument('--wait-for-bank',action='store_true');p.add_argument('--publish',action='store_true');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());root=Path(config['output_dir'])
    with exclusive_lock(root/'review/launcher.lock'):
        progress=root/'progress.json'
        if args.wait_for_bank:
            while not (Path(config['bank_dir'])/'manifest.json').exists():
                saved=json.loads(progress.read_text()) if progress.exists() else {}
                command=command_for_pid(saved.get('pid'))
                if 'mi_localization.supervised_source_bank' not in command:
                    raise RuntimeError('Preparation is not running and the bank is incomplete; inspect its log')
                print('Waiting for the existing source-bank preparation; no duplicate preparation launched',flush=True)
                time.sleep(20)
        elif progress.exists():
            saved=json.loads(progress.read_text());command=command_for_pid(saved.get('pid'))
            if any(s in command for s in ['mi_localization.supervised_source_bank','mi_localization.supervised_source_train']):
                raise RuntimeError('A source worker is already active; monitor it or explicitly wait for its bank')
        command=[sys.executable,'-u','-m','mi_localization.supervised_source_train','--config',args.config]
        subprocess.run(command,check=True)
        subprocess.run(command+['--verify'],check=True)
        summary=json.loads((root/'summary.json').read_text())
        if not summary['complete'] or summary['target_evaluated'] or summary['next_stage_automatically_launched']:
            raise RuntimeError('Source-only stopping contract not satisfied')
        if args.publish:
            names=['cohort_manifest.json','cohort_summary.json','bank_summary.json','training_manifest.json','report_manifest.json',
                   'summary.json','REPORT.md','metrics_by_code.csv','training_history.csv','learning_curves.png','progress.json']
            subprocess.run(['git','add','--']+[str(root/n) for n in names],check=True)
            diff=subprocess.run(['git','diff','--cached','--quiet','--']+[str(root/n) for n in names])
            if diff.returncode==1:
                subprocess.run(['git','commit','-m','Publish independently verified source-only supervised pilot','--']+[str(root/n) for n in names],check=True)
                subprocess.run(['git','push','origin','HEAD:main'],check=True)
            elif diff.returncode!=0: raise RuntimeError('Cannot inspect staged source results')
        print('Source-only training/verification complete. Stopped; no target transfer or next experiment launched.',flush=True)


if __name__=='__main__': main()
