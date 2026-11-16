"""Wait for native source learning; verify it; screen expanded recordings; stop."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from run_supervised_source_pilot import exclusive_lock
from supervised_source_status import command_for_pid


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def update(root,state,**extra):
    save(root/'queue_progress.json',dict(state=state,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**extra))


def queue_scope(config_path,config):
    source=yaml.safe_load(Path(config['source_config']).read_text())
    package=Path(__file__).resolve().parents[1]/'src/mi_localization'
    names=['supervised_source_overlap.py','overlap_screen.py','supervised_source_train.py',
           'supervised_source_pilot_report.py','supervised_source_bank.py','neural.py']
    return dict(config=config,config_sha256=digest(config_path),source_config_sha256=digest(config['source_config']),
                script_sha256=digest(__file__),code_sha256={n:digest(package/n) for n in names},
                source_output_dir=source['output_dir'],target_classification_queued=False,
                stop_after='independently verified expanded recording-overlap audit and aggregate publication')


def wait_for_pilot(config,root,launcher_pid):
    source=yaml.safe_load(Path(config['source_config']).read_text());pilot=Path(source['output_dir'])
    while True:
        parent=command_for_pid(launcher_pid)
        saved=json.loads((pilot/'progress.json').read_text()) if (pilot/'progress.json').exists() else {}
        worker=command_for_pid(saved.get('pid'))
        parent_active='scripts/run_supervised_source_pilot.py' in parent
        worker_active=any(s in worker for s in ['mi_localization.supervised_source_bank','mi_localization.supervised_source_train'])
        if not parent_active and not worker_active: break
        update(root,'waiting_for_source_pilot',source_state=saved.get('state','unknown'),
               source_launcher_active=parent_active,source_worker_active=worker_active)
        time.sleep(20)
    required=[pilot/'summary.json',pilot/'report_manifest.json',pilot/'training_manifest.json',Path(source['bank_dir'])/'manifest.json']
    if any(not p.is_file() for p in required):
        raise RuntimeError('Source pilot stopped without complete artifacts; no audit or target training started')
    summary=json.loads((pilot/'summary.json').read_text())
    expected={(obj,seed) for obj in source['objectives'] for seed in source['training_seeds']}
    if (summary.get('complete') is not True or summary.get('source_only') is not True
            or summary.get('target_evaluated') is not False
            or len(summary.get('runs',[]))!=len(expected)
            or {(r['objective'],r['seed']) for r in summary['runs']}!=expected):
        raise RuntimeError('Declared source objectives/seeds did not complete; audit remains blocked')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/supervised_source_overlap_v1.yaml')
    p.add_argument('--wait-for-pilot-pid',type=int);p.add_argument('--publish',action='store_true');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());root=Path(config['output_dir'])
    scope=queue_scope(args.config,config)
    with exclusive_lock(root/'review/queue.lock'):
        guard=root/'queue_manifest.json'
        if guard.exists() and json.loads(guard.read_text())!=scope: raise ValueError('Queued audit code/config changed; use a new namespace')
        if not guard.exists(): save(guard,scope)
        try:
            wait_for_pilot(config,root,args.wait_for_pilot_pid)
            if queue_scope(args.config,config)!=scope: raise ValueError('Queued code/config changed while waiting')
            update(root,'verifying_completed_source_pilot')
            subprocess.run([sys.executable,'-u','-m','mi_localization.supervised_source_train','--config',config['source_config'],'--verify'],check=True)
            command=[sys.executable,'-u','-m','mi_localization.supervised_source_overlap','--config',args.config]
            update(root,'running_expanded_overlap_screen');subprocess.run(command,check=True)
            update(root,'independently_replaying_expanded_screen');subprocess.run(command+['--verify'],check=True)
            summary=json.loads((root/'summary.json').read_text())
            if not summary['complete'] or summary['target_classification_launched'] or summary['next_experiment_automatically_launched']:
                raise RuntimeError('Expanded-audit stopping contract failed')
            update(root,'complete_verified_stop' if summary['recording_screen_passed'] else 'complete_verified_requires_review',
                   recording_screen_passed=summary['recording_screen_passed'],target_classification_queued=False)
            if args.publish:
                names=['queue_manifest.json','queue_progress.json','screen_manifest.json','output_manifest.json','summary.json','REPORT.md','progress.json']
                paths=[str(root/n) for n in names]
                subprocess.run(['git','add','--']+paths,check=True)
                diff=subprocess.run(['git','diff','--cached','--quiet','--']+paths)
                if diff.returncode==1:
                    subprocess.run(['git','commit','-m','Publish independently replayed expanded source recording audit','--']+paths,check=True)
                    subprocess.run(['git','push','origin','HEAD:main'],check=True)
                elif diff.returncode!=0: raise RuntimeError('Cannot inspect queued-audit result changes')
            print('Expanded overlap audit finished. Stopped; no target transfer or further study launched.',flush=True)
        except Exception as error:
            update(root,'stopped_with_error',error_type=type(error).__name__,error=str(error),target_classification_queued=False)
            raise


if __name__=='__main__': main()
