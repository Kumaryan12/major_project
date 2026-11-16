"""Run/verify staged morphology-context learning and publish aggregate results."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RESULTS='results/neural_staged_context_v1'


def progress(**values):
    target=ROOT/RESULTS/'queue';target.mkdir(parents=True,exist_ok=True)
    p=target/'progress.json.partial';p.write_text(json.dumps(dict(pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**values),indent=2));p.replace(target/'progress.json')


def verified_ready(root):
    p=root/RESULTS/'complete'
    if not all((p/name).is_file() for name in ['summary.json','REPORT.md','study_manifest.json']):return False
    s=json.loads((p/'summary.json').read_text());scope=json.loads((p/'study_manifest.json').read_text())
    return (s.get('complete') is True and s.get('study_fingerprint')==scope.get('study_fingerprint')
            and s.get('phase_fits')==600 and s.get('all_phase_models_restored')==600 and s.get('outer_models_restored')==150
            and s.get('matched_adaptation_partition_checks')==120 and s.get('all_query_beats_covered') is True)


def main():
    from run_residual_context import acquire_lock
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--publish',action='store_true')
    args=p.parse_args();os.chdir(ROOT)
    lock=acquire_lock(ROOT/'tmp/staged_context_v1.lock');verified=False
    (ROOT/'tmp/staged_context_v1_runner.json').write_text(json.dumps(dict(runner_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),planned_phase_fits=600),indent=2))
    try:
        for module,flags in [('mi_localization.staged_context_study',['--ensure-smoke']),('mi_localization.staged_context_study',[]),('mi_localization.staged_context_report',[])]:
            progress(state='running',module=module,flags=flags)
            subprocess.run([sys.executable,'-m',module,*flags],check=True)
        verified=verified_ready(ROOT)
        if not verified:raise RuntimeError('Staged report did not pass complete verification')
        progress(state='complete_verified')
        if args.publish:
            branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
            changed=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True)
            if branch!='main' or changed:raise RuntimeError('Verified staged results ready; publication stopped because branch/tracked files changed')
            subprocess.run(['git','add','--',RESULTS],check=True)
            if subprocess.run(['git','diff','--cached','--quiet']).returncode:
                subprocess.run(['git','commit','-q','-m','Publish verified staged frozen-context comparison'],check=True)
            for attempt in range(3):
                if subprocess.run(['git','push','origin','HEAD:main']).returncode==0:break
                if attempt==2:raise RuntimeError('Verified staged commit exists locally; GitHub push failed')
                time.sleep(10)
            progress(state='complete_published')
    except Exception as error:
        progress(state='publication_failed' if verified else 'failed',error=str(error),training_verified=verified);raise
    finally:lock.close()


if __name__=='__main__':main()
