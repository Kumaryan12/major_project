"""Run the authorized cache-only kernel study, verify/publish aggregates, stop."""
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
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verified_ready(config):
    root=Path(config['output_dir'])/'complete'
    if not all((root/n).is_file() for n in ('summary.json','REPORT.md','study_manifest.json','verification_manifest.json')):return False
    s=json.loads((root/'summary.json').read_text());scope=json.loads((root/'study_manifest.json').read_text())
    groups=config['outer_splits']*len(config['projection_seeds']);reps=len(config['representations']);linear=groups*reps*(config['inner_splits']*len(config['ridge_alphas'])+1)
    nonlinear=groups*reps*(config['inner_splits']*len(config['ridge_alphas'])*len(config['nonlinear_weights'])+1)
    return (s.get('complete') is True and s.get('study_fingerprint')==scope.get('study_fingerprint') and scope.get('config')==config
            and s.get('fits')==s.get('all_models_restored')==linear+nonlinear and s.get('source_equivalence_checks')==linear
            and s.get('outer_models_restored')==groups*reps*2
            and s.get('kernel_contexts_reconstructed')==s.get('matched_context_checks')==groups*(config['inner_splits']+1)*reps
            and all(s.get(k) is True for k in ('source_all_arm_metrics_reproduced','cache_only_source_chain_verified','original_cohort_unchanged'))
            and s.get('new_waveform_extractions')==0
            and all(Path(k).is_file() and sha(Path(k))==v for k,v in scope.get('code',{}).items())
            and s.get('report_sha256')==sha(root/'REPORT.md') and s.get('verification_manifest_sha256')==sha(root/'verification_manifest.json'))


def publish(config):
    if not verified_ready(config):raise RuntimeError('Require complete independently restored kernel report')
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='main':raise RuntimeError('Publication requires main')
    if subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip():raise RuntimeError('Unrelated staged files present')
    base=Path(config['output_dir']);root=base/'complete'
    files=[base/'preflight/summary.json',base/'smoke/summary.json']+[root/n for n in ('summary.json','REPORT.md','study_manifest.json','verification_manifest.json')]
    files+=sorted(root.glob('confusion_patient_*.csv'))+sorted(root.glob('confusion_patient_*.png'))
    subprocess.run(['git','add','--',*map(str,files)],check=True)
    if subprocess.run(['git','diff','--cached','--quiet']).returncode:subprocess.run(['git','commit','-m','Publish independently restored nonlinear distribution readouts'],check=True)
    for attempt in range(3):
        if subprocess.run(['git','push','origin','HEAD:main']).returncode==0:return
        if attempt<2:time.sleep(10)
    raise RuntimeError('Verified results committed locally; GitHub push failed')


def main():
    from run_residual_context import acquire_lock
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/distribution_kernel_v1.yaml');p.add_argument('--publish',action='store_true');args=p.parse_args();os.chdir(ROOT)
    config=yaml.safe_load(Path(args.config).read_text());base=Path(config['output_dir']);lock=acquire_lock(base/'runner.lock')
    def save(state,**kw):
        path=base/'runner.json';temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(dict(pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),state=state,**kw),indent=2));temp.replace(path)
    verified=False
    try:
        # The engineer-reviewed smoke is required before automatic full fitting.
        smoke=json.loads((base/'smoke/summary.json').read_text())
        if smoke.get('complete') is not True or smoke.get('outer_test_patients_used') is not False:raise RuntimeError('Require inner-only engineering smoke')
        for module,extra in [('mi_localization.distribution_kernel_study',['--preflight']),('mi_localization.distribution_kernel_study',[]),('mi_localization.distribution_kernel_report',[])]:
            save('running',module=module,phase='linear_equivalence' if extra else 'independent_report' if module.endswith('report') else 'kernel_fitting')
            subprocess.run([sys.executable,'-m',module,'--config',args.config,*extra],check=True)
        verified=verified_ready(config)
        if not verified:raise RuntimeError('Kernel report did not establish complete coverage')
        save('complete_verified')
        if args.publish:publish(config);save('complete_published')
    except Exception as error:save('publication_failed' if verified else 'failed',error=str(error),training_verified=verified);raise
    finally:lock.close()


if __name__=='__main__':main()
