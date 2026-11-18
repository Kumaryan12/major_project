"""Run one authorized learned-feature pooling study, verify/publish, then stop."""
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
    s=json.loads((root/'summary.json').read_text());scope=json.loads((root/'study_manifest.json').read_text());g=config['outer_splits']*len(config['projection_seeds']);f=g*(config['inner_splits']*len(config['ridge_alphas'])+1);ctx=g*(config['inner_splits']+1)
    return (s.get('complete') is True and s.get('study_fingerprint')==scope.get('study_fingerprint') and scope.get('config')==config
            and s.get('fits')==s.get('all_models_restored')==4*f and s.get('new_models_restored')==3*f and s.get('source_models_restored')==f
            and s.get('outer_models_restored')==4*g and s.get('feature_contexts_reconstructed')==s.get('matched_context_checks')==ctx
            and s.get('teacher_models_restored')==s.get('teacher_query_predictions_reproduced')==ctx and s.get('encoder_training_steps')==0
            and all(s.get(k) is True for k in ('all_original_beats_replayed','original_cohort_unchanged','source_files_unchanged','baseline_metrics_exact'))
            and all(Path(k).is_file() and sha(Path(k))==v for k,v in scope.get('code',{}).items())
            and s.get('report_sha256')==sha(root/'REPORT.md') and s.get('verification_manifest_sha256')==sha(root/'verification_manifest.json'))


def publish(config):
    if not verified_ready(config):raise RuntimeError('Require complete independently verified learned distributions')
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='main':raise RuntimeError('Publication requires main')
    if subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip():raise RuntimeError('Unrelated staged files present')
    base=Path(config['output_dir']);root=base/'complete';files=[base/'teacher_audit/summary.json',base/'benchmark.json',base/'smoke/summary.json']+[root/n for n in ('summary.json','REPORT.md','study_manifest.json','verification_manifest.json')]
    files+=sorted(root.glob('confusion_patient_*.csv'))+sorted(root.glob('confusion_patient_*.png'))
    subprocess.run(['git','add','--',*map(str,files)],check=True)
    if subprocess.run(['git','diff','--cached','--quiet']).returncode:subprocess.run(['git','commit','-m','Publish independently replayed frozen learned-distribution comparison'],check=True)
    for attempt in range(3):
        if subprocess.run(['git','push','origin','HEAD:main']).returncode==0:return
        if attempt<2:time.sleep(10)
    raise RuntimeError('Verified results committed locally; push failed')


def main():
    from run_residual_context import acquire_lock
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/learned_distribution_v1.yaml');p.add_argument('--publish',action='store_true');args=p.parse_args();os.chdir(ROOT)
    config=yaml.safe_load(Path(args.config).read_text());base=Path(config['output_dir']);lock=acquire_lock(base/'runner.lock')
    def save(state,**kw):
        target=base/'runner.json';temp=target.with_suffix('.partial.json');temp.write_text(json.dumps(dict(pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),state=state,**kw),indent=2));temp.replace(target)
    verified=False
    try:
        for module,extra in [('mi_localization.learned_distribution_study',['--audit']),('mi_localization.learned_distribution_study',[]),('mi_localization.learned_distribution_report',[])]:
            save('running',module=module,phase='teacher_audit' if extra else 'independent_embedding_replay' if module.endswith('report') else 'embedding_extraction_and_ridge')
            subprocess.run([sys.executable,'-m',module,'--config',args.config,*extra],check=True)
        verified=verified_ready(config)
        if not verified:raise RuntimeError('Learned feature report did not establish complete verification')
        save('complete_verified')
        if args.publish:publish(config);save('complete_published')
    except Exception as error:save('publication_failed' if verified else 'failed',error=str(error),training_verified=verified);raise
    finally:lock.close()


if __name__=='__main__':main()
