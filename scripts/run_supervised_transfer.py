"""Verify prerequisites, run all matched target arms, restore/report, publish, then stop."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from run_supervised_source_pilot import exclusive_lock


def preflight(config):
    from mi_localization import supervised_transfer as study
    root=Path(config['output_dir'])/'complete';receipt=root/'review/preflight.json'
    data=study.load_inputs(config);scope=study.make_scope(config,data,list(range(1,config['outer_splits']+1)))
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if (saved['scope']!=scope or not saved['source_models_independently_verified'] or not saved['expanded_screen_independently_verified']):
            raise ValueError('Verified preflight scope changed; choose a new namespace')
        return
    commands=[('source_models','mi_localization.supervised_source_train',config['source_config']),
              ('expanded_recordings','mi_localization.supervised_source_overlap',config['overlap_config'])]
    for label,module,path in commands:
        study.progress(root,'preflight_'+label)
        subprocess.run([sys.executable,'-u','-m',module,'--config',path,'--verify'],check=True)
    # Detect any change during prerequisite verification.
    current=study.make_scope(config,study.load_inputs(config),list(range(1,config['outer_splits']+1)))
    if current!=scope: raise ValueError('Prerequisite inputs/code changed during verification')
    study.save_json(receipt,dict(scope=scope,source_models_independently_verified=True,expanded_screen_independently_verified=True))
    study.progress(root,'preflight_passed',completed_fits=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/supervised_transfer_v1.yaml')
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--publish',action='store_true');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());root=Path(config['output_dir'])/'complete'
    with exclusive_lock(root/'review/launcher.lock'):
        preflight(config)
        if args.preflight_only: return
        try:
            subprocess.run([sys.executable,'-u','-m','mi_localization.supervised_transfer','--config',args.config],check=True)
            subprocess.run([sys.executable,'-u','-m','mi_localization.supervised_transfer_report','--config',args.config],check=True)
            summary=json.loads((root/'summary.json').read_text())
            if not summary['complete'] or summary['next_experiment_automatically_launched']:
                raise RuntimeError('Matched-transfer stopping contract failed')
            from mi_localization.supervised_transfer import progress
            progress(root,'complete_independently_validated',completed_fits=summary['restored_fit_count'],next_experiment_queued=False)
            if args.publish:
                names=['study_manifest.json','summary.json','REPORT.md','per_seed.csv','ensemble_metrics.csv',
                       'selection_summary.csv','paired_bootstrap.json','comparison.png','report_manifest.json','progress.json']
                paths=[str(root/n) for n in names]
                subprocess.run(['git','add','--']+paths,check=True)
                diff=subprocess.run(['git','diff','--cached','--quiet','--']+paths)
                if diff.returncode==1:
                    subprocess.run(['git','commit','-m','Publish independently restored matched supervised-source transfer','--']+paths,check=True)
                    subprocess.run(['git','push','origin','HEAD:main'],check=True)
                elif diff.returncode!=0: raise RuntimeError('Cannot inspect matched-transfer aggregate changes')
            print('Matched supervised-source transfer verified and stopped. No next experiment launched.',flush=True)
        except Exception as error:
            from mi_localization.supervised_transfer import progress
            progress(root,'stopped_with_error',error_type=type(error).__name__,error=str(error))
            raise


if __name__=='__main__': main()
