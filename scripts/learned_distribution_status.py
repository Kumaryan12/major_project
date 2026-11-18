"""Read-only local PID-checked progress; Ctrl-C stops only this monitor."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import yaml
from supervised_source_status import command_for_pid


def read(p):return json.loads(p.read_text()) if p.is_file() else {}


def status(config):
    base=Path(config['output_dir']);root=base/'complete';runner=read(base/'runner.json');saved=read(root/'progress.json');audit=read(base/'teacher_audit/progress.json');summary=read(root/'summary.json')
    command=command_for_pid(saved.get('pid'));rc=command_for_pid(runner.get('pid'));ac=command_for_pid(audit.get('pid'))
    active=bool(command and any(x in command for x in ('mi_localization.learned_distribution_study','mi_localization.learned_distribution_report')))
    g=config['outer_splits']*len(config['projection_seeds']);f=g*(config['inner_splits']*len(config['ridge_alphas'])+1);ctx=g*(config['inner_splits']+1)
    fits=list((root/'review/fits').glob('*.json'));refs=list((root/'review/baseline_refs').glob('*.json'));contexts=list((root/'review/contexts').glob('*.json'))
    lines=[f"Frozen learned distributions: {runner.get('state','not started')} | matching local runner: {bool(rc and 'run_learned_distribution.py' in rc)}",
           f"Phase: {runner.get('phase','unavailable')} | {saved.get('state','not started')} | matching local worker: {active}",
           f"Teacher audit: {audit.get('teacher_models',0)}/{ctx} | matching local audit: {bool(ac and '--audit' in ac and 'learned_distribution_study' in ac)}",
           f"Saved new classifier audits: {len(fits)}/{3*f} | baseline references: {len(refs)}/{f}",
           f"Saved outer models: {sum(p.name.startswith('outer_') for p in fits)}/{3*g} new + {sum(p.name.startswith('outer_') for p in refs)}/{g} reused",
           f"Saved full feature contexts: {len(contexts)}/{ctx}",f"Updated UTC: {saved.get('updated_utc','unavailable')}"]
    if 'projection_seed' in saved:lines.append(f"Current seed {saved['projection_seed']}, outer {saved.get('outer_fold')}, inner {saved.get('inner_fold')}")
    if 'records_complete' in saved:lines.append(f"Current context: {saved['records_complete']}/{saved['context_records']} recordings, {saved['context_beats_complete']} original beats processed")
    if active and runner.get('phase') in ('embedding_extraction_and_ridge','independent_embedding_replay') and 'outer_fold' in saved:
        src=yaml.safe_load(Path(config['source_config']).read_text());upstream=Path(src['output_dir'])/'complete/review/contexts';parts=[]
        for fold in range(1,config['outer_splits']+1):
            for seed in config['projection_seeds']:
                for inner in [*range(1,config['inner_splits']+1),0]:
                    a=read(upstream/f'context_{fold:02d}_s{seed}_i{inner}.json')
                    parts.append(((fold,seed,inner),a.get('state',{}).get('original_beats_covered',0)))
        key=(saved['outer_fold'],saved['projection_seed'],saved['inner_fold'])
        if key in [x[0] for x in parts] and all(n>0 for _,n in parts):
            i=next(i for i,x in enumerate(parts) if x[0]==key);current=saved.get('context_beats_complete',parts[i][1] if saved.get('state') in ('fitting_classifiers','verifying') else 0)
            done=sum(n for _,n in parts[:i])+min(current,parts[i][1]);total=sum(n for _,n in parts)
            elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(runner['updated_utc'])).total_seconds()/3600
            if done>0:lines.append(f"Current phase approximately {100*done/total:.1f}% through beat work")
            if done>=.05*total:lines.append(f"Rough remaining {elapsed*(total-done)/done:.1f} h for this phase only (startup/overheads vary)")
    lines+=[f"Validated report recorded: {summary.get('complete') is True}",'Encoder weights stay frozen. Partial counters/smoke/audit are not final patient macro F1. Independent full embedding replay follows fitting.']
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/learned_distribution_v1.yaml');p.add_argument('--watch',action='store_true');args=p.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    try:
        while True:
            if args.watch:print('\033[2J\033[H',end='')
            print(status(config),flush=True)
            if not args.watch:break
            time.sleep(10)
    except KeyboardInterrupt:pass
