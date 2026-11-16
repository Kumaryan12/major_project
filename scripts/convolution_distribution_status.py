"""Read-only local PID/command-checked status; Ctrl-C stops only this monitor."""
import argparse
import json
from pathlib import Path
import time
import yaml
from supervised_source_status import command_for_pid


def read(path):return json.loads(path.read_text()) if path.is_file() else {}


def status(config):
    base=Path(config['output_dir']);root=base/'complete'
    runner=read(base/'runner.json');saved=read(root/'progress.json');preflight=read(base/'source_preflight/progress.json')
    command=command_for_pid(saved.get('pid'));pre_command=command_for_pid(preflight.get('pid'));run_command=command_for_pid(runner.get('pid'))
    expected=('mi_localization.convolution_distribution_study','mi_localization.convolution_distribution_report')
    alive=bool(command and any(x in command for x in expected))
    pre_alive=bool(pre_command and 'mi_localization.convolution_distribution_study' in pre_command and '--preflight' in pre_command)
    run_alive=bool(run_command and 'run_convolution_distribution.py' in run_command)
    groups=config['outer_splits']*len(config['projection_seeds']);source=groups*(config['inner_splits']*len(config['ridge_alphas'])+1)
    fits=list((root/'review/fits').glob('*.json'));refs=list((root/'review/baseline_refs').glob('*.json'));contexts=list((root/'review/contexts').glob('*.json'))
    lines=[f'Convolution distributions: {base}',f"Runner: {runner.get('state','not started')} | matching local command: {run_alive}",
           f"Full study: {saved.get('state','not started')} | matching local worker: {alive}",
           f"Source preflight: {preflight.get('state','not started')} | matching local preflight: {pre_alive} | restored {preflight.get('models_restored',0)}/{source}",
           f'Saved new classifier audits: {len(fits)}/{3*source} | frozen baseline references: {len(refs)}/{source}',
           f"Saved outer classifiers: {sum(p.name.startswith('outer_') for p in fits)}/{3*groups} new + {sum(p.name.startswith('outer_') for p in refs)}/{groups} reused",
           f"Saved complete feature contexts: {len(contexts)}/{groups*(config['inner_splits']+1)}",f"Updated UTC: {saved.get('updated_utc','unavailable')}"]
    if 'projection_seed' in saved:lines.append(f"Current: seed {saved['projection_seed']}, outer {saved.get('outer_fold')}, inner {saved.get('inner_fold')}, {saved.get('arm','shared features')}")
    if 'records_complete' in saved:lines.append(f"Current context: {saved['records_complete']}/{saved['context_records']} recordings, {saved['context_beats_complete']} beats processed; {saved.get('coverage_mode')}")
    summary=read(root/'summary.json')
    lines += [f"Validated report recorded: {summary.get('complete') is True and summary.get('all_original_beats_replayed') is True}",
              'Saved counters/smoke/preflight are not completed performance; the independent full replay follows fitting.']
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/convolution_distribution_v1.yaml');p.add_argument('--watch',action='store_true');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    try:
        while True:
            if args.watch:print('\033[2J\033[H',end='')
            print(status(config),flush=True)
            if not args.watch:break
            time.sleep(10)
    except KeyboardInterrupt:pass
